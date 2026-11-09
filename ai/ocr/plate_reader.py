"""High-accuracy license plate reader with multi-pass preprocessing and character correction."""
import re
from functools import cached_property
from typing import Any
import cv2
import numpy as np

# Common OCR character confusions: digit <-> letter mappings
CHAR_TO_INT = {"O": "0", "I": "1", "J": "3", "A": "4", "G": "6", "S": "5", "B": "8", "Z": "2", "D": "0"}
INT_TO_CHAR = {"0": "O", "1": "I", "3": "J", "4": "A", "6": "G", "5": "S", "8": "B", "2": "Z"}

# Standard country / watermarks often found on number plates that are not part of plate number
WATERMARKS = {"IND", "INDIA", "EU", "USA", "D", "F", "GB", "NL", "HR", "IR"}


def clean_text(raw_text: str) -> str:
    """Strip out all non-alphanumeric characters and normalize."""
    cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
    # Remove leading watermark if present (e.g. "INDMH12AB1234")
    for wm in ("IND", "INDIA"):
        if cleaned.startswith(wm) and len(cleaned) >= len(wm) + 6:
            cleaned = cleaned[len(wm):]
            break
    return cleaned


def fix_plate_syntax(text: str) -> str:
    """
    Format-aware correction for standard license plates (such as Indian HSRP):
    Format: 2 letters (state) + 2 digits (RTO) + 1-3 letters (series) + 4 digits (number)
    Example: 'DL01AB1234', 'UP16XY9999', 'MH12DE1433'
    """
    if not text:
        return text

    # Exact 10-character standard format: AA 00 AA 0000
    if len(text) == 10:
        chars = list(text)
        # Positions 0, 1 -> Letters
        for i in (0, 1):
            if chars[i] in INT_TO_CHAR:
                chars[i] = INT_TO_CHAR[chars[i]]
        # Positions 2, 3 -> Digits
        for i in (2, 3):
            if chars[i] in CHAR_TO_INT:
                chars[i] = CHAR_TO_INT[chars[i]]
        # Positions 4, 5 -> Letters
        for i in (4, 5):
            if chars[i] in INT_TO_CHAR:
                chars[i] = INT_TO_CHAR[chars[i]]
        # Positions 6, 7, 8, 9 -> Digits
        for i in (6, 7, 8, 9):
            if chars[i] in CHAR_TO_INT:
                chars[i] = CHAR_TO_INT[chars[i]]
        return "".join(chars)

    # 9-character format: AA 0 AA 0000 or AA 00 A 0000
    m9 = re.match(r"^([A-Z0-9]{2})([A-Z0-9]{1,2})([A-Z0-9]{1,2})([A-Z0-9]{4})$", text)
    if m9:
        p1, p2, p3, p4 = m9.groups()
        p1 = "".join(INT_TO_CHAR.get(c, c) for c in p1)
        p2 = "".join(CHAR_TO_INT.get(c, c) for c in p2)
        p3 = "".join(INT_TO_CHAR.get(c, c) for c in p3)
        p4 = "".join(CHAR_TO_INT.get(c, c) for c in p4)
        return p1 + p2 + p3 + p4

    return text


def score_plate_candidate(plate: str, ocr_conf: float) -> float:
    """Compute overall quality score based on length, pattern match, and OCR confidence."""
    length = len(plate)
    if length < 4 or length > 12:
        return 0.0

    score = ocr_conf
    # Bonus for standard plate length (8 to 10 characters)
    if 8 <= length <= 10:
        score += 0.15
    # Bonus if matching typical license plate pattern (letters followed by digits)
    if re.search(r"^[A-Z]{2}[0-9]{1,2}", plate):
        score += 0.2
    if re.search(r"[0-9]{4}$", plate):
        score += 0.15

    return min(1.0, score)


class PlateReader:
    def __init__(self) -> None:
        self.char_to_int = CHAR_TO_INT
        self.int_to_char = INT_TO_CHAR

    @cached_property
    def reader(self):
        import easyocr
        return easyocr.Reader(["en"], gpu=False)

    def _preprocess_variants(self, crop: np.ndarray) -> list[np.ndarray]:
        """Generate high-contrast variants of the cropped license plate image."""
        if crop is None or crop.size == 0:
            return []

        h, w = crop.shape[:2]
        # Upscale if crop is too small for character feature recognition
        target_h, target_w = max(h, 80), max(w, 240)
        if (target_h, target_w) != (h, w):
            resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        else:
            resized = crop.copy()

        variants = [resized]  # Variant 1: Enhanced original color

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Variant 2: CLAHE (Contrast-Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        variants.append(clahe.apply(gray))

        # Variant 3: Bilateral Filter + Otsu Binarization
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(otsu)

        # Variant 4: Inverted Otsu (for light text on dark plates)
        variants.append(cv2.bitwise_not(otsu))

        return variants

    def _extract_from_readtext(self, results: list[Any]) -> tuple[str, float]:
        """Aggregate and sort multiple OCR text boxes from a single pass."""
        if not results:
            return "", 0.0

        # Sort boxes primarily top-to-bottom, secondarily left-to-right
        # box format in EasyOCR: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        def box_sort_key(item):
            box = item[0]
            top_y = min(p[1] for p in box)
            left_x = min(p[0] for p in box)
            return (round(top_y / 20.0), left_x)

        sorted_results = sorted(results, key=box_sort_key)
        tokens = []
        confs = []

        for _, raw_text, conf in sorted_results:
            cleaned = clean_text(raw_text)
            if not cleaned or cleaned in WATERMARKS:
                continue
            tokens.append(cleaned)
            confs.append(float(conf))

        if not tokens:
            return "", 0.0

        combined = "".join(tokens)
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return combined, avg_conf

    def read(self, crop: np.ndarray) -> tuple[str | None, float | None]:
        """
        Run multi-pass OCR on the plate crop, disambiguate characters,
        and select the best candidate.
        """
        if crop is None or crop.size == 0:
            return None, None

        variants = self._preprocess_variants(crop)
        best_plate = None
        best_score = -1.0
        best_conf = 0.0

        for var in variants:
            try:
                raw_results = self.reader.readtext(var, detail=1, paragraph=False)
                raw_plate, conf = self._extract_from_readtext(raw_results)
                if not raw_plate or len(raw_plate) < 4:
                    continue

                fixed_plate = fix_plate_syntax(raw_plate)
                cand_score = score_plate_candidate(fixed_plate, conf)

                if cand_score > best_score:
                    best_score = cand_score
                    best_plate = fixed_plate
                    best_conf = conf

                # Early exit if we find an exceptionally high confidence match
                if best_score >= 0.92 and 8 <= len(fixed_plate) <= 10:
                    break
            except Exception:
                continue

        if best_plate and len(best_plate) >= 4:
            return best_plate, round(float(best_conf), 4)

        return None, None
