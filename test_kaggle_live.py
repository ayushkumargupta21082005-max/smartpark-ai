"""
Live Test Script:
1. Loads 'saisirishan/indian-vehicle-dataset' via kagglehub into a structured Pandas DataFrame.
2. Displays the first records (df.head()) with image path, ground truth plate, and bounding boxes.
3. Runs live ANPR Pipeline inference (YOLO + EasyOCR) on sample images and compares with ground truth.
"""
import torch  # Must be imported first on Windows to initialize runtime DLLs
import os
import glob
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import kagglehub
from ai.anpr_pipeline import ANPRPipeline

def load_kaggle_dataset_to_df(dataset_name: str = "saisirishan/indian-vehicle-dataset") -> tuple[pd.DataFrame, Path]:
    print(f"\n[1/3] Loading dataset from kagglehub: '{dataset_name}'...")
    base_dir = Path(kagglehub.dataset_download(dataset_name))
    print(f"Dataset directory: {base_dir}")

    records = []
    # Search for all XML files in the downloaded dataset
    xml_files = list(base_dir.rglob("*.xml"))
    print(f"Found {len(xml_files)} annotation XML files.")

    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Find matching image file
            filename = root.findtext("filename") or ""
            # Search possible image extensions
            img_path = xml_file.parent / filename
            if not img_path.exists():
                stem = xml_file.stem
                candidates = list(xml_file.parent.glob(f"{stem}.*"))
                img_candidates = [c for c in candidates if c.suffix.lower() in [".jpg", ".jpeg", ".png"]]
                img_path = img_candidates[0] if img_candidates else None

            if not img_path or not img_path.exists():
                continue

            # Size
            size_elem = root.find("size")
            width = int(size_elem.findtext("width")) if size_elem is not None and size_elem.findtext("width") else None
            height = int(size_elem.findtext("height")) if size_elem is not None and size_elem.findtext("height") else None

            # Bounding box & Ground Truth plate name
            for obj in root.findall("object"):
                plate_gt = (obj.findtext("name") or "").strip().upper()
                bndbox = obj.find("bndbox")
                xmin = int(float(bndbox.findtext("xmin"))) if bndbox is not None and bndbox.findtext("xmin") else None
                ymin = int(float(bndbox.findtext("ymin"))) if bndbox is not None and bndbox.findtext("ymin") else None
                xmax = int(float(bndbox.findtext("xmax"))) if bndbox is not None and bndbox.findtext("xmax") else None
                ymax = int(float(bndbox.findtext("ymax"))) if bndbox is not None and bndbox.findtext("ymax") else None

                records.append({
                    "image_name": img_path.name,
                    "image_path": str(img_path),
                    "ground_truth_plate": plate_gt,
                    "source_folder": xml_file.parent.name,
                    "width": width,
                    "height": height,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                })
        except Exception as e:
            continue

    df = pd.DataFrame(records)
    return df, base_dir

def run_live_test(df: pd.DataFrame, num_samples: int = 5):
    print("\n" + "="*80)
    print(f"DATASET LOADED SUCCESSFULLY: Total Annotated Plates = {len(df)}")
    print("="*80)
    print("\nFirst 5 records (df.head()):\n")
    print(df[["image_name", "ground_truth_plate", "source_folder", "xmin", "ymin", "xmax", "ymax"]].head())

    print("\n" + "="*80)
    print(f"[2/3] Initializing SmartPark-AI Live ANPR Pipeline (YOLO + EasyOCR)...")
    print("="*80)
    pipeline = ANPRPipeline()

    print(f"\n[3/3] Running Live Inference on {num_samples} diverse samples from the dataset...\n")

    # Select non-empty plates across different folders
    valid_samples = df[df["ground_truth_plate"].str.len() >= 6]
    sample_rows = valid_samples.drop_duplicates(subset=["source_folder"]).head(num_samples)
    if len(sample_rows) < num_samples:
        sample_rows = valid_samples.head(num_samples)

    test_results = []
    for idx, (_, row) in enumerate(sample_rows.iterrows(), 1):
        img_p = Path(row["image_path"])
        gt_plate = row["ground_truth_plate"]
        folder = row["source_folder"]

        print(f"--- [Test {idx}/{len(sample_rows)}] Image: {img_p.name} (Category/State: {folder}) ---")
        print(f"Ground Truth Plate: {gt_plate}")

        try:
            detections = pipeline.process_image(img_p, camera_id=f"TEST_CAM_{idx}")
            if detections:
                det = detections[0]
                detected_plate = det["plateNumber"]
                conf = det["confidence"]
                v_type = det["vehicleType"]
                annotated_img = det.get("annotatedImage", "N/A")

                # Match evaluation
                is_match = (detected_plate == gt_plate) or (gt_plate in detected_plate) or (detected_plate in gt_plate)
                status = "EXACT / CLOSE MATCH" if is_match else "DETECTED"

                print(f"-> Detected Vehicle : {v_type}")
                print(f"-> Detected Plate   : {detected_plate} (Confidence: {int(conf * 100)}%)")
                print(f"-> Annotated Output : {annotated_img}")
                print(f"-> Result Status    : {status}\n")

                test_results.append({
                    "Image": img_p.name,
                    "Ground Truth": gt_plate,
                    "Detected Plate": detected_plate,
                    "Confidence": f"{int(conf * 100)}%",
                    "Vehicle Type": v_type,
                    "Status": status
                })
            else:
                print("-> No plate detected in this frame.\n")
                test_results.append({
                    "Image": img_p.name,
                    "Ground Truth": gt_plate,
                    "Detected Plate": "None",
                    "Confidence": "0%",
                    "Vehicle Type": "None",
                    "Status": "NOT DETECTED"
                })
        except Exception as err:
            print(f"-> Inference Error: {err}\n")

    results_df = pd.DataFrame(test_results)
    print("\n" + "="*80)
    print("LIVE TEST RESULTS SUMMARY TABLE:")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80 + "\n")

if __name__ == "__main__":
    df, base_path = load_kaggle_dataset_to_df()
    run_live_test(df, num_samples=5)
