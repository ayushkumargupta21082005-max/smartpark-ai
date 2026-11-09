"""A thin layer that lets routes use MongoDB or the local development store."""
from copy import deepcopy
from typing import Any
from bson import ObjectId
from .database import database


def serialise(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None
    result = deepcopy(document)
    if "_id" in result:
        result["id"] = str(result.pop("_id"))
    return result


def all_docs(name: str) -> list[dict[str, Any]]:
    collection = database.collection(name)
    if database.mode == "mongodb":
        return [serialise(item) for item in collection.find()]
    return [serialise(item) for item in collection]


def _normalize_query(query: dict[str, Any]) -> dict[str, Any]:
    q = dict(query)
    if "id" in q:
        val = q.pop("id")
        try:
            q["_id"] = ObjectId(val)
        except Exception:
            q["_id"] = val
    return q


def find_one(name: str, query: dict[str, Any]) -> dict[str, Any] | None:
    collection = database.collection(name)
    if database.mode == "mongodb":
        return serialise(collection.find_one(_normalize_query(query)))
    return next((serialise(item) for item in collection if all(item.get(key) == value for key, value in query.items())), None)


def insert_one(name: str, document: dict[str, Any]) -> dict[str, Any]:
    collection = database.collection(name)
    if database.mode == "mongodb":
        inserted = collection.insert_one(document)
        return serialise(collection.find_one({"_id": inserted.inserted_id}))
    item = deepcopy(document)
    item["_id"] = str(ObjectId())
    collection.append(item)
    return serialise(item)


def update_one(name: str, query: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any] | None:
    collection = database.collection(name)
    if database.mode == "mongodb":
        mquery = _normalize_query(query)
        doc = collection.find_one(mquery)
        if not doc:
            return None
        collection.update_one({"_id": doc["_id"]}, {"$set": changes})
        return serialise(collection.find_one({"_id": doc["_id"]}))
    for item in collection:
        if all(item.get(key) == value for key, value in query.items()):
            item.update(changes)
            return serialise(item)
    return None


def delete_one(name: str, query: dict[str, Any]) -> bool:
    collection = database.collection(name)
    if database.mode == "mongodb":
        return bool(collection.delete_one(_normalize_query(query)).deleted_count)
    for index, item in enumerate(collection):
        if all(item.get(key) == value for key, value in query.items()):
            collection.pop(index)
            return True
    return False

