"""JSON-backed temporary chat storage for the Streamlit prototype."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4


BASE_DIR = Path(__file__).resolve().parent
STORE_PATH = BASE_DIR / "data" / "chat_history" / "chats.json"


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def empty_store() -> dict:
    return {"version": "1.0", "updated_at": now_iso(), "chats": {}}


def load_store() -> dict:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not STORE_PATH.exists():
        save_store(empty_store())
    try:
        with STORE_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError:
        backup = STORE_PATH.with_name(f"chats_corrupted_{datetime.now():%Y%m%d_%H%M%S}.json")
        shutil.copy2(STORE_PATH, backup)
        data = empty_store()
        save_store(data)
    data.setdefault("version", "1.0")
    data.setdefault("updated_at", now_iso())
    data.setdefault("chats", {})
    return data


def save_store(store: dict) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    store["updated_at"] = now_iso()
    with STORE_PATH.open("w", encoding="utf-8") as file:
        json.dump(store, file, ensure_ascii=False, indent=2)


def load_chats() -> dict:
    return load_store()["chats"]


def save_chats(chats: dict) -> None:
    store = load_store()
    store["chats"] = chats
    save_store(store)


def create_new_chat() -> str:
    return f"chat_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:4]}"


def get_chat(chat_id: str) -> dict | None:
    return load_chats().get(chat_id)


def save_chat(chat_id: str, title: str, messages: list[dict]) -> None:
    if not messages:
        return
    store = load_store()
    chats = store["chats"]
    existing = chats.get(chat_id, {})
    created_at = existing.get("created_at", now_iso())
    chats[chat_id] = {
        "id": chat_id,
        "title": title or "Hoi thoai phap luat",
        "messages": messages,
        "created_at": created_at,
        "updated_at": now_iso(),
    }
    save_store(store)


def rename_chat(chat_id: str, new_title: str) -> None:
    store = load_store()
    chat = store["chats"].get(chat_id)
    if not chat:
        return
    chat["title"] = new_title[:60].strip()
    chat["updated_at"] = now_iso()
    save_store(store)


def delete_chat(chat_id: str) -> None:
    store = load_store()
    store["chats"].pop(chat_id, None)
    save_store(store)


def cleanup_empty_chats() -> None:
    store = load_store()
    store["chats"] = {
        chat_id: chat for chat_id, chat in store["chats"].items() if chat.get("messages")
    }
    save_store(store)
