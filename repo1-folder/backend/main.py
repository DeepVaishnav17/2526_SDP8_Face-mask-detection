# """
# main.py
# ───────
# FaceGuard FastAPI backend.

# Endpoints:
#   GET  /             → health check
#   GET  /persons      → list registered persons
#   POST /register     → register a new person (name + base64 image)
#   POST /detect       → detect mask + identify person (base64 image)
#   GET  /logs/entries → JSON list of all log entries
#   GET  /logs/download→ check if Word doc exists
#   DELETE /persons/{id} → remove a person
# """

# import base64
# import json
# import traceback
# from datetime import datetime
# from pathlib import Path

# import cv2
# import numpy as np
# from fastapi import FastAPI, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware

# from face_utils import register_face, recognize_face
# from mask_utils import detect_mask
# from log_utils  import log_entry_to_word

# # ── App setup ──────────────────────────────────────────────────────────────
# app = FastAPI(title="FaceGuard API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ── Paths ──────────────────────────────────────────────────────────────────
# BASE_DIR      = Path(__file__).parent
# FACES_DIR     = BASE_DIR / "registered_faces"
# DB_PATH       = BASE_DIR / "registered_db.json"
# LOGS_DIR      = BASE_DIR.parent / "logs"

# FACES_DIR.mkdir(exist_ok=True)
# LOGS_DIR.mkdir(exist_ok=True)

# # ── Cooldown: don't re-log same person within N seconds ───────────────────
# LOG_COOLDOWN  = 15          # seconds
# _last_logged  : dict = {}   # person_id → datetime


# # ── DB helpers ─────────────────────────────────────────────────────────────
# def load_db() -> dict:
#     if DB_PATH.exists():
#         try:
#             with open(DB_PATH) as f:
#                 return json.load(f)
#         except Exception:
#             return {}
#     return {}

# def save_db(db: dict):
#     with open(DB_PATH, "w") as f:
#         json.dump(db, f, indent=2)


# # ── Image decode helper ────────────────────────────────────────────────────
# def b64_to_frame(b64: str) -> np.ndarray:
#     raw    = base64.b64decode(b64.split(",")[-1])
#     arr    = np.frombuffer(raw, np.uint8)
#     frame  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
#     if frame is None:
#         raise ValueError("Cannot decode image")
#     return frame


# # ── Routes ─────────────────────────────────────────────────────────────────

# @app.get("/")
# def root():
#     return {"status": "ok", "service": "FaceGuard API"}


# @app.get("/persons")
# def get_persons():
#     db = load_db()
#     return {
#         "persons": [
#             {
#                 "id"           : pid,
#                 "name"         : info["name"],
#                 "department"   : info.get("department", ""),
#                 "registered_at": info.get("registered_at", ""),
#             }
#             for pid, info in db.items()
#         ]
#     }


# @app.post("/register")
# async def register(
#     name      : str = Form(...),
#     department: str = Form(""),
#     image     : str = Form(...),   # base64 JPEG
# ):
#     try:
#         frame = b64_to_frame(image)
#         db    = load_db()

#         # Duplicate check
#         for info in db.values():
#             if info["name"].strip().lower() == name.strip().lower():
#                 raise HTTPException(400, detail=f"'{name}' is already registered.")

#         pid       = f"p_{int(datetime.now().timestamp() * 1000)}"
#         face_path = FACES_DIR / f"{pid}.jpg"

#         ok, embedding = register_face(frame, str(face_path))
#         if not ok or embedding is None:
#             raise HTTPException(
#                 400,
#                 detail=(
#                     "No face detected in the photo.\n"
#                     "Tips: face the camera directly, good lighting, NO mask while registering."
#                 ),
#             )

#         db[pid] = {
#             "name"         : name.strip(),
#             "department"   : department.strip(),
#             "face_path"    : str(face_path),
#             "face_encoding": embedding,          # plain Python float list → JSON-safe
#             "registered_at": datetime.now().isoformat(),
#         }
#         save_db(db)
#         return {"success": True, "person_id": pid, "name": name}

#     except HTTPException:
#         raise
#     except Exception:
#         raise HTTPException(500, detail=traceback.format_exc())


# @app.post("/detect")
# async def detect(image: str = Form(...)):
#     try:
#         frame = b64_to_frame(image)

#         # ── 1. Mask detection (always first) ──────────────────────────────
#         mask = detect_mask(frame)

#         if not mask.get("face_detected", False):
#             return {
#                 "status" : "no_face",
#                 "label"  : "NO FACE DETECTED",
#                 "color"  : "#6B7280",
#                 "logged" : False,
#             }

#         if mask.get("wearing_mask", False):
#             return {
#                 "status"    : "masked",
#                 "label"     : "MASK ON 😷",
#                 "message"   : "Person wearing mask — identity protected",
#                 "color"     : "#F59E0B",
#                 "confidence": round(mask.get("confidence", 0) * 100, 1),
#                 "logged"    : False,   # ← never log masked persons
#             }

#         # ── 2. Face recognition ───────────────────────────────────────────
#         db = load_db()
#         if not db:
#             return {
#                 "status" : "unknown",
#                 "label"  : "UNKNOWN ❓",
#                 "message": "No persons registered yet",
#                 "color"  : "#EF4444",
#                 "logged" : False,
#             }

#         match = recognize_face(frame, db)

#         if not match:
#             return {
#                 "status" : "unknown",
#                 "label"  : "UNKNOWN ❓",
#                 "message": "Face not in database",
#                 "color"  : "#EF4444",
#                 "logged" : False,
#             }

#         # ── 3. Known person → log with cooldown ───────────────────────────
#         pid        = match["person_id"]
#         person     = match["person"]
#         confidence = match["confidence"]
#         now        = datetime.now()
#         did_log    = False

#         last = _last_logged.get(pid)
#         if last is None or (now - last).total_seconds() >= LOG_COOLDOWN:
#             log_path = LOGS_DIR / "entry_log.docx"
#             log_entry_to_word(
#                 doc_path      = str(log_path),
#                 name          = person["name"],
#                 department    = person.get("department", ""),
#                 timestamp     = now.strftime("%Y-%m-%d %H:%M:%S"),
#                 confidence    = confidence,
#                 screenshot_b64= image,
#             )
#             _last_logged[pid] = now
#             did_log = True

#         return {
#             "status"    : "identified",
#             "label"     : f"✅ {person['name']}",
#             "name"      : person["name"],
#             "department": person.get("department", ""),
#             "color"     : "#10B981",
#             "confidence": confidence,
#             "logged"    : did_log,
#         }

#     except Exception:
#         raise HTTPException(500, detail=traceback.format_exc())


# @app.get("/logs/entries")
# def log_entries():
#     p = LOGS_DIR / "entries.json"
#     if p.exists():
#         try:
#             with open(p) as f:
#                 return {"entries": json.load(f)}
#         except Exception:
#             pass
#     return {"entries": []}


# @app.get("/logs/download")
# def log_download():
#     p = LOGS_DIR / "entry_log.docx"
#     if not p.exists():
#         raise HTTPException(404, detail="No log file yet — run detection first.")
#     return {"exists": True}


# @app.delete("/persons/{person_id}")
# def delete_person(person_id: str):
#     db = load_db()
#     if person_id not in db:
#         raise HTTPException(404, detail="Person not found")
#     info = db.pop(person_id)
#     face_path = Path(info.get("face_path", ""))
#     if face_path.exists():
#         try:
#             face_path.unlink()
#         except Exception:
#             pass
#     save_db(db)
#     _last_logged.pop(person_id, None)
#     return {"success": True, "deleted": info["name"]}


"""
main.py
───────
FaceGuard FastAPI backend.

Endpoints:
  GET  /             → health check
  GET  /persons      → list registered persons
  POST /register     → register a new person (name + base64 image)
  POST /detect       → detect mask + identify person (base64 image)
  GET  /logs/entries → JSON list of all log entries
  GET  /logs/download→ check if Word doc exists
  DELETE /persons/{id} → remove a person
"""

import base64
import json
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from face_utils import register_face, recognize_face
from mask_utils import detect_mask
from log_utils  import log_entry_to_word

# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(title="FaceGuard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
FACES_DIR     = BASE_DIR / "registered_faces"
DB_PATH       = BASE_DIR / "registered_db.json"
LOGS_DIR      = BASE_DIR.parent / "logs"

FACES_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ── Cooldown: don't re-log same person within N seconds ───────────────────
LOG_COOLDOWN  = 15          # seconds
_last_logged  : dict = {}   # person_id → datetime


# ── DB helpers ─────────────────────────────────────────────────────────────
def load_db() -> dict:
    if DB_PATH.exists():
        try:
            with open(DB_PATH) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_db(db: dict):
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=2)


# ── Image decode helper ────────────────────────────────────────────────────
def b64_to_frame(b64: str) -> np.ndarray:
    raw    = base64.b64decode(b64.split(",")[-1])
    arr    = np.frombuffer(raw, np.uint8)
    frame  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Cannot decode image")
    return frame


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "FaceGuard API"}


@app.get("/persons")
def get_persons():
    db = load_db()
    return {
        "persons": [
            {
                "id"           : pid,
                "name"         : info["name"],
                "department"   : info.get("department", ""),
                "registered_at": info.get("registered_at", ""),
            }
            for pid, info in db.items()
        ]
    }


@app.post("/register")
async def register(
    name      : str = Form(...),
    department: str = Form(""),
    image     : str = Form(...),   # base64 JPEG
):
    try:
        frame = b64_to_frame(image)
        db    = load_db()

        # Duplicate check
        for info in db.values():
            if info["name"].strip().lower() == name.strip().lower():
                raise HTTPException(400, detail=f"'{name}' is already registered.")

        pid       = f"p_{int(datetime.now().timestamp() * 1000)}"
        face_path = FACES_DIR / f"{pid}.jpg"

        ok, embedding = register_face(frame, str(face_path))
        if not ok or embedding is None:
            raise HTTPException(
                400,
                detail=(
                    "No face detected in the photo.\n"
                    "Tips: face the camera directly, good lighting, NO mask while registering."
                ),
            )

        db[pid] = {
            "name"         : name.strip(),
            "department"   : department.strip(),
            "face_path"    : str(face_path),
            "face_encoding": embedding,          # plain Python float list → JSON-safe
            "registered_at": datetime.now().isoformat(),
        }
        save_db(db)
        return {"success": True, "person_id": pid, "name": name}

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(500, detail=traceback.format_exc())


@app.post("/detect")
async def detect(image: str = Form(...)):
    try:
        frame = b64_to_frame(image)

        # ── 1. Mask detection (always first) ──────────────────────────────
        mask = detect_mask(frame)

        if not mask.get("face_detected", False):
            return {
                "status" : "no_face",
                "label"  : "NO FACE DETECTED",
                "color"  : "#6B7280",
                "logged" : False,
            }

        if mask.get("wearing_mask", False):
            return {
                "status"    : "masked",
                "label"     : "MASK ON 😷",
                "message"   : "Person wearing mask — identity protected",
                "color"     : "#F59E0B",
                "confidence": round(mask.get("confidence", 0) * 100, 1),
                "logged"    : False,   # ← never log masked persons
            }

        # ── 2. Face recognition ───────────────────────────────────────────
        db = load_db()
        if not db:
            return {
                "status" : "unknown",
                "label"  : "UNKNOWN ❓",
                "message": "No persons registered yet",
                "color"  : "#EF4444",
                "logged" : False,
            }

        match = recognize_face(frame, db)

        if not match:
            return {
                "status" : "unknown",
                "label"  : "UNKNOWN ❓",
                "message": "Face not in database",
                "color"  : "#EF4444",
                "logged" : False,
            }

        # ── 3. Known person → log with cooldown ───────────────────────────
        pid        = match["person_id"]
        person     = match["person"]
        confidence = match["confidence"]
        now        = datetime.now()
        did_log    = False

        last = _last_logged.get(pid)
        if last is None or (now - last).total_seconds() >= LOG_COOLDOWN:
            log_path = LOGS_DIR / "entry_log.docx"
            log_entry_to_word(
                doc_path      = str(log_path),
                name          = person["name"],
                department    = person.get("department", ""),
                timestamp     = now.strftime("%Y-%m-%d %H:%M:%S"),
                confidence    = confidence,
                screenshot_b64= image,
            )
            _last_logged[pid] = now
            did_log = True

        return {
            "status"    : "identified",
            "label"     : f"✅ {person['name']}",
            "name"      : person["name"],
            "department": person.get("department", ""),
            "color"     : "#10B981",
            "confidence": confidence,
            "logged"    : did_log,
        }

    except Exception:
        raise HTTPException(500, detail=traceback.format_exc())


@app.get("/logs/entries")
def log_entries():
    p = LOGS_DIR / "entries.json"
    if p.exists():
        try:
            with open(p) as f:
                return {"entries": json.load(f)}
        except Exception:
            pass
    return {"entries": []}


@app.get("/logs/check")
def log_check():
    p = LOGS_DIR / "entry_log.docx"
    return {"exists": p.exists()}


@app.get("/logs/download")
def log_download():
    p = LOGS_DIR / "entry_log.docx"
    if not p.exists():
        raise HTTPException(404, detail="No log file yet — run detection first.")
    return FileResponse(
        path=p,
        filename="entry_log.docx",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )


@app.delete("/persons/{person_id}")
def delete_person(person_id: str):
    db = load_db()
    if person_id not in db:
        raise HTTPException(404, detail="Person not found")
    info = db.pop(person_id)
    face_path = Path(info.get("face_path", ""))
    if face_path.exists():
        try:
            face_path.unlink()
        except Exception:
            pass
    save_db(db)
    _last_logged.pop(person_id, None)
    return {"success": True, "deleted": info["name"]}
