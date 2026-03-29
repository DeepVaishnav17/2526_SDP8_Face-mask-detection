"""
face_utils.py
─────────────
Uses DeepFace (Facenet512 model) for face recognition.
- No dlib, no CMake, no C++ compiler needed
- Installs with: pip install deepface tensorflow
- Very accurate, persists across restarts via JSON embeddings
"""

import cv2
import numpy as np
import json
import os
import traceback
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# DeepFace lazy-loads its models on first call — no global init needed
from deepface import DeepFace

MODEL_NAME   = "VGG-Face"     # VGG-Face downloads perfectly. Facenet URL is currently globally broken.
DETECTOR     = "opencv"       # fastest, most reliable detector
DISTANCE     = "cosine"
THRESHOLD    = 0.40           # cosine distance — default 0.40 for VGG-Face


def get_embedding(frame_bgr: np.ndarray) -> Optional[list]:
    """
    Extract 512-d face embedding from a BGR frame.
    Returns list of floats, or None if no face found.
    """
    try:
        result = DeepFace.represent(
            img_path      = frame_bgr,
            model_name    = MODEL_NAME,
            detector_backend = DETECTOR,
            enforce_detection = True,
            align         = True,
        )
        if not result:
            return None
        emb = result[0]["embedding"]          # list of floats — JSON-safe
        return emb
    except Exception:
        return None


def register_face(frame_bgr: np.ndarray, save_path: str) -> Tuple[bool, Optional[list]]:
    """
    Detect face in frame, save cropped face image, return embedding.
    Returns (True, embedding) on success, (False, None) on failure.
    """
    try:
        # Detect face bounding box first
        face_objs = DeepFace.extract_faces(
            img_path          = frame_bgr,
            detector_backend  = DETECTOR,
            enforce_detection = True,
            align             = True,
        )
        if not face_objs:
            return False, None

        # Crop and save the best face
        face_obj = face_objs[0]
        region   = face_obj["facial_area"]
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        pad = 20
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame_bgr.shape[1], x + w + pad)
        y2 = min(frame_bgr.shape[0], y + h + pad)
        face_crop = frame_bgr[y1:y2, x1:x2]
        cv2.imwrite(save_path, face_crop)

        # Get embedding from full frame (more context = better embedding)
        emb = get_embedding(frame_bgr)
        if emb is None:
            return False, None

        return True, emb

    except Exception as e:
        print(f"[register_face] Error: {e}")
        return False, None


def recognize_face(frame_bgr: np.ndarray, db: Dict[str, Any]) -> Optional[Dict]:
    """
    Compare face in frame against all registered embeddings.
    Returns best match dict or None.
    """
    # Build known embeddings list
    known = []
    for pid, info in db.items():
        enc = info.get("face_encoding")
        if enc and isinstance(enc, list) and len(enc) > 0:
            try:
                known.append((pid, np.array(enc, dtype=np.float64)))
            except Exception:
                continue

    if not known:
        return None

    # Get embedding for current frame
    emb = get_embedding(frame_bgr)
    if emb is None:
        return None

    query = np.array(emb, dtype=np.float64)
    query_norm = query / (np.linalg.norm(query) + 1e-10)

    best_pid  = None
    best_dist = float("inf")

    for pid, ref in known:
        ref_norm = ref / (np.linalg.norm(ref) + 1e-10)
        # Cosine distance = 1 - cosine_similarity
        dist = float(1.0 - np.dot(query_norm, ref_norm))
        if dist < best_dist:
            best_dist = dist
            best_pid  = pid

    if best_pid and best_dist <= THRESHOLD:
        confidence = round((1.0 - best_dist) * 100, 1)
        return {
            "person_id" : best_pid,
            "person"    : db[best_pid],
            "confidence": confidence,
            "distance"  : best_dist,
        }

    return None
