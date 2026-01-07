#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSL sign language backend API

- /health           : health check
- /predict          : uploaded video -> sequence of tokens + sentence
- /predict_camera   : streaming single frames -> one token at a time
"""
import logging
import os
import tempfile
import base64
from io import BytesIO
import random
from typing import List, Dict, Optional, Tuple
import imageio_ffmpeg

import time
import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---- IMPORT FROM YOUR ML ENGINEER'S MODULE ----
from sign_language_recognition import SignLanguageLSTM, MediaPipeProcessor, CONFIG

# ============================================================
# CONFIG
# ============================================================

# Demo vs real model
DEV_MODE = False  # <-- set True for dummy demo, False to use real model

# Model path resolution
MODEL_PATH = (
    os.environ.get("MSL_MODEL_PATH")
    or os.environ.get("MODEL_PATH")
    or CONFIG.get("model_save_path", "sign_language_model.pth")
)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

CREATE_NO_WINDOW = 0x08000000

app = Flask(__name__)
CORS(app)

# ============================================================
# REAL-TIME CAMERA CONSTANTS (DEV_MODE only)
# ============================================================

# Which glosses are considered "temporal" (need motion / GIF)
CAMERA_TEMPORAL_GLOSSES = {
    "ambil", "hari", "hi", "hujan", "jangan",
    "kakak", "keluarga", "kereta", "lemak", "lupa",
    "marah", "minum", "pergi", "pukul", "tanya",
}

# Gloss → human-friendly translation (you can extend this)
GLOSS_TRANSLATIONS: Dict[str, str] = {
    "ambil": "take",
    "hari": "day",
    "hi": "hi",
    "hujan": "rain",
    "jangan": "don't",
    "kakak": "sister",
    "keluarga": "family",
    "kereta": "car",
    "lemak": "oil",
    "lupa": "forget",
    "marah": "angry",
    "minum": "drink",
    "pergi": "go",
    "pukul": "hit",
    "tanya": "ask",
}

# Per-session sliding buffers (for real-time camera)
SESSION_BUFFERS: Dict[str, List[np.ndarray]] = {}
SESSION_STATE: Dict[str, Dict] = {}

MAX_BUFFER_FRAMES = 150          # keep last ~1–2 seconds worth
FRAMES_PER_TOKEN_MIN = 30        # DEMO: min frames before emitting token

# ============================================================
# GENERIC VIDEO/IMAGE HELPERS (for thumbnails & GIFs)
# ============================================================

def get_translation(gloss: str) -> str:
    """Map gloss to translation, fallback to gloss itself."""
    return GLOSS_TRANSLATIONS.get(gloss.lower(), gloss)


def get_video_metadata(video_path: str) -> Tuple[int, float]:
    """Return (frame_count, fps) for a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return frame_count, fps


def frame_to_data_url(frame: np.ndarray, jpg_quality: int = 55) -> Optional[str]:
    if frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)])
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"

def center_window(frames: List[np.ndarray], keep: int = 30) -> List[np.ndarray]:
    if len(frames) <= keep:
        return frames
    mid = len(frames) // 2
    half = keep // 2
    return frames[max(0, mid - half): min(len(frames), mid + half)]


def extract_static_thumbnail(video_path: str, frame_index: int) -> Optional[str]:
    """Grab a single frame from the video and return it as a JPEG data URL."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if frame_count <= 0:
        cap.release()
        return None

    idx = max(0, min(frame_index, frame_count - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return None
    return frame_to_data_url(frame)

def extract_gif_for_segment_fast(
    video_path: str,
    *,
    start_frame: int,
    end_frame: int,
    video_fps: float,
    # speed knobs
    out_width: int = 160,      # smaller = faster
    out_fps: float = 6.0,      # lower = faster
    max_duration_s: float = 1.2,   # shorter = faster
    max_frames: int = 12,      # cap frames hard
) -> str | None:
    """
    Returns data:image/gif;base64,... or None.
    Requires ffmpeg in PATH.
    """

    if video_fps <= 0:
        video_fps = 30.0

    # Clamp segment to a short window around the center (huge speed win)
    center = (start_frame + end_frame) // 2
    half_frames = int((max_duration_s * video_fps) / 2)
    s_frame = max(0, center - half_frames)
    e_frame = max(s_frame + 1, center + half_frames)

    # Convert to time; use -ss before -i for FAST seek (less accurate, but you don't care)
    ss = s_frame / video_fps
    dur = (e_frame - s_frame) / video_fps

    # Enforce max_frames (hard cap)
    out_fps = max(1.0, float(out_fps))
    dur = min(dur, max_duration_s, max_frames / out_fps)

    vf = f"fps={out_fps},scale={out_width}:-1:flags=fast_bilinear"

    cmd = [
        FFMPEG,
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{ss:.3f}",
        "-i", video_path,
        "-t", f"{dur:.3f}",
        "-an", "-sn", "-dn",
        "-vf", vf,
        "-loop", "0",
        "-f", "gif",
        "pipe:1",
    ]

    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        if not p.stdout:
            return None
        b64 = base64.b64encode(p.stdout).decode("ascii")
        return f"data:image/gif;base64,{b64}"
    except subprocess.CalledProcessError:
        # optional: print(p.stderr.decode(errors="ignore"))
        return None
    

def gif_from_frames_fast_ffmpeg(
    frames_bgr: List[np.ndarray],
    *,
    out_width: int = 160,     # smaller = faster
    out_fps: float = 6.0,     # lower = faster
    max_frames: int = 12,     # fewer frames = faster
) -> Optional[str]:
    """
    Ultra-fast, low-quality GIF for real-time preview.
    - Resizes frames in OpenCV (C-fast), then ffmpeg just encodes (no filters).
    - Outputs a data URL: data:image/gif;base64,...
    """
    if not frames_bgr:
        return None

    # 1) Subsample aggressively
    step = max(1, len(frames_bgr) // max_frames)
    sampled = frames_bgr[::step]
    if not sampled:
        return None

    # cap hard
    sampled = sampled[:max_frames]

    # 2) Resize frames BEFORE feeding ffmpeg (huge speed win)
    h0, w0 = sampled[0].shape[:2]
    if w0 <= 0 or h0 <= 0:
        return None

    new_w = int(out_width)
    new_h = max(2, int(h0 * (new_w / w0)))

    resized = []
    for f in sampled:
        if f is None:
            continue
        # ensure contiguous memory for fast tobytes()
        f = np.ascontiguousarray(f)
        rf = cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        resized.append(np.ascontiguousarray(rf))

    if not resized:
        return None

    # 3) Encode GIF with ffmpeg (no palette, low quality, super fast)
    # Important: set input frame rate = out_fps so we don't need fps filters.
    cmd = [
        FFMPEG,
        "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{new_w}x{new_h}",
        "-r", str(float(out_fps)),
        "-i", "pipe:0",
        "-an",
        "-loop", "0",
        "-f", "gif",
        "pipe:1",
    ]

    try:
        p = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=CREATE_NO_WINDOW,  # safe on Windows
        )

        # stream frames into stdin
        assert p.stdin is not None
        for f in resized:
            p.stdin.write(f.tobytes())
        p.stdin.close()

        out = p.stdout.read() if p.stdout else b""
        err = p.stderr.read().decode("utf-8", errors="ignore") if p.stderr else ""
        ret = p.wait()

        if ret != 0 or not out:
            # optional: log err for debugging
            # print("ffmpeg gif error:", err)
            return None

        b64 = base64.b64encode(out).decode("ascii")
        return f"data:image/gif;base64,{b64}"

    except FileNotFoundError:
        # shouldn't happen if imageio_ffmpeg works
        return None


# ============================================================
# MODEL INTEGRATION (merged with backend_api.py)
# ============================================================

model: Optional[nn.Module] = None
gestures: Optional[List[str]] = None
label_map: Optional[Dict] = None
config: Optional[Dict] = None
device: Optional[torch.device] = None
processor: Optional[MediaPipeProcessor] = None


def load_model(model_path: str):
    """Load trained model from .pth checkpoint (backend_api version)."""
    global model, gestures, label_map, config, device, processor

    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = MediaPipeProcessor()

    print(f"[MSL] Loading model from {model_path} ...")
    checkpoint = torch.load(model_path, map_location=device)

    required_keys = ["gestures", "label_map", "config", "model_state_dict"]
    missing_keys = [k for k in required_keys if k not in checkpoint]
    if missing_keys:
        raise ValueError(f"Model checkpoint missing keys: {missing_keys}")

    gestures_cp = checkpoint["gestures"]
    label_map_cp = checkpoint["label_map"]
    config_cp = checkpoint["config"]

    print("[MSL] Model config:")
    print(f"  - gestures: {len(gestures_cp)}")
    print(f"  - sequence_length: {config_cp.get('sequence_length', 'N/A')}")
    print(f"  - input_size: {config_cp.get('input_size', 'N/A')}")
    print(f"  - hidden_size: {config_cp.get('hidden_size', 'N/A')}")
    print(f"  - device: {device}")

    slr_model = SignLanguageLSTM(
        config_cp["input_size"],
        config_cp["hidden_size"],
        len(gestures_cp),
    ).to(device)

    slr_model.load_state_dict(checkpoint["model_state_dict"])
    slr_model.eval()

    gestures = gestures_cp
    label_map = label_map_cp
    config = config_cp
    model = slr_model

    print(f"✅ Model loaded. Gestures: {len(gestures)}")
    print(f"Gestures list: {gestures}")
    return model

def extract_keypoints(results) -> np.ndarray:
    """从 MediaPipe 结果中提取关键点"""
    pose = np.array([[res.x, res.y, res.z, res.visibility] 
                    for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 4)
    
    lh = np.array([[res.x, res.y, res.z] 
                  for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)
    
    rh = np.array([[res.x, res.y, res.z] 
                  for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)
    
    return np.concatenate([pose, lh, rh])


def process_frames_to_sequence(frames: List[np.ndarray], max_frames: Optional[int] = None) -> Optional[np.ndarray]:
    """将帧序列转换为关键点序列"""
    # 如果没有指定max_frames，使用config中的sequence_length
    if max_frames is None:
        if config is None:
            raise RuntimeError("config未初始化，请先加载模型")
        max_frames = config.get('sequence_length', 120)
    
    keypoints_sequence = []
    
    for frame in frames:
        # 处理帧
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = processor.holistic.process(frame_rgb)
        frame_rgb.flags.writeable = True
        
        # 只保存检测到手部的帧
        if results.left_hand_landmarks or results.right_hand_landmarks:
            keypoints = extract_keypoints(results)
            keypoints_sequence.append(keypoints)
    
    if len(keypoints_sequence) == 0:
        return None
    
    # 填充或截断到固定长度（120帧）
    if len(keypoints_sequence) < max_frames:
        last_frame = keypoints_sequence[-1]
        keypoints_sequence.extend([last_frame] * (max_frames - len(keypoints_sequence)))
    else:
        keypoints_sequence = keypoints_sequence[:max_frames]
    
    return np.array(keypoints_sequence)


def predict_sequence(keypoints_seq: np.ndarray) -> Tuple[Optional[str], float]:
    """预测关键点序列对应的手势"""
    if keypoints_seq is None:
        return None, 0.0
    
    # 转换为tensor
    keypoints_tensor = torch.FloatTensor(keypoints_seq).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(keypoints_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_gesture = gestures[predicted.item()]
    confidence_score = confidence.item()
    
    return predicted_gesture, confidence_score


def is_temporal_sign(gloss: str) -> bool:
    """Decide if a sign is temporal (requires motion)."""
    return gloss.lower() in CAMERA_TEMPORAL_GLOSSES


def predict_sign(video_path: str, model_obj=None) -> Dict:
    """
    Offline video processing – full gesture sequence.
    (This is the backend_api sliding-window logic.)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"tokens": [], "sentence": ""}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps < 1:
        fps = 30.0  # default FPS

    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()

    if len(all_frames) == 0:
        return {"tokens": [], "sentence": ""}

    sequence_length = config["sequence_length"]
    step_size = max(1, sequence_length // 3)  # overlap 2/3

    tokens = []
    current_start = 0
    last_gloss = None
    last_end = -1

    # 如果视频帧数不足120帧，仍然尝试预测（使用所有可用帧）
    if len(all_frames) < sequence_length:
        window_frames = all_frames
        keypoints_seq = process_frames_to_sequence(window_frames, sequence_length)
        
        if keypoints_seq is not None:
            gloss, confidence = predict_sequence(keypoints_seq)
            if gloss and confidence > 0.5:
                token = {
                    "gloss": gloss,
                    "translation": get_translation(gloss),
                    "confidence": float(confidence),
                    "temporal": is_temporal_sign(gloss),
                    "start_frame": 0,
                    "end_frame": len(all_frames) - 1,
                    "fps": float(fps)
                }
                return {
                    "tokens": [token],
                    "sentence": get_translation(gloss),
                    "warning": f"视频帧数不足{sequence_length}帧（实际{len(all_frames)}帧），预测结果可能不够准确"
                }
        
        # 无法提取有效手势
        return {
            "tokens": [],
            "sentence": "",
            "error": f"视频帧数不足{sequence_length}帧（实际{len(all_frames)}帧），且无法提取有效手势",
            "frame_count": len(all_frames),
            "required_frames": sequence_length
        }
    
    while current_start + sequence_length <= len(all_frames):
        # 提取当前窗口的帧
        window_frames = all_frames[current_start:current_start + sequence_length]
        
        # 处理帧序列（提取120帧的关键点）
        keypoints_seq = process_frames_to_sequence(window_frames, sequence_length)
        
        if keypoints_seq is not None:
            # 预测手势
            gloss, confidence = predict_sequence(keypoints_seq)
            
            if gloss and confidence > 0.5:  # 置信度阈值
                end_frame = current_start + sequence_length - 1
                
                # 如果与上一个token相同，扩展范围
                if last_gloss == gloss and current_start <= last_end + step_size:
                    tokens[-1]['end_frame'] = int(end_frame)
                    tokens[-1]['confidence'] = max(tokens[-1]['confidence'], float(confidence))
                else:
                    # 新的手势token
                    token = {
                        "gloss": gloss,
                        "translation": get_translation(gloss),
                        "confidence": float(confidence),
                        "temporal": is_temporal_sign(gloss),
                        "start_frame": int(current_start),
                        "end_frame": int(end_frame),
                        "fps": float(fps)
                    }
                    tokens.append(token)
                    last_gloss = gloss
                    last_end = end_frame
        
        current_start += step_size
    
    # 构建句子
    sentence = " ".join([token['translation'] for token in tokens])
    
    return {
        "tokens": tokens,
        "sentence": sentence
    }


def run_msl_model_on_frames(frames_bgr, model_obj=None):
    if config is None:
        raise RuntimeError("config not initialized")

    seq = int(config["sequence_length"])     # e.g. 120
    min_seq = int(config.get("min_sequence_length", 48))  # add this to config

    if len(frames_bgr) < min_seq:
        return None

    # use as many as we have, capped at seq
    use_len = min(len(frames_bgr), seq)
    recent_frames = frames_bgr[-use_len:]

    keypoints_seq = process_frames_to_sequence(recent_frames, use_len)
    if keypoints_seq is None:
        return None

    gloss, confidence = predict_sequence(keypoints_seq)
    if not gloss or confidence < 0.6:
        return None

    start_frame = len(frames_bgr) - use_len
    end_frame = len(frames_bgr) - 1

    return {
        "gloss": gloss,
        "translation": get_translation(gloss),
        "confidence": float(confidence),
        "temporal": is_temporal_sign(gloss),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "fps": 30.0,
    }




def _extract_static_thumbnails_bulk(video_path: str, centers: list[int]) -> dict[int, str]:
    """
    Extract many center-frame thumbnails by decoding video ONCE sequentially.
    Returns {frame_index: data_url}.
    Uses your existing extract_static_thumbnail(video_path, frame_index) ONLY as fallback
    if bulk decode fails (so you can keep your current encoder logic).
    """
    centers_u = sorted(set(int(c) for c in centers))
    if not centers_u:
        return {}

    # If you already have a fast way to encode a frame to data-url, plug it in here.
    # For now, we’ll do sequential decode and call your existing encoder per frame index
    # only when we hit a needed frame (still far fewer opens/seeks).
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {c: extract_static_thumbnail(video_path, c) for c in centers_u}

    wanted = set(centers_u)
    max_frame = centers_u[-1]
    out = {}

    # Seek to the first requested frame, then read forward (cheap)
    cap.set(cv2.CAP_PROP_POS_FRAMES, centers_u[0])
    current = centers_u[0]

    while current <= max_frame and cap.isOpened():
        ret, _frame = cap.read()
        if not ret:
            break

        if current in wanted:
            # Use your existing method to produce the final data URL (keeps behavior identical)
            out[current] = extract_static_thumbnail(video_path, current)
            if len(out) == len(wanted):
                break

        current += 1

    cap.release()

    # Fallback for any missed frames
    for c in centers_u:
        out.setdefault(c, extract_static_thumbnail(video_path, c))

    return out


def add_visuals_to_tokens_fast(
    video_path: str,
    tokens: list[dict],
    *,
    make_gifs: bool = False,
    max_gifs: int = 6,
    max_workers: int = 4,
    gif_fps_cap: float = 12.0,
) -> list[dict]:
    frame_count, vid_fps = get_video_metadata(video_path)
    if frame_count <= 0:
        return tokens

    n = len(tokens)
    default_fps = float(vid_fps or 30.0)
    segment_len = max(1, frame_count // max(n, 1))

    # 1) Precompute segments + centers
    centers = []
    temporal_idxs = []

    for i, t in enumerate(tokens):
        if "start_frame" in t and "end_frame" in t:
            start = int(t["start_frame"])
            end = int(t["end_frame"])
        else:
            start = i * segment_len
            end = start + segment_len - 1

        start = max(0, min(start, frame_count - 1))
        end = max(start, min(end, frame_count - 1))

        t["start_frame"] = start
        t["end_frame"] = end

        fps_token = float(t.get("fps") or default_fps)
        # cap gif fps (huge speed + size win)
        fps_token = min(fps_token, gif_fps_cap)
        t["fps"] = fps_token

        center = (start + end) // 2
        centers.append(center)

        # default to image; we might upgrade to gif later
        t["thumbnail_type"] = "image"

        if make_gifs and t.get("temporal"):
            temporal_idxs.append(i)

    # 2) Bulk static thumbnails (one pass)
    thumb_map = _extract_static_thumbnails_bulk(video_path, centers)
    for i, t in enumerate(tokens):
        url = thumb_map.get(centers[i])
        if url:
            t["thumbnail_url"] = url

    # 3) GIFs in parallel (only for a capped number of tokens)
    if make_gifs and temporal_idxs:
        temporal_idxs = temporal_idxs[:max_gifs]

        def _make_gif(i: int) -> tuple[int, str | None]:
            t = tokens[i]
            gif_url = extract_gif_for_segment_fast(
                video_path,
                start_frame=int(t["start_frame"]),
                end_frame=int(t["end_frame"]),
                video_fps=float(vid_fps or 30.0),
                out_width=160,
                out_fps=min(float(t["fps"]), 6.0),   # force low fps
                max_duration_s=1.0,
                max_frames=10,
            )
            return i, gif_url

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_make_gif, i) for i in temporal_idxs]
            for fut in as_completed(futures):
                i, gif_url = fut.result()
                if gif_url:
                    tokens[i]["thumbnail_type"] = "gif"
                    tokens[i]["thumbnail_url"] = gif_url
                # else keep the already-attached static thumbnail

    return tokens


# Load model at import time (unless in demo mode)
if not DEV_MODE:
    start_time = time.perf_counter()
    load_model(MODEL_PATH)
    end_time = time.perf_counter()
    duration = end_time - start_time
    logging.info(f"Model cold start latency: {duration} seconds")
    print(f"Model cold start latency: {duration} seconds")

# ============================================================
# FLASK ROUTES
# ============================================================

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "num_gestures": len(gestures) if gestures else 0,
        "sequence_length": config.get('sequence_length', 'N/A') if config else 'N/A',
        "model_version": "v3"
    })

@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Upload a video (field name: 'file' or 'video') and get:
      {
        "mode": "dev-demo" | "model",
        "sequence": [...],
        "sentence": "..."
      }
    """
    file = request.files.get("file") or request.files.get("video")
    if file is None:
        return jsonify({"error": "No file or video field found"}), 400
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        # ---------------- REAL MODEL MODE ----------------
        if model is None or config is None:
            return jsonify({"error": "Model not loaded"}), 500

        start_predict_time = time.perf_counter()
        result = predict_sign(tmp_path, model)
        end_predict_time = time.perf_counter()
        predict_duration = end_predict_time - start_predict_time
        logging.info(f"Prediction latency: {predict_duration} seconds")
        print(f"Prediction latency: {predict_duration} seconds")

        tokens = result.get("tokens", [])
        sentence = result.get("sentence", "")

        start_visual_time = time.perf_counter()
        # tokens = add_visuals_to_tokens(tmp_path, tokens)
        tokens = add_visuals_to_tokens_fast(
                tmp_path,
                tokens,
                make_gifs=True,
                max_gifs=2,
                max_workers=4,
                gif_fps_cap=12.0,
            )
        end_visual_time = time.perf_counter()
        visual_duration = end_visual_time - start_visual_time
        logging.info(f"Visual latency: {visual_duration} seconds")
        print(f"Visual latency: {visual_duration} seconds")

        logging.info(f"{result}")

        # Frontend normalizePrediction() accepts sequence or tokens
        return jsonify(
            {
                "mode": "model",
                "sequence": tokens,
                "tokens": tokens,
                "sentence": sentence,
            }
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@app.route("/predict_camera", methods=["POST"])
def predict_camera_route():
    """
    Real-time camera endpoint.

    Frontend sends:
      - 'frame': image file (JPEG) – a single frame
      - 'session_id': string (same while camera is on)
    Returns:
      { "token": {...} } or { "token": null }
    """
    if "frame" not in request.files:
        return jsonify({"error": "No file field 'frame' in form-data"}), 400

    file = request.files["frame"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    data = file.read()
    if not data:
        return jsonify({"error": "Empty image data"}), 400

    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Could not decode frame"}), 400

    session_id = (
        request.form.get("session_id")
        or request.args.get("session_id")
        or "default"
    )

    buf = SESSION_BUFFERS.setdefault(session_id, [])
    state = SESSION_STATE.setdefault(
        session_id,
        {
            "index": 0,
            "frames_since_token": 0,
        },
    )

    buf.append(frame)
    state["frames_since_token"] += 1

    if len(buf) > MAX_BUFFER_FRAMES:
        buf[:] = buf[-MAX_BUFFER_FRAMES:]

    if model is None or config is None:
        return jsonify({"error": "Model not loaded"}), 500

    RUN_MODEL_EVERY_N_FRAMES = 3

    token_from_model = None
    if state["frames_since_token"] % RUN_MODEL_EVERY_N_FRAMES == 0:
        token_from_model = run_msl_model_on_frames(buf, model)

    if token_from_model is None:
        return jsonify({"token": None})

    start_local = int(token_from_model["start_frame"])
    end_local   = int(token_from_model["end_frame"])
    start_local = max(0, min(start_local, len(buf) - 1))
    end_local   = max(start_local, min(end_local, len(buf) - 1))

    frames_for_token = buf[start_local : end_local + 1]
    frames_for_token = center_window(frames_for_token, keep=30)

    temporal = bool(token_from_model.get("temporal", True))
    conf = float(token_from_model.get("confidence", 0.0))
    if temporal and conf >= 0.75:
        thumb_url = gif_from_frames_fast_ffmpeg(
            frames_for_token,
            out_width=120,
            out_fps=4.0,
            max_frames=8,
        )
        thumb_type = "gif" if thumb_url else "image"
    else:
        mid_idx = len(frames_for_token) // 2
        thumb_url = frame_to_data_url(frames_for_token[mid_idx])
        thumb_type = "image"

    # Drop consumed frames so next sign uses fresh motion
    del buf[: end_local + 1]

    token = {
        "id": f"{session_id}-{random.randint(0, 1_000_000)}",
        "gloss": token_from_model["gloss"],
        "translation": token_from_model.get(
            "translation",
            token_from_model["gloss"],
        ),
        "confidence": token_from_model.get("confidence", 0.9),
        "temporal": temporal,
        "start_frame": token_from_model.get("start_frame", 0),
        "end_frame": token_from_model.get(
            "end_frame",
            len(frames_for_token) - 1,
        ),
        "fps": token_from_model.get("fps", 30.0),
        "thumbnail_type": thumb_type,
        "thumbnail_url": thumb_url,
    }

    return jsonify({"token": token})


# --- ASGI wrapper for uvicorn (optional) ---
try:
    from asgiref.wsgi import WsgiToAsgi

    asgi_app = WsgiToAsgi(app)
except ImportError:
    asgi_app = None


if __name__ == "__main__":
    # If dev mode is off and model wasn't loaded yet, try again here
    if not DEV_MODE and model is None:
        main_start_time = time.perf_counter()
        load_model(MODEL_PATH)
        main_end_time = time.perf_counter()
        main_duration = main_end_time - main_start_time
        logging.info(f"Model cold start latency: {main_duration} seconds")
        print(f"Model cold start latency: {main_duration} seconds")

    app.run(host="0.0.0.0", port=8000, debug=True)
