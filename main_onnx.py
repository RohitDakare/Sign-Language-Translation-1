# main_onnx.py
# Realtime ISL Translator using:
# - Mediapipe Hands (2 hands, 42*3 = 126 features)
# - GRU classifier exported to ONNX (model_gru_v1.onnx)
# - ONNX Runtime for inference (no TensorFlow / TFLite)
# - Same logic: sequence window, debouncing, sentence building, TTS

import sys
import json
import time
import threading
import queue
from pathlib import Path
from collections import deque

import numpy as np
import cv2
import mediapipe as mp
import onnxruntime as ort

# TTS optional
try:
    import pyttsx3
except Exception:
    pyttsx3 = None

from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QHBoxLayout
from PyQt6.QtCore import Qt

# --------------- CONFIG -----------------
PROJECT_ROOT = Path(__file__).parent
META_PATH = PROJECT_ROOT / "artifacts" / "metadata.json"
if not META_PATH.exists():
    META_PATH = PROJECT_ROOT / "metadata.json"

MODEL_ONNX_PATH = PROJECT_ROOT / "models" / "model_gru_v1.onnx"
if not MODEL_ONNX_PATH.exists():
    MODEL_ONNX_PATH = PROJECT_ROOT / "model_gru_v1.onnx"

NORMALIZE = False          # keep False unless you trained with normalization
THRESH = 0.80              # min probability to accept top class
DEBOUNCE_FRAMES = 4        # how many consistent frames before commit
NO_HANDS_CLEAR_FRAMES = 4  # how many frames with no hands before we consider pause
INACTIVITY_SECONDS_TO_FINALIZE = 1.0

SEQ_LEN_DEFAULT = 45
FEAT_DIM_DEFAULT = 126

WINDOW_NAME = "Realtime ISL Translator (ONNX) - ESC/Q to quit"
RECENT_COMMIT_MAX = 6
SHOW_FPS = False
STRETCH_FULLSCREEN = False  # True = fill screen (stretched), False = letterbox
# ---------------------------------------

# ---------- LOAD METADATA -------------
if not META_PATH.exists():
    print(f"[WARN] Missing metadata.json at {META_PATH}, using defaults.")
    # <-- FIXED: fallback includes the 6 labels your ONNX export used
    meta = {"labels": ['eat', 'fine', 'go', 'i', 'morning', 'you'],
            "seq_len": SEQ_LEN_DEFAULT,
            "feat_dim": FEAT_DIM_DEFAULT}
else:
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))

labels = list(meta.get("labels", []))
SEQ_LEN = int(meta.get("seq_len", SEQ_LEN_DEFAULT))
FEAT_DIM = int(meta.get("feat_dim", FEAT_DIM_DEFAULT))

print(f"[INFO] labels = {labels}")
print(f"[INFO] SEQ_LEN={SEQ_LEN} FEAT_DIM={FEAT_DIM}")
# ---------------------------------------

# ---------- LOAD ONNX MODEL -----------
if not MODEL_ONNX_PATH.exists():
    raise FileNotFoundError(f"ONNX model not found: {MODEL_ONNX_PATH}")

print("[INFO] Loading ONNX model:", MODEL_ONNX_PATH)
session = ort.InferenceSession(str(MODEL_ONNX_PATH), providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print("[INFO] ONNX session ready. Input:", input_name, "Output:", output_name)

# <-- FIXED: warn if model's class count and metadata labels length mismatch
out_shape = session.get_outputs()[0].shape  # might contain None
num_model_classes = None
if out_shape:
    num_model_classes = out_shape[-1] if out_shape[-1] is not None else None
if num_model_classes is not None and num_model_classes != len(labels):
    print(f"[WARN] Model outputs {num_model_classes} classes but metadata has {len(labels)} labels!")
# ---------------------------------------

def predict_seq(window_seq):
    """
    window_seq: deque/list length SEQ_LEN, each is (FEAT_DIM,) float
    returns: np.array of probs shape (n_classes,) or None
    """
    if len(window_seq) != SEQ_LEN:
        return None
    arr = np.array(window_seq, dtype=np.float32)
    if arr.shape != (SEQ_LEN, FEAT_DIM):
        # defensive reshape attempt
        try:
            arr = arr.reshape(SEQ_LEN, FEAT_DIM)
        except Exception:
            return None
    if NORMALIZE:
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)
    arr = arr[None, ...]  # (1,SEQ_LEN,FEAT_DIM)
    preds = session.run([output_name], {input_name: arr})[0]  # (1, n_classes)
    return preds[0]

# ---------- MEDIAPIPE HANDS ----------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

def two_hand_vector(results):
    """
    Returns flattened vector [right(21*3), left(21*3)] = (126,)
    Same structure as training.
    """
    right = np.zeros((21, 3), dtype=np.float32)
    left = np.zeros((21, 3), dtype=np.float32)
    if getattr(results, "multi_handedness", None) and getattr(results, "multi_hand_landmarks", None):
        for lm_list, handness in zip(results.multi_hand_landmarks, results.multi_handedness):
            coords = np.array([[lm.x, lm.y, lm.z] for lm in lm_list.landmark], dtype=np.float32)
            label = handness.classification[0].label.lower()
            if label.startswith("r"):
                right = coords
            else:
                left = coords
    return np.concatenate([right.flatten(), left.flatten()]).astype(np.float32)
# -------------------------------------

# ---------- SENTENCE ASSEMBLER -------
WORD_LEXICON = {
    "i": ("I", "pronoun"),
    "you": ("you", "pronoun"),
    "good": ("good", "adj"),
    "morning": ("morning", "noun_time"),
    "afternoon": ("afternoon", "noun_time"),
    "evening": ("evening", "noun_time"),
    "night": ("night", "noun_time"),
    "fine": ("fine", "adj"),
    "hungry": ("hungry", "adj"),
    "eat": ("eat", "verb"),
    "go": ("go", "verb"),
    "sorry": ("sorry", "adj"),
    "how": ("how", "qword"),
}

def word_info(label):
    lab = label.lower()
    if lab in WORD_LEXICON:
        return WORD_LEXICON[lab]
    return (label, "word")

def assemble_sentence_auto(labels_list):
    """
    Basic version: use your older rules.
    We want:
      i + fine -> "I'm fine."
      good + morning -> "Good morning."
      go + eat -> "Let's go eat."
    and similar.
    """
    if not labels_list:
        return ""
    tokens = []
    classes = []
    for L in labels_list:
        tok, cls = word_info(L)
        tokens.append(tok)
        classes.append(cls)

    # common patterns
    if len(tokens) == 2 and classes == ["adj", "noun_time"]:
        return f"{tokens[0].capitalize()} {tokens[1]}."
    if len(tokens) == 2 and classes[0] == "pronoun" and classes[1] == "adj":
        if tokens[0].lower() == "i":
            return f"I'm {tokens[1]}."
        return f"Are {tokens[0]} {tokens[1]}?"
    if len(tokens) == 2 and classes[0] == "qword" and classes[1] == "pronoun":
        return f"{tokens[0].capitalize()} are {tokens[1]}?"
    if len(tokens) == 2 and classes[0] == "pronoun" and classes[1] == "verb":
        if tokens[1].lower() == "eat":
            return "Did I eat?" if tokens[0].lower() == "i" else f"{tokens[0].capitalize()} {tokens[1]}."
        return f"{tokens[0].capitalize()} {tokens[1]}."
    if len(tokens) == 2 and classes[0] == "verb" and classes[1] in ("verb", "noun", "adj", "word"):
        return f"Let's {tokens[0]} {tokens[1]}."
    if len(tokens) == 1:
        t = tokens[0].capitalize()
        if classes[0] == "adj" and t.lower() == "sorry":
            return "I'm sorry."
        if classes[0] == "noun_time":
            return f"Good {tokens[0]}."
        return t + "."

    # fallback join
    s = " ".join(tokens)
    s = s.capitalize() + "."
    s = s.replace("I am ", "I'm ")
    return s
# -------------------------------------

# ---------- TTS WORKER ---------------
tts_queue = queue.Queue()
TTS_ENABLED = False
tts_thread = None

if pyttsx3 is not None:
    try:
        def tts_worker(q):
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            while True:
                text = q.get()
                if text is None:
                    break
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception as e:
                    print("[TTS ERROR]", e)
                q.task_done()

        tts_thread = threading.Thread(target=tts_worker, args=(tts_queue,), daemon=True)
        tts_thread.start()
        TTS_ENABLED = True
        print("[INFO] TTS enabled.")
    except Exception as e:
        print("[WARN] Failed to init pyttsx3:", e)
else:
    print("[WARN] pyttsx3 not installed, TTS disabled.")
# -------------------------------------

# ---------- DISPLAY HELPERS ----------
import ctypes
from ctypes import wintypes
user32 = ctypes.windll.user32

class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]

SPI_GETWORKAREA = 0x0030
work_area = RECT()
ctypes.windll.user32.SystemParametersInfoW(
    SPI_GETWORKAREA, 0, ctypes.byref(work_area), 0
)

SCREEN_W = work_area.right - work_area.left
SCREEN_H = work_area.bottom - work_area.top

GWL_STYLE = -16
WS_CAPTION = 0x00C00000
WS_THICKFRAME = 0x00040000
WS_MINIMIZE = 0x20000000
WS_MAXIMIZEBOX = 0x00010000
WS_SYSMENU = 0x00080000

# ---------- WINDOWS TRUE FULLSCREEN (FORCED) ----------
def force_true_fullscreen(window_name):
    time.sleep(0.2)  # allow window to exist

    hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
    if hwnd == 0:
        print("[WARN] Window handle not found")
        return

    GWL_STYLE = -16
    WS_OVERLAPPEDWINDOW = 0x00CF0000
    WS_POPUP = 0x80000000

    # Remove window borders
    style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_STYLE)
    style &= ~WS_OVERLAPPEDWINDOW
    style |= WS_POPUP
    ctypes.windll.user32.SetWindowLongW(hwnd, GWL_STYLE, style)

    # Force window to cover screen
    ctypes.windll.user32.SetWindowPos(
        hwnd,
        None,
        0, 0,
        SCREEN_W, SCREEN_H,
        0x0040  # SWP_FRAMECHANGED
    )

def force_borderless_fullscreen(win_name):
    time.sleep(0.15)
    hwnd = user32.FindWindowW(None, win_name)
    if hwnd == 0:
        return
    style = user32.GetWindowLongW(hwnd, GWL_STYLE)
    style &= ~(WS_CAPTION | WS_THICKFRAME | WS_MINIMIZE |
               WS_MAXIMIZEBOX | WS_SYSMENU)
    user32.SetWindowLongW(hwnd, GWL_STYLE, style)
    user32.SetWindowPos(hwnd, None, 0, 0, SCREEN_W, SCREEN_H, 0x0040)

def show_frame_preserve_aspect(window_name, frame, screen_w=SCREEN_W, screen_h=SCREEN_H):
    h, w = frame.shape[:2]
    if STRETCH_FULLSCREEN:
        resized = cv2.resize(frame, (screen_w, screen_h))
        cv2.imshow(window_name, resized)
        return 0, 0, screen_w, screen_h
    scale = min(screen_w / w, screen_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    x_off = (screen_w - new_w) // 2
    y_off = (screen_h - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    cv2.imshow(window_name, canvas)
    return x_off, y_off, new_w, new_h
# -------------------------------------

# ---------- RUNTIME STATE ------------
window = deque(maxlen=SEQ_LEN)
display_label = ""
last_top = None
top_count = 0
no_hands_count = 0

committed_labels = []
last_committed_time = 0.0
last_spoken_sentence = ""
last_displayed_sentence = ""
sentence_lock = threading.Lock()

conf_last = 0.0
show_confidence_bar = True
show_settings_hud = True
recent_commits = deque(maxlen=RECENT_COMMIT_MAX)
runtime_thresh = THRESH
runtime_debounce = DEBOUNCE_FRAMES

fullscreen_applied = False

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera (index 0)")

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL
)

def attempt_fullscreen_methods():
    try:
        if STRETCH_FULLSCREEN:
            cv2.resizeWindow(WINDOW_NAME, SCREEN_W, SCREEN_H)
            cv2.moveWindow(WINDOW_NAME, 0, 0)
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except Exception:
        pass
    time.sleep(0.05)

#attempt_fullscreen_methods()
# -------------------------------------

def finalize_sentence_and_speak():
    global committed_labels, last_spoken_sentence, last_displayed_sentence
    with sentence_lock:
        if not committed_labels:
            return
        sent = assemble_sentence_auto(committed_labels)
        committed_labels = []
    if sent and sent != last_spoken_sentence:
        last_spoken_sentence = sent
        last_displayed_sentence = sent
        print("[SENTENCE]", sent)
        if TTS_ENABLED:
            tts_queue.put(sent)
        def _clear():
            time.sleep(1.2)
            with sentence_lock:
                global last_displayed_sentence
                last_displayed_sentence = ""
        threading.Thread(target=_clear, daemon=True).start()

# ---------- START WINDOW -------------
class StartWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ISL Realtime Translator (ONNX)")
        self.setFixedSize(520, 260)
        screen = QApplication.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            x = geo.x() + (geo.width() - self.width()) // 2
            y = geo.y() + (geo.height() - self.height()) // 2
            self.move(x, y)

        layout = QVBoxLayout()
        title = QLabel("<h2>Realtime ISL Translator</h2>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        desc = QLabel("GRU + ONNX Runtime, realtime ISL to speech.\nClick Start to open camera.")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc)

        btn_row = QHBoxLayout()
        start_btn = QPushButton("Start Realtime Translation")
        start_btn.clicked.connect(self.start_main_loop)
        btn_row.addWidget(start_btn)
        demo_btn = QPushButton("Close")
        demo_btn.clicked.connect(self.close)
        btn_row.addWidget(demo_btn)
        layout.addLayout(btn_row)

        info = QLabel("Controls: +/- threshold, [/] debounce, s toggle HUD")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info)

        self.setLayout(layout)

    def start_main_loop(self):
        self.close()

# ---------- OVERLAYS ------------------
def draw_overlays(frame, prob, top_label):
    global recent_commits, runtime_thresh, runtime_debounce, show_confidence_bar, show_settings_hud
    h, w = frame.shape[:2]

    # recent commits panel
    panel_h = 26 + (len(recent_commits) * 20)
    panel = frame.copy()
    cv2.rectangle(panel, (10, h - panel_h - 10), (260, h - 10), (0, 0, 0), -1)
    frame = cv2.addWeighted(panel, 0.5, frame, 0.5, 0)

    tx = 16
    ty = h - panel_h
    cv2.putText(frame, "Recent:", (tx, ty + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    i = 0
    for lbl in reversed(recent_commits):
        cv2.putText(frame, f"- {lbl}", (tx, ty + 18 + (i + 1) * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 240, 240), 1, cv2.LINE_AA)
        i += 1

    if show_confidence_bar:
        bar_w = 260
        bar_h = 18
        bx = w - bar_w - 20
        by = 20
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (30, 30, 30), -1)
        p = float(max(0.0, min(1.0, prob)))
        fill = int(bar_w * p)
        color = (50, 180, 50) if prob >= runtime_thresh else (50, 50, 180)
        cv2.rectangle(frame, (bx, by), (bx + fill, by + bar_h), color, -1)
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (200, 200, 200), 1)
        txt = f"Confidence: {prob*100:4.1f}%"
        cv2.putText(frame, txt, (bx + 6, by + bar_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
        if top_label:
            cv2.putText(frame, f"Top: {top_label}", (bx, by + bar_h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    if show_settings_hud:
        hud_x = 16
        hud_y = 16
        info_lines = [
            f"THRESH={runtime_thresh:.2f}",
            f"DEBOUNCE={runtime_debounce}",
            "Model=GRU+ONNX"
        ]
        for idx, ln in enumerate(info_lines):
            cv2.putText(frame, ln, (hud_x, hud_y + idx * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
    return frame
# -------------------------------------
def maximize_opencv_window(window_name):
    hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
    if hwnd:
        ctypes.windll.user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE

def main_loop():
    global window, display_label, last_top, top_count, no_hands_count, last_committed_time
    global conf_last, recent_commits, runtime_thresh, runtime_debounce, show_confidence_bar, show_settings_hud
    global last_displayed_sentence

    #attempt_fullscreen_methods()
    print("Press ESC or Q to quit")
    # ---- MAXIMIZE WINDOW (WINDOWED, NOT FULLSCREEN) ----
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.resizeWindow(WINDOW_NAME, SCREEN_W, SCREEN_H)
    prev_time_local = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        hand_present = bool(getattr(res, "multi_hand_landmarks", None))
        now = time.time()

        if hand_present:
            no_hands_count = 0
            for hnd in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hnd, mp_hands.HAND_CONNECTIONS)

            feat = two_hand_vector(res)
            # <-- FIXED: defensive check to ensure feature vector length matches FEAT_DIM
            if feat is None or feat.size != FEAT_DIM:
                # skip this frame if feature length isn't correct
                continue
            window.append(feat)

            if len(window) == SEQ_LEN:
                preds = predict_seq(window)
                if preds is not None:
                    idx = int(np.argmax(preds))
                    prob = float(preds[idx])
                    conf_last = prob
                    top = labels[idx] if (idx < len(labels)) else None
                else:
                    idx = None
                    prob = 0.0
                    conf_last = 0.0
                    top = None

                top_ok = (top is not None and prob >= runtime_thresh)

                if top_ok:
                    if top == last_top:
                        top_count += 1
                    else:
                        top_count = 1
                    last_top = top
                else:
                    last_top = None
                    top_count = 0

                if top_ok and top_count >= runtime_debounce and top is not None:
                    with sentence_lock:
                        if not committed_labels or committed_labels[-1] != top:
                            committed_labels.append(top)
                            recent_commits.append(top)
                            last_committed_time = now
                            display_label = top
                            print("[COMMIT]", committed_labels)

        else:
            no_hands_count += 1
            window.clear()
            last_top = None
            top_count = 0
            display_label = ""

        # finalize after pause
        with sentence_lock:
            idle = now - last_committed_time
            has_committed = len(committed_labels) > 0
        if has_committed and (no_hands_count >= NO_HANDS_CLEAR_FRAMES
                              or idle >= INACTIVITY_SECONDS_TO_FINALIZE):
            finalize_sentence_and_speak()

        # overlays
        overlay_frame = frame.copy()
        if last_displayed_sentence:
            text = str(last_displayed_sentence)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1.2
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
            x, y = 20, 50
            box_tl = (x - 8, y - th - 8)
            box_br = (x + tw + 8, y + baseline + 8)
            cv2.rectangle(overlay_frame, box_tl, box_br, (0, 0, 0), -1)
            alpha = 0.50
            frame = cv2.addWeighted(overlay_frame, alpha, frame, 1 - alpha, 0)
            cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255),
                        thickness, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Perform signs... (ESC/Q to quit)", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200),
                        1, cv2.LINE_AA)

        if SHOW_FPS:
            now2 = time.time()
            fps = 1.0 / (now2 - prev_time_local) if now2 != prev_time_local else 0.0
            prev_time_local = now2
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200),
                        1, cv2.LINE_AA)

        frame = draw_overlays(frame, conf_last, display_label)

        try:
            show_frame_preserve_aspect(WINDOW_NAME, frame)
            if not hasattr(main_loop, "_max_done"):
                maximize_opencv_window(WINDOW_NAME)
                main_loop._max_done = True

        except Exception:
            cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break
        if key in (ord("+"), ord("=")):
            runtime_thresh = min(0.99, runtime_thresh + 0.01)
        if key in (ord("-"), ord("_")):
            runtime_thresh = max(0.01, runtime_thresh - 0.01)
        if key == ord("]"):
            runtime_debounce = min(12, runtime_debounce + 1)
        if key == ord("["):
            runtime_debounce = max(1, runtime_debounce - 1)
        if key == ord("p"):
            show_confidence_bar = not show_confidence_bar
        if key == ord("s"):
            show_settings_hud = not show_settings_hud

    cap.release()
    cv2.destroyAllWindows()
    if TTS_ENABLED:
        tts_queue.put(None)
        if tts_thread is not None:
            tts_thread.join(timeout=2)
    hands.close()
    print("Exited realtime translator (ONNX).")

def run_app():
    app = QApplication(sys.argv)
    w = StartWindow()
    w.show()
    app.exec()
    attempt_fullscreen_methods()
    main_loop()

if __name__ == "__main__":
    run_app()