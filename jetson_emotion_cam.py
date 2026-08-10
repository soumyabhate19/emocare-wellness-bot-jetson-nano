#!/usr/bin/env python3
import os
import time
import cv2
import numpy as np


# --- Paths (match your wellness.py) ---
YUNET_PATH = "models/face_detection_yunet_2023mar.onnx"
EMOTION_PATH = "models/emotion-ferplus-8.onnx"

# FER+ label order used in your wellness.py
EMOTION_LABELS = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
]


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def main():
    # --- Validate models exist ---
    if not os.path.exists(YUNET_PATH):
        raise FileNotFoundError(f"YuNet model not found: {YUNET_PATH}")
    if not os.path.exists(EMOTION_PATH):
        raise FileNotFoundError(f"Emotion model not found: {EMOTION_PATH}")

    # --- Camera open (Jetson-friendly) ---
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    # Reduce load
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Ask for MJPG if supported (often faster)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass

    # Reduce buffering lag (may not be supported everywhere)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check /dev/video0 permissions and camera connection.")

    # --- Create detectors (same as in wellness.py) ---
    detector = cv2.FaceDetectorYN.create(
        YUNET_PATH,
        "",
        (320, 320),
        score_threshold=0.7,
        nms_threshold=0.3,
        top_k=5000,
    )

    emotion_net = cv2.dnn.readNetFromONNX(EMOTION_PATH)

    frame_count = 0
    last_emotion = None
    last_conf = 0.0

    print("Starting camera. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Failed to read frame from webcam.")
            break

        frame_count += 1
        do_emotion = (frame_count % 5 == 0)  # run emotion every 5th frame (same throttle)

        h, w = frame.shape[:2]

        # --- Face detect (always) ---
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)

        best_face = None  # (x,y,bw,bh) largest
        if faces is not None and len(faces) > 0:
            boxes = [tuple(map(int, f[:4])) for f in faces]
            for (x, y, bw, bh) in boxes:
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

            best_face = max(boxes, key=lambda b: b[2] * b[3])

        # --- Emotion inference (throttled) ---
        if do_emotion and best_face is not None:
            x, y, bw, bh = best_face

            x0 = clamp(x, 0, w - 1)
            y0 = clamp(y, 0, h - 1)
            x1 = clamp(x + bw, 0, w)
            y1 = clamp(y + bh, 0, h)

            face = frame[y0:y1, x0:x1]
            if face.size > 0:
                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (64, 64))

                # Match your wellness.py input: (1,1,64,64) float32
                blob = gray.astype("float32").reshape(1, 1, 64, 64)

                emotion_net.setInput(blob)
                scores = emotion_net.forward().reshape(-1)
                probs = softmax(scores)

                idx = int(np.argmax(probs))
                last_emotion = EMOTION_LABELS[idx]
                last_conf = float(probs[idx])

        # --- Overlay last known emotion ---
        if last_emotion is not None:
            label_txt = f"{last_emotion} ({last_conf:.2f})"
            cv2.putText(
                frame,
                label_txt,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        cv2.imshow("Jetson Face + Emotion", frame)

        # Quit on q
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # small sleep to avoid maxing CPU (same idea as your code)
        time.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
