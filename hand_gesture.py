# hand_draw_streamlit.py

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time
import math

# ===================== Streamlit UI =====================
st.set_page_config(page_title="✋ Hand Drawing", page_icon="🎨")
st.title("✋ Virtual Drawing with Hand Gestures")

# Sidebar controls
st.sidebar.header("Brush Settings")
color_choice = st.sidebar.radio("Color", ["White", "Red", "Green", "Blue"])
thickness = st.sidebar.slider("Brush Thickness", 1, 30, 8)

if color_choice == "White":
    color = (255, 255, 255)
elif color_choice == "Red":
    color = (0, 0, 255)
elif color_choice == "Green":
    color = (0, 255, 0)
else:
    color = (255, 0, 0)

# Clear and Undo buttons
clear_canvas = st.sidebar.button("Clear Canvas 🗑️")
save_canvas = st.sidebar.button("Save Drawing 💾")

# ===================== Mediapipe Setup =====================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# ===================== Canvas + Webcam =====================
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
h, w, _ = frame.shape
canvas = np.zeros((h, w, 3), dtype=np.uint8)

stframe = st.empty()  # placeholder for video

last_point = None

# ===================== Main Loop =====================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("❌ Cannot access webcam")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    draw_point = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Index finger tip
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)
            draw_point = (x, y)

    if clear_canvas:
        canvas[:] = 0

    if draw_point:
        if last_point:
            cv2.line(canvas, last_point, draw_point, color, thickness)
        last_point = draw_point
    else:
        last_point = None

    # Overlay canvas on webcam
    blended = cv2.addWeighted(frame, 0.7, canvas, 1, 0)

    stframe.image(blended, channels="BGR")

    if save_canvas:
        fname = f"drawing_{int(time.time())}.png"
        cv2.imwrite(fname, canvas)
        st.success(f"Saved drawing as {fname}")
        save_canvas = False
