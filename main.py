import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
from ultralytics import YOLO

# Charger modèle
model = YOLO("yolov8n.pt")

st.title("🎥 Detection auto en temps réel")

class VideoProcessor:

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        # Détection
        results = model(img, conf=0.5)

        # Dessiner
        annotated_frame = results[0].plot()

        return av.VideoFrame.from_ndarray(
            annotated_frame,
            format="bgr24"
        )

webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)