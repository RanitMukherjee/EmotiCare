import av
import threading
import numpy as np
import speech_recognition as sr
import streamlit as st
from streamlit_webrtc import VideoProcessorBase
from collections import deque
from typing import List
from emotion_detection import detect_emotion

# Create video processor class for emotion detection
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.current_emotion = "Neutral"
        self.lock = threading.Lock()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed_img, emotion_label = detect_emotion(img)

        # Update emotion with thread lock
        with self.lock:
            self.current_emotion = emotion_label

        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

    def get_emotion(self):
        with self.lock:
            return self.current_emotion

# Audio context to share state between the WebRTC thread and Streamlit
class AudioContext:
    def __init__(self):
        self.lock = threading.Lock()
        self.frames = deque()
        self.recording = False

@st.cache_resource
def get_audio_context():
    return AudioContext()

async def queued_audio_frames_callback(frames: List[av.AudioFrame]) -> av.AudioFrame:
    ctx = get_audio_context()
    with ctx.lock:
        if ctx.recording:
            ctx.frames.extend(frames)
    
    # Return empty frames to be silent (or pass through if needed, but we are just recording)
    new_frames = []
    for frame in frames:
        input_array = frame.to_ndarray()
        new_frame = av.AudioFrame.from_ndarray(
            np.zeros(input_array.shape, dtype=input_array.dtype),
            layout=frame.layout.name,
        )
        new_frame.sample_rate = frame.sample_rate
        new_frames.append(new_frame)
    return new_frames

def process_voice_from_webrtc():
    """Process recorded audio from WebRTC and return recognized text."""
    ctx = get_audio_context()
    with ctx.lock:
        audio_frames = list(ctx.frames)
        ctx.frames.clear()
        ctx.recording = False # Ensure recording stops

    if not audio_frames:
        st.warning("No audio data captured.")
        return None

    # Convert av.AudioFrame to numpy array
    audio_data = []
    for frame in audio_frames:
        audio_data.append(frame.to_ndarray())

    if not audio_data:
        st.warning("No audio data captured.")
        return None

    try:
        # Combine audio frames
        audio_combined = np.concatenate(audio_data, axis=0)

        # Handle stereo audio (convert to mono)
        if len(audio_combined.shape) > 1:
            audio_combined = np.mean(audio_combined, axis=1)

        # Check for silence to avoid division by zero
        max_val = np.max(np.abs(audio_combined))
        if max_val == 0:
            st.warning("Audio is silent.")
            return None

        # Normalize and convert to int16
        audio_normalized = np.int16(audio_combined / max_val * 32767)

        # Create AudioData for speech recognition
        recognizer = sr.Recognizer()
        audio_segment = sr.AudioData(
            audio_normalized.tobytes(),
            sample_rate=48000,  # WebRTC typically uses 48kHz
            sample_width=2
        )

        # Perform speech recognition
        text = recognizer.recognize_google(audio_segment)
        return text

    except sr.UnknownValueError:
        st.warning("Could not understand the audio. Please try again.")
        return None
    except sr.RequestError as e:
        st.error(f"Speech recognition error: {e}")
        return None
    except Exception as e:
        st.error(f"Audio processing error: {e}")
        return None
