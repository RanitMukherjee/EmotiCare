import os
import streamlit as st
from dotenv import load_dotenv
import av
import threading
import numpy as np
import speech_recognition as sr
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase, AudioProcessorBase
from emotion_detection import detect_emotion
from chatbot import initialize_chatbot, generate_response, extract_query_from_response, search_youtube
from utils.text_to_speech import speak
import webbrowser
import queue

# Load environment variables
load_dotenv()

# Initialize the chatbot
groq_chat, app, config = initialize_chatbot()

# Streamlit app
st.title("Mental Health Companion Chatbot")

# Sidebar for YouTube recommendations
st.sidebar.title("YouTube Recommendations")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    initial_message_content = "Hey there! How's your day been? 😊"
    initial_message = {"role": "assistant", "content": initial_message_content}
    st.session_state.messages.append(initial_message)

# Initialize session state for recording control
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False
if "current_emotion" not in st.session_state:
    st.session_state.current_emotion = "Neutral"

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

# Create audio processor class for voice recording
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_buffer = []
        self.is_recording = False
        self.lock = threading.Lock()

    def recv(self, frame):
        if self.is_recording:
            sound = frame.to_ndarray()
            with self.lock:
                self.audio_buffer.append(sound.copy())
        return frame

    def start_recording(self):
        with self.lock:
            self.is_recording = True
            self.audio_buffer = []

    def stop_recording(self):
        with self.lock:
            self.is_recording = False
            audio_data = self.audio_buffer.copy()
            self.audio_buffer = []
        return audio_data

# WebRTC configuration for cloud deployment
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

# Create WebRTC streamer with both video and audio
st.write("### Live Emotion Detection")

# Use factory pattern for processors
def video_processor_factory():
    return EmotionProcessor()

def audio_processor_factory():
    return AudioProcessor()

webrtc_ctx = webrtc_streamer(
    key="emotion-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=video_processor_factory,
    audio_processor_factory=audio_processor_factory,
    media_stream_constraints={
        "video": True,
        "audio": True,
    },
    async_processing=True,
)

# Get current emotion from processor
def get_current_emotion():
    """Get the current detected emotion from the video processor."""
    if webrtc_ctx.video_processor:
        return webrtc_ctx.video_processor.get_emotion()
    return "Neutral"

# Process voice input from audio processor
def process_voice_from_webrtc():
    """Process recorded audio from WebRTC and return recognized text."""
    if not webrtc_ctx.audio_processor:
        return None

    audio_data = webrtc_ctx.audio_processor.stop_recording()

    if not audio_data:
        return None

    try:
        # Combine audio frames
        audio_combined = np.concatenate(audio_data, axis=0)

        # Convert to proper format for speech recognition
        # Flatten if multi-channel
        if len(audio_combined.shape) > 1:
            audio_combined = audio_combined.flatten()

        # Normalize and convert to int16
        audio_normalized = np.int16(audio_combined / np.max(np.abs(audio_combined)) * 32767)

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

# Show current emotion status
if webrtc_ctx.state.playing:
    placeholder = st.empty()
    with placeholder.container():
        current_emotion = get_current_emotion()
        st.info(f"Current detected emotion: **{current_emotion}**")
        st.session_state.current_emotion = current_emotion

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Unified input handling
def handle_input(user_input):
    if user_input:
        # Get current emotion
        emotion_label = get_current_emotion()

        # Display the user's input in the chat interface
        with st.chat_message("user"):
            st.markdown(user_input)

        # Add the user's input to the session state
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate a response
        full_response = generate_response(
            user_input, emotion_label, groq_chat, app, config, st.session_state.messages
        )

        # Extract the chatbot's message (before the delimiter)
        if "|||" in full_response:
            chatbot_message = full_response.split("|||", 1)[0].strip()
        else:
            chatbot_message = full_response

        # Display the assistant's response (without the query)
        with st.chat_message("assistant"):
            st.markdown(chatbot_message)

        # Add the assistant's response to the session state
        st.session_state.messages.append({"role": "assistant", "content": chatbot_message})

        # Extract the YouTube query from the chatbot's response
        youtube_query = extract_query_from_response(full_response)

        # Fetch YouTube recommendations only if a valid query is found
        if youtube_query:
            youtube_results = search_youtube(youtube_query)

            # Display YouTube recommendations in the sidebar
            if youtube_results:
                st.sidebar.markdown("### Recommended Videos")
                for video in youtube_results:
                    st.sidebar.markdown(f"- [{video['title']}]({video['url']})")
            else:
                st.sidebar.markdown("No videos found for the given query.")
        else:
            st.sidebar.markdown("No YouTube query found in the chatbot's response.")

        # Convert the assistant's response to speech
        speak(chatbot_message)

# Button to find nearby psychiatrists
if st.button("Find Help"):
    # Open the Google Maps link in a new tab
    webbrowser.open_new_tab("https://www.google.com/maps/search/psychiatrists+near+me")

# Voice input section with WebRTC
st.write("### Voice Input")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🎤 Start Recording"):
        if webrtc_ctx.audio_processor:
            webrtc_ctx.audio_processor.start_recording()
            st.session_state.is_recording = True
            st.success("Recording started...")
        else:
            st.warning("Please allow microphone access and ensure the stream is running.")

with col2:
    if st.button("⏹️ Stop & Process"):
        if webrtc_ctx.audio_processor and st.session_state.is_recording:
            with st.spinner("Processing speech..."):
                text = process_voice_from_webrtc()
                if text:
                    st.success(f"Recognized: {text}")
                    handle_input(text)
                else:
                    st.warning("No speech detected.")
            st.session_state.is_recording = False
        else:
            st.warning("Not recording or stream not active.")

with col3:
    if st.session_state.is_recording:
        st.error("🔴 Recording...")
    else:
        st.info("⚫ Ready")

# Text input (existing chat input)
if prompt := st.chat_input("Talk to me..."):
    handle_input(prompt)