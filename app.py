import os
import streamlit as st
from dotenv import load_dotenv
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from chatbot import initialize_chatbot, generate_response, extract_query_from_response, search_youtube
from utils.text_to_speech import text_to_speech, autoplay_hidden_audio
from utils.webrtc_logic import (
    EmotionProcessor,
    get_audio_context,
    queued_audio_frames_callback,
    process_voice_from_webrtc
)
import webbrowser

# Load environment variables
load_dotenv()

# Initialize the chatbot (Global components)
groq_chat = initialize_chatbot()

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

# WebRTC configuration for cloud deployment
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

# Create WebRTC streamer with both video and audio
st.write("### Live Emotion Detection")

# Use factory pattern for processors
def video_processor_factory():
    return EmotionProcessor()

webrtc_ctx = webrtc_streamer(
    key="emotion-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=video_processor_factory,
    queued_audio_frames_callback=queued_audio_frames_callback,
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


# Show current emotion status
if webrtc_ctx.state.playing:
    placeholder = st.empty()
    with placeholder.container():
        current_emotion = get_current_emotion()
        st.info(f"Current detected emotion (updates on interaction): **{current_emotion}**")
        st.session_state.current_emotion = current_emotion

# Display previous messages
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            st.markdown(message["content"])
        
        # Add TTS button for assistant messages
        if message["role"] == "assistant":
            with col2:
                if st.button("🔊", key=f"tts_{i}", help="Read aloud"):
                    audio_fp = text_to_speech(message["content"])
                    if audio_fp:
                        autoplay_hidden_audio(audio_fp)

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
            user_input, emotion_label, groq_chat, st.session_state.messages
        )

        # Extract the chatbot's message (before the delimiter)
        if "|||" in full_response:
            chatbot_message = full_response.split("|||", 1)[0].strip()
        else:
            chatbot_message = full_response

        # Display the assistant's response (without the query)
        # Note: We don't need to display it here explicitly because the rerun will handle it in the loop above?
        # Actually, Streamlit execution flow: handle_input runs, updates state.
        # We usually want to show it immediately.
        # But since we moved the display logic to the top loop, we might need to rerun to show it?
        # Or we can just append to state and st.rerun().
        
        # Let's keep the immediate display for responsiveness, but without the button (complex to add dynamically).
        # Or better: Just append to state and rerun. This ensures consistent UI.
        
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

        # Force a rerun to update the chat history with the new message and button
        st.rerun()

# Button to find nearby psychiatrists
if st.button("Find Help"):
    # Open the Google Maps link in a new tab
    webbrowser.open_new_tab("https://www.google.com/maps/search/psychiatrists+near+me")

# Voice input section with WebRTC
st.write("### Voice Input")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🎤 Start Recording"):
        if webrtc_ctx.state.playing:
            ctx = get_audio_context()
            with ctx.lock:
                ctx.frames.clear()
                ctx.recording = True
            st.session_state.is_recording = True
            st.success("Recording started...")
        else:
            st.warning("Please allow microphone access and ensure the stream is running.")

with col2:
    if st.button("⏹️ Stop & Process"):
        if st.session_state.is_recording:
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