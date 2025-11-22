from gtts import gTTS
import streamlit as st
import io
import base64

def text_to_speech(text):
    """
    Converts text to speech using gTTS and returns the audio file pointer.
    """
    try:
        tts = gTTS(text=text, lang='en')
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
        
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None

def autoplay_hidden_audio(audio_fp):
    """
    Plays the audio file pointer invisibly using HTML5 audio tag.
    """
    try:
        audio_bytes = audio_fp.read()
        b64 = base64.b64encode(audio_bytes).decode()
        md = f"""
            <audio autoplay style="display:none;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Audio Playback Error: {e}")