import pyttsx3

def speak(text):
    engine = None
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except RuntimeError:
        # If the TTS engine is already running, stop it and restart
        if engine:
            engine.stop()
            engine.say(text)
            engine.runAndWait()
    finally:
        if engine:
            engine.stop()  # Ensure the engine is stopped after speaking