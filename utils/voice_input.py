import speech_recognition as sr

def get_voice_input():
    """
    Get voice input from microphone and convert to text.

    Returns:
        str: Recognized text if successful
        None: If recognition failed

    Raises:
        sr.UnknownValueError: Could not understand audio
        sr.RequestError: Could not request results from service
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        audio = recognizer.listen(source)
        user_input = recognizer.recognize_google(audio)
        return user_input