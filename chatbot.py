from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from tools.youtube_tool import search_youtube_videos
import os
import streamlit as st

# System prompt for the chatbot
system_prompt = """You are a compassionate and empathetic AI assistant. The user is feeling '{emotion_label}'. Please respond in a way that is supportive, understanding, and validates their feelings. Use emotes to convey emotions. Offer helpful suggestions if appropriate, but prioritize being a good listener and showing genuine care. 😊

If the user mentions music, relaxation, or any topic that could benefit from a YouTube video, provide a helpful response first, and then include a relevant YouTube search query after the delimiter `|||`. For example:
- "I recommend trying some relaxing ASMR sounds. It’s great for relaxation! ||| relaxing ASMR sounds"
- "Calming piano music can be very soothing. Try searching for this on YouTube! ||| calming piano music"
"""



@st.cache_resource
def initialize_chatbot():
    """
    Initializes the Groq client.
    Returns:
        - groq_chat: Initialized Groq client.
    """
    # Initialize Groq client
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_chat = ChatGroq(temperature=0.7, model_name="moonshotai/kimi-k2-instruct-0905", groq_api_key=groq_api_key)

    return groq_chat

def extract_query_from_response(response: str) -> str:
    """
    Extracts the portion after the delimiter `|||` from the chatbot's response.
    If no delimiter is found, returns None.
    """
    print("Chatbot Response:", response)  # Debugging: Print the chatbot's response
    if "|||" in response:
        # Split the response into the chatbot's message and the query
        chatbot_message, query = response.split("|||", 1)
        query = query.strip()  # Remove any leading/trailing whitespace
        print("Extracted Query:", query)  # Debugging: Print the extracted query
        return query
    else:
        print("No delimiter found. Skipping YouTube search.")  # Debugging
        return None

def search_youtube(query: str) -> list:
    """
    Searches YouTube for videos based on the given query and returns a list of video dictionaries.
    """
    try:
        # Call the YouTube tool
        videos = search_youtube_videos(query)
        return videos
    except Exception as e:
        return [{"error": str(e)}]



def generate_response(user_input, emotion_label, groq_chat, chat_history):
    """
    Generates a response using the chatbot via direct Groq invocation.
    """
    # Convert chat_history (list of dicts) to LangChain message objects
    converted_messages = []
    
    # Add System Message with dynamic emotion
    converted_messages.append(SystemMessage(content=system_prompt.format(emotion_label=emotion_label)))
    
    for msg in chat_history:
        if msg["role"] == "user":
            converted_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            converted_messages.append(AIMessage(content=msg["content"]))
            
    # Invoke the Groq client directly with the full history
    response = groq_chat.invoke(converted_messages)
    
    return response.content