from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableSequence
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
import uuid
from tools.youtube_tool import search_youtube_videos
import os
import re

# System prompt for the chatbot
system_prompt = """You are a compassionate and empathetic AI assistant. The user is feeling '{emotion_label}'. Please respond in a way that is supportive, understanding, and validates their feelings. Use emotes to convey emotions. Offer helpful suggestions if appropriate, but prioritize being a good listener and showing genuine care. 😊

If the user mentions music, relaxation, or any topic that could benefit from a YouTube video, provide a helpful response first, and then include a relevant YouTube search query after the delimiter `|||`. For example:
- "I recommend trying some relaxing ASMR sounds. It’s great for relaxation! ||| relaxing ASMR sounds"
- "Calming piano music can be very soothing. Try searching for this on YouTube! ||| calming piano music"
"""

def initialize_chatbot():
    """
    Initializes the chatbot components (Groq client, memory, etc.).
    Returns:
        - groq_chat: Initialized Groq client.
        - app: Compiled StateGraph application.
        - config: Configuration for the conversation thread.
    """
    # Initialize Groq client
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_chat = ChatGroq(temperature=0.7, model_name="moonshotai/kimi-k2-instruct-0905", groq_api_key=groq_api_key)

    # Define a new graph
    workflow = StateGraph(state_schema=MessagesState)

    # Define the function that calls the model
    def call_model(state: MessagesState):
        selected_messages = state["messages"]
        response = groq_chat.invoke(selected_messages)
        return {"messages": [response]}

    # Define the two nodes we will cycle between
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")

    # Adding memory is straightforward in langgraph!
    memory = MemorySaver()

    app = workflow.compile(
        checkpointer=memory
    )

    # The thread id is a unique key that identifies this particular conversation.
    # We'll just generate a random uuid here.
    thread_id = uuid.uuid4()
    config = {"configurable": {"thread_id": thread_id}}

    return groq_chat, app, config

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

def generate_response(user_input, emotion_label, groq_chat, app, config, chat_history):
    """
    Generates a response using the chatbot.
    Returns:
        - full_response: The generated response.
    """
    # Construct the chat prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt.format(emotion_label=emotion_label)),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}"),
        ]
    )

    # Create a RunnableSequence (replaces LLMChain)
    chain: RunnableSequence = prompt | groq_chat

    # Generate a response using the chain
    response = chain.invoke({"human_input": user_input, "chat_history": chat_history})
    return response.content