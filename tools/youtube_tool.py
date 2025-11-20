import os
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Access the YouTube API key
load_dotenv()
youtube_api_key = os.getenv("YOUTUBE_API_KEY")

# Initialize YouTube API client
youtube = build('youtube', 'v3', developerKey=youtube_api_key)

def search_youtube_videos(query: str, max_results: int = 5) -> list:
    """
    Searches YouTube for videos based on the given query and returns a list of video dictionaries.
    """
    try:
        print("Searching YouTube for:", query)  # Debugging: Print the search query

        # Call the YouTube API
        request = youtube.search().list(
            q=query,
            part="snippet",
            type="video",  # Search for videos
            maxResults=max_results
        )
        response = request.execute()

        # Debugging: Print the raw API response
        print("YouTube API Response:", response)

        # Format the results
        videos = []
        for item in response['items']:
            video_id = item['id']['videoId']
            title = item['snippet']['title']
            url = f"https://www.youtube.com/watch?v={video_id}"
            videos.append({"title": title, "url": url})

        return videos
    except Exception as e:
        print("YouTube API Error:", e)  # Debugging: Print any errors
        return [{"error": str(e)}]