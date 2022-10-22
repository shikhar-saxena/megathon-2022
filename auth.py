import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import os

def auth():
    load_dotenv()
    client_credentials_manager = SpotifyClientCredentials(
        client_id=os.environ["CLIENT_ID"],
        client_secret=os.environ["CLIENT_SECRET"]
    )
    return spotipy.Spotify(client_credentials_manager=client_credentials_manager)