import spotipy.util as util
from dotenv import load_dotenv
import os

from moodtape_functions import authenticate_spotify, aggregate_top_artists, aggregate_top_tracks, select_tracks, create_playlist

load_dotenv()
client_id = os.environ["CLIENT_ID"]
client_secret = os.environ["CLIENT_SECRET"]
username = os.environ["USERNAME"]

scope = 'user-library-read user-top-read playlist-modify-public user-follow-read'

token = util.prompt_for_user_token(username, scope, client_id, client_secret, redirect_uri="http://localhost:8888/callback")

spotify_auth = authenticate_spotify(token)
top_artists = aggregate_top_artists(spotify_auth)
top_tracks = aggregate_top_tracks(spotify_auth, top_artists)

def recommender(mood):

    mood = float(mood)

    selected_tracks = select_tracks(spotify_auth, top_tracks, mood)
    create_playlist(spotify_auth, selected_tracks, mood)
    # spotify_auth.playlist_tracks(playlist_uri, limit=15)

recommender(0.7)