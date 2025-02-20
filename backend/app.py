from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import requests
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables
API_KEY = os.getenv("OPENWEATHER_API_KEY")


app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

API_KEY = "YOUR_OPENWEATHER_API_KEY"  # Secure it with env variables
dataset_path = "spotify_data.csv"

# Load dataset
df = pd.read_csv(dataset_path)
selected_features = ['track_name', 'artist_name', 'danceability', 'energy', 'loudness', 'tempo', 'valence', 'acousticness']
df = df[selected_features].dropna()
scaler = StandardScaler()
df[selected_features[2:]] = scaler.fit_transform(df[selected_features[2:]])

# Train GMM model
gmm = GaussianMixture(n_components=10, random_state=42)
df['cluster'] = gmm.fit_predict(df[selected_features[2:]])

def get_weather(city):
    """Fetch real-time weather data."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'rainfall': data.get('rain', {}).get('1h', 0)
        }
    return None

@app.route('/recommend', methods=['POST'])
def recommend():
    city = request.json.get('city')
    weather_data = get_weather(city)
    if not weather_data:
        return jsonify({"error": "Invalid city or API issue"}), 400

    # Convert weather to features
    weather_df = pd.DataFrame([weather_data])
    weather_df['danceability'] = np.clip((100 - weather_df['humidity']) / 100, 0, 1)
    weather_df['energy'] = np.clip(weather_df['temperature'] / 50, 0, 1)
    weather_df['loudness'] = -10 + (weather_df['rainfall'] * -5)
    weather_df['tempo'] = 100 + (weather_df['temperature'] * 2)
    weather_df['valence'] = np.clip(weather_df['temperature'] / 40, 0, 1)
    weather_df['acousticness'] = np.clip(weather_df['humidity'] / 100, 0, 1)

    weather_scaled = scaler.transform(weather_df[['danceability', 'energy', 'loudness', 'tempo', 'valence', 'acousticness']])
    cluster_label = gmm.predict(weather_scaled)[0]

    recommended_songs = df[df['cluster'] == cluster_label].sample(n=5, random_state=42)
    return jsonify(recommended_songs[['track_name', 'artist_name']].to_dict(orient="records"))

if __name__ == '__main__':
    app.run(debug=True)
