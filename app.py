from flask import Flask, render_template, jsonify, request
import requests
import json
from datetime import datetime, timedelta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

app = Flask(__name__)

# APIs
ISS_LOCATION_API = "http://api.open-notify.org/iss-now.json"
ISS_PASS_API = "http://api.open-notify.org/iss-pass.json"
WEATHER_API_KEY = "7089d31fc236b0e8fcc61da88bd76000"
WEATHER_API = "http://api.openweathermap.org/data/2.5/weather"

class ISSVisibilityPredictor:
    def __init__(self):
        self.model = None
        self.load_or_train_model()
    
    def generate_training_data(self):
        """Generate synthetic training data for ISS visibility prediction"""
        np.random.seed(42)
        n_samples = 1000
        
        # Features: cloud_cover(%), humidity(%), wind_speed(km/h), light_pollution(1-10), elevation_angle(degrees)
        cloud_cover = np.random.randint(0, 101, n_samples)
        humidity = np.random.randint(30, 100, n_samples)
        wind_speed = np.random.randint(0, 50, n_samples)
        light_pollution = np.random.randint(1, 11, n_samples)
        elevation_angle = np.random.randint(10, 90, n_samples)
        
        X = np.column_stack([cloud_cover, humidity, wind_speed, light_pollution, elevation_angle])
        
        # Target: visibility (1 = visible, 0 = not visible)
        # Logic: Clear skies, low humidity, low light pollution = better visibility
        visibility_prob = (
            (100 - cloud_cover) * 0.4 +  # Clear skies are crucial
            (100 - humidity) * 0.2 +     # Low humidity helps
            (11 - light_pollution) * 10 * 0.3 +  # Dark skies important
            elevation_angle * 0.1        # Higher angle = better
        ) / 100
        
        # Add some noise and threshold
        y = (visibility_prob + np.random.normal(0, 0.1, n_samples)) > 0.6
        
        return X, y.astype(int)
    
    def load_or_train_model(self):
        """Load existing model or train a new one"""
        model_path = 'iss_visibility_model.joblib'
        
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            # Generate training data and train model
            X, y = self.generate_training_data()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Save model
            joblib.dump(self.model, model_path)
            
            # Print accuracy
            accuracy = self.model.score(X_test, y_test)
            print(f"Model trained with accuracy: {accuracy:.2f}")
    
    def predict_visibility(self, weather_data, elevation_angle, light_pollution_score=5):
        """Predict ISS visibility based on conditions"""
        if not weather_data:
            return None
        
        # Extract weather features
        cloud_cover = weather_data.get('clouds', {}).get('all', 50)
        humidity = weather_data.get('main', {}).get('humidity', 70)
        wind_speed = weather_data.get('wind', {}).get('speed', 10) * 3.6  # Convert m/s to km/h
        
        # Prepare features for prediction
        features = np.array([[cloud_cover, humidity, wind_speed, light_pollution_score, elevation_angle]])
        
        # Get prediction and probability
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        return {
            'visible': bool(prediction),
            'visibility_probability': float(probability[1]),
            'conditions': {
                'cloud_cover': cloud_cover,
                'humidity': humidity,
                'wind_speed': round(wind_speed, 1),
                'light_pollution': light_pollution_score,
                'elevation_angle': elevation_angle
            }
        }

# Initialize AI predictor
visibility_predictor = ISSVisibilityPredictor()

def get_weather_data(lat, lon):
    """Get current weather data for location"""
    try:
        if not WEATHER_API_KEY or WEATHER_API_KEY == "your_openweather_api_key":
            # Return mock data if no API key
            return {
                'clouds': {'all': np.random.randint(0, 80)},
                'main': {'humidity': np.random.randint(40, 90)},
                'wind': {'speed': np.random.randint(1, 15)}
            }
        
        params = {
            'lat': lat,
            'lon': lon,
            'appid': WEATHER_API_KEY
        }
        response = requests.get(WEATHER_API, params=params, timeout=5)
        return response.json()
    except Exception as e:
        print(f"Weather API error: {e}")
        return None

def calculate_elevation_angle(iss_lat, iss_lon, user_lat, user_lon):
    """Calculate elevation angle of ISS from user location (simplified)"""
    # Simplified calculation - in reality this is much more complex
    lat_diff = abs(iss_lat - user_lat)
    lon_diff = abs(iss_lon - user_lon)
    distance_factor = (lat_diff + lon_diff) / 2
    
    # Mock elevation angle based on distance
    if distance_factor < 10:
        return np.random.randint(60, 90)  # High elevation
    elif distance_factor < 30:
        return np.random.randint(30, 60)  # Medium elevation
    else:
        return np.random.randint(10, 30)  # Low elevation

@app.route('/')
def index():
    return render_template('ai_index.html')

@app.route('/api/iss-location')
def iss_location():
    """Get current ISS location"""
    try:
        response = requests.get(ISS_LOCATION_API, timeout=5)
        data = response.json()
        
        if data['message'] == 'success':
            return jsonify({
                'latitude': float(data['iss_position']['latitude']),
                'longitude': float(data['iss_position']['longitude']),
                'timestamp': data['timestamp'],
                'datetime': datetime.fromtimestamp(data['timestamp']).strftime('%Y-%m-%d %H:%M:%S UTC')
            })
    except Exception as e:
        return jsonify({'error': f'Unable to fetch ISS location: {e}'}), 500

@app.route('/api/ai-visibility-prediction', methods=['POST'])
def ai_visibility_prediction():
    """AI-powered ISS visibility prediction"""
    try:
        data = request.json
        user_lat = data.get('lat')
        user_lon = data.get('lon')
        light_pollution = data.get('light_pollution', 5)  # User can input their area's light pollution
        
        # Get current ISS location
        iss_response = requests.get(ISS_LOCATION_API, timeout=5)
        iss_data = iss_response.json()
        
        if iss_data['message'] != 'success':
            return jsonify({'error': 'Could not get ISS location'}), 500
        
        iss_lat = float(iss_data['iss_position']['latitude'])
        iss_lon = float(iss_data['iss_position']['longitude'])
        
        # Get weather data
        weather_data = get_weather_data(user_lat, user_lon)
        
        # Calculate elevation angle
        elevation_angle = calculate_elevation_angle(iss_lat, iss_lon, user_lat, user_lon)
        
        # Get AI prediction
        prediction = visibility_predictor.predict_visibility(
            weather_data, elevation_angle, light_pollution
        )
        
        if prediction:
            return jsonify({
                'prediction': prediction,
                'iss_location': {'lat': iss_lat, 'lon': iss_lon},
                'message': f"🤖 AI Analysis: {prediction['visibility_probability']:.0%} chance of seeing the ISS!"
            })
        else:
            return jsonify({'error': 'Could not generate prediction'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500

@app.route('/api/smart-notifications', methods=['POST'])
def smart_notifications():
    """AI-powered smart notification system"""
    try:
        data = request.json
        user_lat = data.get('lat')
        user_lon = data.get('lon')
        
        # Get upcoming ISS passes
        params = {'lat': user_lat, 'lon': user_lon, 'n': 5}
        response = requests.get(ISS_PASS_API, params=params, timeout=5)
        passes_data = response.json()
        
        if passes_data['message'] != 'success':
            return jsonify({'error': 'Could not get ISS passes'}), 500
        
        smart_recommendations = []
        
        for pass_info in passes_data['response']:
            pass_time = datetime.fromtimestamp(pass_info['risetime'])
            
            # Mock weather prediction for pass time
            mock_weather = {
                'clouds': {'all': np.random.randint(0, 80)},
                'main': {'humidity': np.random.randint(40, 90)},
                'wind': {'speed': np.random.randint(1, 15)}
            }
            
            # Predict visibility for this pass
            prediction = visibility_predictor.predict_visibility(
                mock_weather, np.random.randint(20, 80), 5
            )
            
            recommendation = {
                'datetime': pass_time.strftime('%Y-%m-%d %H:%M:%S UTC'),
                'duration': pass_info['duration'],
                'visibility_score': prediction['visibility_probability'] if prediction else 0.5,
                'ai_recommendation': 'Highly Recommended! 🌟' if prediction and prediction['visibility_probability'] > 0.7 
                                   else 'Good Conditions ✨' if prediction and prediction['visibility_probability'] > 0.5
                                   else 'Poor Visibility ☁️'
            }
            smart_recommendations.append(recommendation)
        
        # Sort by visibility score
        smart_recommendations.sort(key=lambda x: x['visibility_score'], reverse=True)
        
        return jsonify({
            'recommendations': smart_recommendations,
            'best_pass': smart_recommendations[0] if smart_recommendations else None,
            'message': '🤖 AI has analyzed weather patterns and orbital data to rank your best viewing opportunities!'
        })
        
    except Exception as e:
        return jsonify({'error': f'Smart notification failed: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)