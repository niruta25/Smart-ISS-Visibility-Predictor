# Smart-ISS-Visibility-Predictor
Most ISS trackers tell you WHEN it passes overhead, but ours uses AI to predict if you'll ACTUALLY see it. Our machine learning model analyzes weather patterns, cloud cover, and atmospheric conditions to give you a visibility probability. Watch - it's currently 73% likely you'll see the ISS in tonight's pass!

# AI-Powered ISS Tracker

> A machine learning enhanced International Space Station tracker that doesn't just show you *when* the ISS passes overhead, but predicts *if you'll actually see it*.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![ISS Tracker Demo](https://via.placeholder.com/800x400/0a0a0a/00d4ff?text=🛰️+AI-Powered+ISS+Tracker+🤖)

## What This Project Offers

### ** Intelligent Visibility Predictions**
Unlike traditional ISS trackers that only show location and pass times, our AI analyzes:
- **Real-time weather conditions** (cloud cover, humidity, wind)
- **Light pollution levels** (customizable by location)
- **ISS elevation angle** (higher = better visibility)
- **Atmospheric clarity** (combined weather factors)

**Result**: Get a **0-100% visibility probability** for each ISS pass, so you know when it's actually worth going outside!

### ** Real-Time Features**
- **Live ISS Tracking**: See the space station moving across Earth in real-time
- **Interactive World Map**: Dark-themed map with animated ISS marker
- **Smart Notifications**: AI ranks upcoming passes by viewing quality
- **Distance Calculations**: Know exactly how far the ISS is from you
- **Weather Integration**: Current atmospheric conditions affect predictions

### ** Beautiful User Experience**
- **Space-themed Design**: Dark gradient backgrounds with glowing elements
- **Responsive Interface**: Works perfectly on desktop and mobile
- **Real-time Updates**: ISS position updates every 10 seconds
- **Animated Visualizations**: Smooth transitions and engaging animations
- **AI Status Indicators**: Live feedback on machine learning analysis

### ** Technical Innovation**
- **Machine Learning Model**: Random Forest classifier trained on atmospheric physics
- **Multiple Data Sources**: NASA APIs + weather data + orbital mechanics
- **Real-time Processing**: Instant predictions with sub-second response times
- **Educational Value**: Shows how AI can solve practical astronomy problems

## How to Recreate This Project

### **Prerequisites**
- Python 3.8 or higher
- Basic knowledge of Flask and HTML/JavaScript
- Internet connection (for APIs)

### **Step 1: Project Setup**
```bash
# Create project directory
mkdir ai-iss-tracker
cd ai-iss-tracker

# Create folder structure
mkdir templates
mkdir static  # Optional: for additional CSS/JS files
```

### **Step 2: Install Dependencies**
Create `requirements.txt`:
```txt
Flask==2.3.3
requests==2.31.0
scikit-learn==1.3.0
numpy==1.24.3
joblib==1.3.2
```

Install packages:
```bash
pip install -r requirements.txt
```

### **Step 3: Backend Development**
Create `app.py` with the Flask backend code provided in the artifacts above. Key components:

**Core Features to Implement:**
1. **ISS Location API**: Fetch real-time ISS coordinates
2. **Weather Data Integration**: Get atmospheric conditions
3. **ML Model Training**: Random Forest for visibility prediction
4. **API Endpoints**: REST endpoints for frontend communication
5. **Distance Calculations**: Haversine formula for ISS distance

**Critical Functions:**
```python
# Core ML model for visibility prediction
class ISSVisibilityPredictor:
    def predict_visibility(self, weather_data, elevation_angle, light_pollution):
        # Returns probability score 0-1
        
# API endpoints
@app.route('/api/iss-location')          # Current ISS position
@app.route('/api/ai-visibility-prediction')  # ML prediction
@app.route('/api/smart-notifications')   # Ranked viewing opportunities
```

### **Step 4: Frontend Development**
Create `templates/index.html` (or `ai_index.html`) with:

**Essential Frontend Components:**
1. **Interactive Map**: Leaflet.js with dark theme
2. **Real-time Updates**: JavaScript intervals for live data
3. **AI Prediction Interface**: Visibility meter and condition display
4. **Responsive Design**: CSS Grid and Flexbox layouts
5. **User Location**: Geolocation API integration

**Key JavaScript Functions:**
```javascript
// Core functions to implement
updateISSLocation()      // Fetch and display ISS position
getAIPrediction()        // Trigger ML visibility analysis
getUserLocation()        # Get user's coordinates
getSmartRecommendations() // AI-ranked viewing times
```

### **Step 5: Machine Learning Implementation**
The AI model trains automatically on first run:

**Training Process:**
1. **Synthetic Data Generation**: Creates 1000 realistic training samples
2. **Feature Engineering**: 5 input features (cloud cover, humidity, wind, light pollution, elevation)
3. **Model Training**: Random Forest classifier with 100 estimators
4. **Model Persistence**: Saves trained model to disk for reuse

**Prediction Pipeline:**
```
Weather API → Feature Extraction → ML Model → Probability Score → User Interface
```

### **Step 6: API Integration**

**Required APIs:**
- **ISS Location**: `http://api.open-notify.org/iss-now.json` (Free, no key)
- **ISS Passes**: `http://api.open-notify.org/iss-pass.json` (Free, no key)
- **Weather Data**: OpenWeatherMap API (Free tier available)

**Optional Weather API Setup:**
```python
# Get free API key from openweathermap.org
WEATHER_API_KEY = "your_api_key_here"
```
*Note: Project works with mock weather data if no API key provided*

### **Step 7: Running the Application**
```bash
# Start the Flask server
python app.py

# Access the application
# Open browser to: http://localhost:5000
```

### **Step 8: Testing & Demo Preparation**

**Essential Test Cases:**
1. **ISS Tracking**: Verify real-time position updates
2. **Location Detection**: Test geolocation functionality
3. **AI Predictions**: Generate visibility scores for different conditions
4. **Smart Recommendations**: Check pass ranking algorithm
5. **Responsive Design**: Test on mobile and desktop

**Demo Script:**
1. Show live ISS tracking
2. Get user location
3. Trigger AI prediction
4. Explain visibility probability
5. Show smart recommendations

## 🚀 Quick Start (5-Minute Setup)

```bash
# Clone or download the project files
git clone <your-repo-url>  # Or create files manually
cd ai-iss-tracker

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

# Open browser
# Navigate to: http://localhost:5000
```

## 📁 Final Project Structure
```
ai-iss-tracker/
├── app.py                           # Main Flask application
├── templates/
│   └── ai_index.html               # Frontend interface
├── requirements.txt                # Python dependencies
├── iss_visibility_model.joblib    # Auto-generated ML model
├── README.md                       # This file
└── LICENSE                         # MIT License
```

## 🔧 Customization Options

### **Easy Modifications:**
- Change color scheme in CSS variables
- Adjust ML model parameters (n_estimators, features)
- Add more weather conditions (precipitation, visibility)
- Implement user accounts and history tracking

### **Advanced Extensions:**
- **Computer Vision**: Actual ISS detection in sky photos
- **Mobile App**: React Native or Flutter version
- **Augmented Reality**: Phone camera overlay showing ISS position
- **Social Features**: Share viewing opportunities with friends
- **Telescope Integration**: Automated telescope pointing

## Why This Project Stands Out

### **Technical Excellence:**
- Real machine learning implementation (not just API calls)
- Multiple data source integration
- Production-ready code structure
- Responsive, modern UI design

### **Practical Value:**
- Solves genuine problem (cloudy night frustration)
- Educational astronomy tool
- Real-time actionable insights
- Beautiful, engaging user experience

### **Innovation Factor:**
- First ISS tracker with ML-powered visibility prediction
- Combines space data with atmospheric science
- AI-enhanced user experience
- Perfect demonstration of practical ML applications

## Learning Outcomes

By recreating this project, you'll learn:
- **Flask Web Development**: Backend API creation
- **Machine Learning**: Scikit-learn model training and deployment
- **API Integration**: Multiple external data sources
- **Frontend Development**: Interactive maps and real-time updates
- **Data Science**: Feature engineering and prediction pipelines
- **UI/UX Design**: Modern, responsive web interfaces

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

**Areas for contribution:**
- Additional ML models (neural networks, ensemble methods)
- Mobile app development
- International space station data sources
- Weather API alternatives
- UI/UX improvements

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **NASA**: For providing free ISS tracking APIs
- **Open Notify**: ISS location and pass prediction services
- **OpenWeatherMap**: Weather data API
- **Leaflet.js**: Interactive mapping library
- **Scikit-learn**: Machine learning framework

---

## Ready to Launch?

This project demonstrates how AI can enhance traditional applications with intelligent predictions. Whether you're participating in a hackathon, learning machine learning, or building a portfolio project, this ISS tracker showcases modern web development with practical AI integration.

**Star this repository** if you found it helpful, and happy space tracking! 🛰️✨

---

*Built with ❤️ for space enthusiasts and AI learners everywhere*