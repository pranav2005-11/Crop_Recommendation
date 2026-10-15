import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify
import numpy as np
import pandas
import sklearn
import pickle
import hashlib
import sqlite3
import requests
from functools import wraps
import os
from datetime import datetime, timedelta
from PIL import Image
import json
from werkzeug.utils import secure_filename

# importing model
model = pickle.load(open('model.pkl','rb'))

# Crop Disease Detection model is loaded lazily so the rest of the app
# runs without TensorFlow installed.
_disease_model = None

def get_disease_model():
    global _disease_model
    if _disease_model is None:
        import tensorflow as tf
        _disease_model = tf.keras.models.load_model('crop_disease_model.keras')
    return _disease_model
with open('class_names.json', 'r') as f:
    disease_class_names = json.load(f)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Basic remedy/treatment info per class (crop_disease -> advice)
DISEASE_INFO = {
    "Rice_BacterialBlight": "Use resistant varieties, avoid excess nitrogen, ensure proper field drainage. Apply copper-based bactericides if severe.",
    "Rice_Blast": "Apply Tricyclazole or Isoprothiolane fungicide. Avoid excess nitrogen and maintain proper water management.",
    "Rice_BrownSpot": "Apply Mancozeb or Propiconazole. Improve soil fertility, especially potassium levels.",
    "Rice_Tungro": "Remove and destroy infected plants. Control leafhopper vectors with insecticides. Use resistant varieties.",
    "Mango_Anthracnose": "Apply Carbendazim or Copper oxychloride spray. Prune affected branches and improve air circulation.",
    "Mango_BacterialCanker": "Apply copper-based bactericide. Remove and burn infected twigs. Avoid overhead irrigation.",
    "Mango_CuttingWeevil": "Remove and destroy infested shoots. Apply recommended insecticide (e.g., Chlorpyrifos) on new flush.",
    "Mango_DieBack": "Prune dead/infected wood well below the affected area. Apply Carbendazim paste on cut surfaces.",
    "Mango_GallMidge": "Collect and destroy fallen infested leaves/flowers. Apply Dimethoate or equivalent insecticide.",
    "Mango_Healthy": "No disease detected. Maintain regular care and monitoring.",
    "Mango_PowderyMildew": "Apply Wettable Sulphur or Hexaconazole spray. Ensure good air circulation via pruning.",
    "Mango_SootyMould": "Control associated sap-sucking insects (aphids, scale). Wash affected leaves and apply neem oil.",
    "Coconut_BudRootDropping": "Remove and destroy affected palms. Apply neem-based insecticide to control mite/insect vectors.",
    "Coconut_BudRot": "Remove and burn infected bud tissue. Apply Bordeaux mixture as a protective drench.",
    "Coconut_GrayLeafSpot": "Apply Mancozeb or Copper oxychloride spray. Improve field drainage and avoid water stagnation.",
    "Coconut_LeafRot": "Remove affected fronds. Apply Bordeaux mixture and improve plantation sanitation.",
    "Coconut_StemBleeding": "Scrape affected bark and apply Bordeaux paste. Avoid mechanical injuries to trunk.",
    "Cashew_anthracnose": "Apply Copper oxychloride or Carbendazim spray during flowering and fruiting stages.",
    "Cashew_gumosis": "Avoid trunk injuries. Apply Bordeaux paste on affected area, improve drainage.",
    "Cashew_healthy": "No disease detected. Maintain regular care and monitoring.",
    "Cashew_leaf_miner": "Apply neem-based insecticide or Dimethoate. Remove and destroy heavily infested leaves.",
    "Cashew_red_rust": "Apply Copper oxychloride spray. Ensure adequate spacing for air circulation.",
    "Banana_Cordana": "Remove infected leaves. Apply Mancozeb or Propiconazole fungicide spray.",
    "Banana_Healthy": "No disease detected. Maintain regular care and monitoring.",
    "Banana_Pestalotiopsis": "Improve drainage, avoid leaf injury. Apply Copper oxychloride if severe.",
    "Banana_Sigatoka": "Apply Propiconazole or Mancozeb fungicide. Remove and destroy heavily infected leaves.",
    "Sugarcane_Healthy": "No disease detected. Maintain regular care and monitoring.",
    "Sugarcane_Mosaic": "Use disease-free planting material. Control aphid vectors. Remove infected clumps.",
    "Sugarcane_RedRot": "Use resistant varieties. Avoid waterlogging. Remove and burn infected clumps.",
    "Sugarcane_Rust": "Apply Mancozeb or Propiconazole spray. Ensure balanced fertilization.",
    "Sugarcane_Yellow": "Improve field drainage, ensure balanced nutrition, control leafhopper vectors.",
}

# Weather-based disease risk rules: (min_humidity, min_temp, max_temp, risk_level, diseases, advice)
DISEASE_RISK_RULES = [
    {
        'condition': lambda h, t, r: h >= 80 and 20 <= t <= 30,
        'level': 'High',
        'diseases': ['Rice Blast', 'Rice Bacterial Blight', 'Mango Anthracnose', 'Sugarcane Rust'],
        'advice': 'High humidity and warm temperatures favor fungal and bacterial spread. Inspect crops closely and consider a preventive fungicide spray.'
    },
    {
        'condition': lambda h, t, r: h >= 70 and r >= 2.5,
        'level': 'Moderate',
        'diseases': ['Sugarcane Red Rot', 'Banana Sigatoka', 'Coconut Leaf Rot'],
        'advice': 'Persistent moisture and rainfall raise the chance of leaf and root diseases. Ensure proper field drainage.'
    },
    {
        'condition': lambda h, t, r: h < 50 and t >= 32,
        'level': 'Low',
        'diseases': [],
        'advice': 'Hot, dry conditions are generally less favorable for common fungal diseases. Continue routine monitoring.'
    },
]

def assess_disease_risk(humidity, temperature, rainfall):
    for rule in DISEASE_RISK_RULES:
        if rule['condition'](humidity, temperature, rainfall):
            return {
                'level': rule['level'],
                'diseases': rule['diseases'],
                'advice': rule['advice']
            }
    return {
        'level': 'Low',
        'diseases': [],
        'advice': 'Current conditions do not strongly favor any known disease. Continue routine monitoring.'
    }

def get_crop_and_disease(class_name):
    parts = class_name.split('_', 1)
    crop = parts[0]
    disease = parts[1] if len(parts) > 1 else "Unknown"
    return crop, disease

def prepare_image(img_path):
    from tensorflow.keras.applications.efficientnet import preprocess_input
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

sc = pickle.load(open('standscaler.pkl','rb'))
ms = pickle.load(open('minmaxscaler.pkl','rb'))

# Weather API configuration
WEATHER_API_KEY = 'demo_key'  # Using demo key for testing
WEATHER_API_URL = 'http://api.openweathermap.org/data/2.5/weather'

# Kokan region coordinates (approximate)
KOKAN_REGIONS = {
    'ratnagiri': {'lat': 16.99, 'lon': 73.30, 'name': 'Ratnagiri'},
    'sindhudurg': {'lat': 15.86, 'lon': 73.88, 'name': 'Sindhudurg'},
    'mumbai': {'lat': 19.07, 'lon': 72.87, 'name': 'Mumbai'},
    'thane': {'lat': 19.20, 'lon': 72.98, 'name': 'Thane'},
    'palghar': {'lat': 19.68, 'lon': 72.77, 'name': 'Palghar'}
}

# creating flask app
app = Flask(__name__)
app.secret_key = 'crop_recommendation_secret_key_2024'

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Database setup
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Backfill is_admin for databases created before this column existed
    c.execute("PRAGMA table_info(users)")
    columns = [row[1] for row in c.fetchall()]
    if 'is_admin' not in columns:
        c.execute('ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0')

    c.execute('''
        CREATE TABLE IF NOT EXISTS unknown_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            username TEXT NOT NULL,
            image_path TEXT NOT NULL,
            predicted_class TEXT,
            confidence REAL,
            user_suggested_label TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Confidence below this means "doesn't confidently match a known disease"
DISEASE_CONFIDENCE_THRESHOLD = 60.0

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Admin required decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        if not session.get('is_admin'):
            flash('Admin access required.', 'error')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        remember = request.form.get('remember')
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT id, password, is_admin FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        conn.close()

        if user and hashlib.sha256(password.encode()).hexdigest() == user[1]:
            session['user_id'] = user[0]
            session['username'] = username
            session['is_admin'] = bool(user[2])

            if remember:
                session.permanent = True
                app.permanent_session_lifetime = timedelta(days=30)
            
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('signup.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long', 'error')
            return render_template('signup.html')
        
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            c.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                     (username, email, hashed_password))
            conn.commit()
            conn.close()
            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists', 'error')
    
    return render_template('signup.html')

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT username FROM users WHERE email = ?', (email,))
        user = c.fetchone()
        conn.close()
        
        if user:
            flash(f'Password reset instructions sent to {email}', 'success')
        else:
            flash('Email not found', 'error')
    
    return render_template('forgot_password.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=session['username'])

@app.route('/history')
@login_required
def history():
    conn = sqlite3.connect('users.db')
    return render_template('history.html', username=session['username'])

@app.route('/profile')
@login_required
def profile():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT username, email, created_at FROM users WHERE id = ?', (session['user_id'],))
    user = c.fetchone()
    conn.close()
    
    if user:
        user_data = {
            'username': user[0],
            'email': user[1],
            'created_at': user[2]
        }
        return render_template('profile.html', user=user_data, username=session['username'])
    else:
        flash('User not found', 'error')
        return redirect(url_for('dashboard'))

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    flash('Profile update functionality coming soon!', 'info')
    return redirect(url_for('profile'))

@app.route('/change_password', methods=['POST'])
@login_required
def change_password():
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')
    
    if not current_password or not new_password or not confirm_password:
        flash('All password fields are required', 'error')
        return redirect(url_for('profile'))
    
    if new_password != confirm_password:
        flash('New passwords do not match', 'error')
        return redirect(url_for('profile'))
    
    if len(new_password) < 6:
        flash('Password must be at least 6 characters long', 'error')
        return redirect(url_for('profile'))
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT password FROM users WHERE id = ?', (session['user_id'],))
    user = c.fetchone()
    
    if user and hashlib.sha256(current_password.encode()).hexdigest() == user[0]:
        hashed_new_password = hashlib.sha256(new_password.encode()).hexdigest()
        c.execute('UPDATE users SET password = ? WHERE id = ?', (hashed_new_password, session['user_id']))
        conn.commit()
        conn.close()
        flash('Password updated successfully!', 'success')
    else:
        conn.close()
        flash('Current password is incorrect', 'error')
    
    return redirect(url_for('profile'))

@app.route('/export_data')
@login_required
def export_data():
    flash('Data export functionality coming soon!', 'info')
    return redirect(url_for('profile'))

@app.route('/delete_account', methods=['POST'])
@login_required
def delete_account():
    flash('Account deletion functionality coming soon!', 'info')
    return redirect(url_for('profile'))

@app.route('/get_weather/<region>')
@login_required
def get_weather(region):
    """Get comprehensive weather data for Kokan region"""
    if region not in KOKAN_REGIONS:
        return jsonify({'error': 'Invalid region'}), 400
    
    try:
        coords = KOKAN_REGIONS[region]
        region_name = coords['name']
        
        comprehensive_weather = {
            'ratnagiri': {
                'current': {'temp': 28.5, 'humidity': 75, 'rain': 2.5, 'desc': 'partly cloudy'},
                'avg': {'summer': 32.5, 'monsoon': 28.0, 'winter': 24.5, 'annual': 28.3},
                'rainfall': {'current': 2.5, 'monthly': 450, 'annual': 3200},
                'conditions': {'current': 'partly cloudy', 'forecast': 'light rain expected', 'season': 'pre-monsoon'}
            },
            'sindhudurg': {
                'current': {'temp': 27.8, 'humidity': 78, 'rain': 3.2, 'desc': 'humid'},
                'avg': {'summer': 31.8, 'monsoon': 27.5, 'winter': 24.0, 'annual': 27.8},
                'rainfall': {'current': 3.2, 'monthly': 480, 'annual': 3500},
                'conditions': {'current': 'humid', 'forecast': 'moderate rain', 'season': 'pre-monsoon'}
            },
            'mumbai': {
                'current': {'temp': 30.2, 'humidity': 70, 'rain': 1.8, 'desc': 'hazy'},
                'avg': {'summer': 33.0, 'monsoon': 29.5, 'winter': 25.0, 'annual': 29.2},
                'rainfall': {'current': 1.8, 'monthly': 380, 'annual': 2400},
                'conditions': {'current': 'hazy', 'forecast': 'clear skies', 'season': 'summer'}
            },
            'thane': {
                'current': {'temp': 29.5, 'humidity': 72, 'rain': 2.1, 'desc': 'partly cloudy'},
                'avg': {'summer': 32.2, 'monsoon': 28.8, 'winter': 24.8, 'annual': 28.6},
                'rainfall': {'current': 2.1, 'monthly': 420, 'annual': 2800},
                'conditions': {'current': 'partly cloudy', 'forecast': 'light showers', 'season': 'pre-monsoon'}
            },
            'palghar': {
                'current': {'temp': 29.8, 'humidity': 74, 'rain': 2.8, 'desc': 'overcast'},
                'avg': {'summer': 32.8, 'monsoon': 29.2, 'winter': 25.2, 'annual': 29.1},
                'rainfall': {'current': 2.8, 'monthly': 460, 'annual': 3000},
                'conditions': {'current': 'overcast', 'forecast': 'rain expected', 'season': 'pre-monsoon'}
            }
        }
        
        weather_data = comprehensive_weather.get(region, comprehensive_weather['mumbai'])

        disease_risk = assess_disease_risk(
            weather_data['current']['humidity'],
            weather_data['current']['temp'],
            weather_data['current']['rain']
        )

        response_data = {
            'temperature': weather_data['current']['temp'],
            'humidity': weather_data['current']['humidity'],
            'rainfall': weather_data['current']['rain'],
            'description': weather_data['conditions']['current'],
            'location': region_name,
            'avg_temp': weather_data['avg']['annual'],
            'seasonal_avg': weather_data['avg'],
            'monthly_rainfall': weather_data['rainfall']['monthly'],
            'annual_rainfall': weather_data['rainfall']['annual'],
            'forecast': weather_data['conditions']['forecast'],
            'season': weather_data['conditions']['season'],
            'disease_risk': disease_risk,
            'region_info': {
                'name': region_name,
                'coordinates': {'lat': coords['lat'], 'lon': coords['lon']},
                'climate_zone': 'Coastal Kokan',
                'soil_type': 'Lateritic Red Soil',
                'cropping_season': 'Kharif (Jun-Oct), Rabi (Oct-Feb)'
            }
        }

        return jsonify(response_data)
            
    except Exception as e:
        fallback_data = {
            'temperature': 28.0,
            'humidity': 75.0,
            'rainfall': 2.0,
            'description': 'typical Kokan weather',
            'location': KOKAN_REGIONS[region]['name'],
            'avg_temp': 28.5,
            'seasonal_avg': {'summer': 32.0, 'monsoon': 28.0, 'winter': 24.5, 'annual': 28.2},
            'monthly_rainfall': 400,
            'annual_rainfall': 2800,
            'forecast': 'stable conditions',
            'season': 'pre-monsoon',
            'disease_risk': assess_disease_risk(75.0, 28.0, 2.0),
            'region_info': {
                'name': KOKAN_REGIONS[region]['name'],
                'coordinates': {'lat': KOKAN_REGIONS[region]['lat'], 'lon': KOKAN_REGIONS[region]['lon']},
                'climate_zone': 'Coastal Kokan',
                'soil_type': 'Lateritic Red Soil',
                'cropping_season': 'Kharif (Jun-Oct), Rabi (Oct-Feb)'
            }
        }
        return jsonify(fallback_data)

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route("/predict", methods=['POST'])
@login_required
def predict():
    recommendation_region = request.form.get('recommendation_region')
    print(f"DEBUG: Received recommendation_region value: '{recommendation_region}'")
    print(f"DEBUG: Recommendation region type: {type(recommendation_region)}")
    print(f"DEBUG: All form data: {dict(request.form)}")
    
    if recommendation_region == 'None' or recommendation_region == 'null' or recommendation_region == '' or not recommendation_region:
        print(f"DEBUG: Recommendation region validation failed. Region: '{recommendation_region}', KOKAN_REGIONS keys: {list(KOKAN_REGIONS.keys())}")
        flash('⚠️ Please select your recommendation region to get accurate crop recommendations. Location is required for precise agricultural advice.', 'error')
        return render_template('dashboard.html', username=session['username'])
    
    if recommendation_region not in KOKAN_REGIONS:
        print(f"DEBUG: Invalid recommendation region: '{recommendation_region}' not in KOKAN_REGIONS")
        flash('⚠️ Invalid recommendation region selected. Please select a valid Kokan region.', 'error')
        return render_template('dashboard.html', username=session['username'])
    
    print(f"DEBUG: Recommendation region validation passed: {recommendation_region}")
    
    region = recommendation_region
    
    try:
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])
        npk_ratio = request.form['NPK_ratio']
        temp_humidity_index = float(request.form['temp_humidity_index'])
        soil_fertility_score = float(request.form['soil_fertility_score'])
    except ValueError:
        flash('Please enter valid numeric values for all parameters.', 'error')
        return render_template('dashboard.html', username=session['username'])
    
    errors = []
    
    if not (0 <= N <= 500):
        errors.append('Nitrogen must be between 0 and 500 kg/ha')
    if not (0 <= P <= 300):
        errors.append('Phosphorus must be between 0 and 300 kg/ha')
    if not (0 <= K <= 400):
        errors.append('Potassium must be between 0 and 400 kg/ha')
    if not (15 <= temp <= 45):
        errors.append('Temperature must be between 15°C and 45°C')
    if not (0 <= humidity <= 100):
        errors.append('Humidity must be between 0% and 100%')
    if not (1 <= ph <= 14):
        errors.append('pH value must be between 1 and 14')
    if not (0 <= rainfall <= 4500):
        errors.append('Rainfall must be between 0 and 4500 mm (Kokan region has very high rainfall)')
    
    if errors:
        flash('Validation errors: ' + '; '.join(errors), 'error')
        return render_template('dashboard.html', username=session['username'])

    try:
        if ':' in npk_ratio:
            npk_parts = npk_ratio.split(':')
            npk_numeric = sum(float(part) for part in npk_parts) / len(npk_parts)
        else:
            npk_numeric = float(npk_ratio)
    except:
        npk_numeric = 0.0

    region_info = KOKAN_REGIONS[region]
    region_name = region_info['name']
    
    location_crop_preferences = {
        'ratnagiri': {
            'preferred_crops': [1, 5, 12, 13, 22],
            'soil_type': 'Lateritic Red Soil',
            'specialty': 'Mango and Cashew cultivation'
        },
        'sindhudurg': {
            'preferred_crops': [1, 5, 12, 13, 22],
            'soil_type': 'Coastal Alluvial',
            'specialty': 'Mango and Kokam'
        },
        'mumbai': {
            'preferred_crops': [2, 6, 11, 12, 13],
            'soil_type': 'Coastal Sandy',
            'specialty': 'Vegetables and Fruits'
        },
        'thane': {
            'preferred_crops': [1, 5, 12, 13, 14],
            'soil_type': 'Lateritic Red Soil',
            'specialty': 'Rice and Vegetables'
        },
        'palghar': {
            'preferred_crops': [1, 5, 12, 13, 15],
            'soil_type': 'Lateritic Red Soil',
            'specialty': 'Rice and Pulses'
        }
    }

    feature_list = [N, P, K, temp, humidity, ph, rainfall, npk_numeric, temp_humidity_index, soil_fertility_score]
    print(f"DEBUG: Feature list: {feature_list}")
    
    single_pred = np.array(feature_list).reshape(1, -1)
    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)
    print(f"DEBUG: Model prediction: {prediction}")
    print(f"DEBUG: Prediction value: {prediction[0]}")

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}
    
    print(f"DEBUG: Crop dictionary: {crop_dict}")
    print(f"DEBUG: Looking for crop with prediction number: {prediction[0]}")

    location_prefs = location_crop_preferences.get(region, {})
    preferred_crops = location_prefs.get('preferred_crops', [])
    soil_type = location_prefs.get('soil_type', 'Kokan Coastal Soil')
    specialty = location_prefs.get('specialty', 'Mixed Cropping')
    
    if prediction[0] in crop_dict:
        predicted_crop = crop_dict[prediction[0]]
        
        if (N >= 80 and N <= 120 and 
            P >= 40 and P <= 60 and 
            K >= 40 and K <= 80 and 
            ph >= 5.5 and ph <= 6.5 and 
            rainfall >= 2500 and rainfall <= 4000):
            predicted_crop = "Rice"
            prediction[0] = 1
            print(f"DEBUG: Overriding to Rice based on optimal values")
        
        if prediction[0] in preferred_crops:
            result = f"{predicted_crop} is highly recommended for {region_name} region! This crop thrives in {soil_type} and matches the local {specialty} patterns."
            flash(f'Excellent choice! {predicted_crop} is perfect for {region_name} region and your soil conditions.', 'success')
        else:
            if preferred_crops:
                alt_crop = crop_dict[preferred_crops[0]]
                result = f"{predicted_crop} is suitable for your conditions, but for {region_name} region, we especially recommend {alt_crop} which is better adapted to {soil_type} and local {specialty}."
                flash(f'Recommendation: {predicted_crop} works, but {alt_crop} would be ideal for {region_name} region.', 'info')
            else:
                result = f"{predicted_crop} is the best crop to be cultivated in the Kokan region with these conditions."
                flash(f'Recommendation successful! {predicted_crop} is suitable for your conditions.', 'success')
    else:
        result = f"Sorry, we could not determine the best crop to be cultivated with the provided data for {region_name} region."
        flash('Unable to determine suitable crop for the provided conditions.', 'warning')
        
    return render_template('dashboard.html', result=result, username=session['username'])

@app.route('/disease-detection')
@login_required
def disease_detection():
    return render_template('disease.html', username=session['username'])

@app.route('/predict-disease', methods=['POST'])
@login_required
def predict_disease():
    if 'leaf_image' not in request.files:
        flash('No file uploaded.', 'error')
        return redirect(url_for('disease_detection'))

    file = request.files['leaf_image']

    if file.filename == '':
        flash('No file selected.', 'error')
        return redirect(url_for('disease_detection'))

    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload a JPG or PNG image.', 'error')
        return redirect(url_for('disease_detection'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        img_array = prepare_image(filepath)
        predictions = get_disease_model().predict(img_array)
        predicted_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_index]) * 100
        predicted_class = disease_class_names[predicted_index]

        if confidence < DISEASE_CONFIDENCE_THRESHOLD:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute(
                'INSERT INTO unknown_reports (user_id, username, image_path, predicted_class, confidence) VALUES (?, ?, ?, ?, ?)',
                (session['user_id'], session['username'], filepath, predicted_class, round(confidence, 2))
            )
            conn.commit()
            report_id = c.lastrowid
            conn.close()

            return render_template(
                'disease.html',
                username=session['username'],
                uploaded_image=filepath,
                confidence=round(confidence, 2),
                is_unknown=True,
                report_id=report_id
            )

        crop, disease = get_crop_and_disease(predicted_class)
        advice = DISEASE_INFO.get(predicted_class, "No specific advice available for this condition.")
        is_healthy = 'healthy' in disease.lower()

        return render_template(
            'disease.html',
            username=session['username'],
            uploaded_image=filepath,
            crop=crop,
            disease=disease.replace('_', ' '),
            confidence=round(confidence, 2),
            advice=advice,
            is_healthy=is_healthy
        )
    except Exception as e:
        flash(f'Error processing image: {str(e)}', 'error')
        return redirect(url_for('disease_detection'))

@app.route('/report-unknown', methods=['POST'])
@login_required
def report_unknown():
    report_id = request.form.get('report_id')
    suggested_label = request.form.get('suggested_label', '').strip()

    if not report_id or not suggested_label:
        flash('Please describe what you think this condition is.', 'error')
        return redirect(url_for('disease_detection'))

    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute(
        'UPDATE unknown_reports SET user_suggested_label = ? WHERE id = ? AND user_id = ?',
        (suggested_label, report_id, session['user_id'])
    )
    conn.commit()
    conn.close()

    flash('Thanks! Your input has been recorded and will help us improve the model.', 'success')
    return redirect(url_for('disease_detection'))

@app.route('/admin/unknown-reports')
@admin_required
def admin_unknown_reports():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM unknown_reports ORDER BY created_at DESC')
    reports = c.fetchall()
    conn.close()
    return render_template('admin_reports.html', username=session['username'], reports=reports)

@app.route('/admin/unknown-reports/<int:report_id>/resolve', methods=['POST'])
@admin_required
def resolve_unknown_report(report_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("UPDATE unknown_reports SET status = 'reviewed' WHERE id = ?", (report_id,))
    conn.commit()
    conn.close()
    flash('Report marked as reviewed.', 'success')
    return redirect(url_for('admin_unknown_reports'))

# python main
if __name__ == "__main__":
    app.run(debug=True)