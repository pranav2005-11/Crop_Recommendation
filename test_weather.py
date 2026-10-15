import requests

# Test weather API functionality
WEATHER_API_URL = 'http://api.openweathermap.org/data/2.5/weather'

# Test with Mumbai coordinates
params = {
    'lat': 19.07,
    'lon': 72.87,
    'appid': 'demo_key',
    'units': 'metric'
}

print("Testing weather API...")
try:
    response = requests.get(WEATHER_API_URL, params=params, timeout=5)
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {response.headers}")
    
    if response.status_code == 200:
        data = response.json()
        print("✅ API Response:")
        print(f"Temperature: {data['main']['temp']}°C")
        print(f"Humidity: {data['main']['humidity']}%")
        print(f"Weather: {data['weather'][0]['description']}")
        print(f"Location: {data.get('name', 'Unknown')}")
    else:
        print(f"❌ API Error: {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"❌ Exception: {e}")

print("\nTesting fallback data...")
# Test fallback weather data
fallback_weather = {
    'mumbai': {'temp': 30.2, 'humidity': 70, 'rain': 1.8, 'desc': 'hazy'},
    'ratnagiri': {'temp': 28.5, 'humidity': 75, 'rain': 2.5, 'desc': 'partly cloudy'},
    'thane': {'temp': 29.5, 'humidity': 72, 'rain': 2.1, 'desc': 'partly cloudy'}
}

for region, weather in fallback_weather.items():
    print(f"{region}: {weather['temp']}°C, {weather['humidity']}%, {weather['rain']}mm, {weather['desc']}")
