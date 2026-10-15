# Crop Recommendation & Disease Detection System

## What is this project?
This is a smart farming assistant. A farmer or student enters some simple details (like soil nutrients, temperature, humidity, rainfall, or a photo of a crop leaf), and the system tells them:
1. **Which crop is best to grow** in those conditions, and
2. **Whether the crop leaf has a disease**, just by uploading a photo.

The goal is to help farmers make better decisions using Machine Learning (ML) instead of guesswork.

## Key Features
- **Crop Recommendation** – Enter values like Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, pH, and rainfall. The system predicts the most suitable crop to grow.
- **Crop Disease Detection** – Upload a photo of a plant leaf. A trained deep learning model (CNN, built with Keras/TensorFlow) identifies the disease from the image.
- **Live Weather Integration** – Fetches real-time weather for a region so recommendations can factor in current conditions.
- **User Accounts** – Sign up, log in, and manage a profile securely.
- **Prediction History** – Every past recommendation/detection a user made is saved and can be viewed later from their dashboard.
- **Dashboard** – A single place for the user to see their profile, history, and start new predictions.

## Technologies Used
- **Python** – core programming language
- **Scikit-learn** – ML models for crop recommendation (trained on the `Crop_recommendation.csv` dataset)
- **TensorFlow / Keras** – deep learning model (`crop_disease_model.keras`) for image-based disease detection
- **Pandas & NumPy** – data handling and preprocessing
- **Flask** – backend web framework
- **SQLite** – stores user accounts and prediction history (`users.db`)
- **HTML/CSS/JavaScript** – frontend pages (login, signup, dashboard, disease detection, etc.)

## How It Works (Simple Flow)
1. User logs in / signs up.
2. User either:
   - Enters soil and climate values → gets a **recommended crop**, or
   - Uploads a leaf photo → gets the **detected disease** and its likely cause.
3. The result is saved to the user's history and shown on their dashboard.

## Installation and Usage
```
git clone <repository-url>
cd Crop-Recommendation-System-Using-Machine-Learning-main
pip install -r requirements.txt
python app.py
```
Then open `http://localhost:5000` in your browser.

---

## Faculty Question: "What if a NEW disease appears that the model has never seen?"

This is an important real-world limitation of any image-based ML model: it can only recognize the diseases it was trained on (see `class_names.json`). If a brand-new or rare disease appears, the model may either misclassify it or force it into the closest known category. Here is how we explain and plan to handle this, in simple terms:

### 1. Confidence Score + "Unknown Disease" Flag
Instead of always giving a confident answer, the model can report **how sure it is** (e.g., "87% confident: Leaf Blight"). If the confidence is low (say below 60%), the system tells the user:
> "This doesn't clearly match any known disease. It may be a new or rare condition — please consult an expert."
This prevents the system from confidently giving a wrong answer.

### 2. Human-in-the-loop Feedback Learning
When the system is unsure (or when a user says "this prediction was wrong"), it saves the uploaded image into a "review" folder along with the user's correction. Periodically, this new data is used to **retrain and improve** the model — so the system keeps learning over time instead of staying frozen at its original training.

### 3. Expert/Community Verification Layer
Add an option where uncertain cases are forwarded to an agricultural expert or a community of verified users (like a crowdsourced second opinion), similar to how some plant-disease apps work. Their verified answer is fed back into the training data.

### 4. Regular Model Updates
Just like antivirus software updates its virus database, our model can be periodically retrained with newly confirmed disease images so it stays up to date with emerging diseases in the region.

---

## "Wow Factor" Features We Can Add

These are extra features that make the project stand out beyond a basic recommendation tool:

1. **Early Outbreak Alerts** – If many users in the same region report the same disease within a short time, automatically alert other nearby farmers, similar to how disease outbreaks spread. This turns individual predictions into a community early-warning system.
2. **Treatment & Prevention Suggestions** – After detecting a disease, don't just name it — suggest the pesticide/organic treatment and preventive steps to take.
3. **Multi-language & Voice Support** – Many farmers are more comfortable speaking than typing. Adding voice input and regional language support (Hindi, Marathi, etc.) makes the system accessible to more people.
4. **Offline/Low-network Mode** – A lightweight version of the disease detection model that works on a basic smartphone app even with poor internet, since rural areas often have limited connectivity.
5. **Yield & Profit Prediction** – Combine the crop recommendation with expected yield and current market price, so the farmer sees not just "what to grow" but "how much they could earn."
6. **Satellite/Drone Image Support** – Instead of a single leaf photo, allow a whole-field image (from drone or satellite) to detect disease spread across a larger area.
7. **Chatbot Assistant** – An AI chat assistant that farmers can ask questions to in plain language (e.g., "why is my tomato plant turning yellow?").

These features address the faculty's concern directly: the "Unknown Disease" flag and feedback-learning loop explain **how the system handles new diseases**, and the outbreak-alert / treatment-suggestion / chatbot ideas give the project a strong **wow factor** for evaluation.

## Future Enhancements
- Real-time weather-based recommendation improvements
- Crop market price and profitability analysis
- Mobile application
- Continuous learning from user feedback and corrections

## Contributing
Contributions are welcome. If you have suggestions, bug reports, or feature requests, please submit them through the issue tracker on the GitHub repository.

## Acknowledgements
We would like to thank the agricultural research community, farmers, and organizations for the insights, data, and domain knowledge that made this project possible.

## Contact
For any inquiries or questions, please contact us at 611noorsaeed@gmail.com

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
