# ✈️ Flight Delay Prediction System

## 🎯 Project Overview

A complete **Machine Learning & Flask** web application that predicts whether a flight will be delayed or arrive on time based on various flight parameters like airline, source, destination, departure time, and distance.

This project is perfect for **college final year projects** or **assignments** as it demonstrates:
- Flask web development
- Machine Learning implementation
- End-to-end ML pipeline
- User authentication
- Responsive web design with Bootstrap

---



## 📊 Dataset Information

### Columns Description

| Column Name | Type | Description | Example Values |
|-------------|------|-------------|----------------|
| `Airline` | Categorical | Name of the airline | IndiGo, Air India, Vistara, SpiceJet, Akasa Air |
| `Source` | Categorical | Departure city | Mumbai, Delhi, Bangalore, Chennai, etc. |
| `Destination` | Categorical | Arrival city | Delhi, Bangalore, Chennai, Hyderabad, etc. |
| `DepartureHour` | Numerical | Hour of departure (24-hour format) | 0-23 (e.g., 6, 12, 18) |
| `DayOfWeek` | Numerical | Day of the week (1=Monday to 7=Sunday) | 1, 2, 3, 4, 5, 6, 7 |
| `Distance` | Numerical | Flight distance in kilometers | 150, 350, 840, 1150, 2200 |
| `Delayed` | Binary | Target variable (0=On Time, 1=Delayed) | 0 or 1 |

### Dataset Statistics
- **Total Records:** 150+ flights
- **Airlines:** 6 major Indian airlines
- **Cities:** 20+ Indian cities
- **Target Distribution:** Balanced (approx. 40% delayed, 60% on time)

---

## 🤖 Machine Learning Details

### Algorithm Used: **Random Forest Classifier**

Random Forest was chosen because:
- Handles both categorical and numerical features well
- Less prone to overfitting compared to single decision trees
- Provides feature importance rankings
- Good accuracy for classification problems

### ML Pipeline Steps:

1. **Data Loading:** Load CSV dataset
2. **Data Cleaning:** Remove duplicates, handle missing values
3. **Feature Encoding:** Convert categorical variables using Label Encoding
4. **Train-Test Split:** 80% training, 20% testing
5. **Model Training:** Random Forest with 100 trees
6. **Model Evaluation:** Calculate accuracy
7. **Model Saving:** Export model using pickle/joblib

---

## 🎨 Features Implemented

### 1. **User Authentication**
- Login page with username/password
- Session management
- Protected routes (only logged-in users can access)

### 2. **Home Page**
- Welcome message with username
- Real-time clock (updates every second)
- Current date display
- Navigation menu
- Quick stats about the system

### 3. **Flight Details Form**
- Airline selection (dropdown)
- Source airport selection
- Destination airport selection
- Departure hour input
- Day of week selection
- Distance input
- Form validation

### 4. **Prediction Result**
- Clear result display (On Time / Delayed)
- Confidence score
- Flight summary
- Easy navigation to make new predictions

### 5. **About Page**
- Project overview
- Technologies used
- Dataset information
- ML workflow explanation
- Step-by-step setup instructions
- Project structure diagram

---

## 📱 Screenshots

### Login Page
- Clean login form with demo credentials displayed

### Home Page
- Welcome message
- Real-time clock
- Quick action buttons
- System statistics

### Flight Details Form
- User-friendly form with dropdowns
- Input validation
- Helpful tooltips

### Result Page
- Clear prediction result
- Confidence percentage
- Flight summary
- Information about ML model

---

## 🔧 Customization

### Adding More Airlines
Edit `data/flights.csv` and add new airline data:
```csv
NewAirline,Mumbai,Delhi,10,1,1150,0
```

### Changing Credentials
Edit `app.py` and modify:
```python
USERNAME = 'your_username'
PASSWORD = 'your_password'
```

### Adding More Cities
Add new cities in `data/flights.csv` with proper encoding.

---

## 📝 Code Explanation (Hinglish)

### `app.py` (Flask Application)
```python
# Is file mein hamara Flask web application hai
# - login(): Login page render karta hai
# - home(): Home page dikhata hai with real-time clock
# - predict(): Flight form se data lekar prediction karta hai
# - logout(): Session clear karke login page par redirect karta hai
```

### `train_model.py` (ML Training)
```python
# Is file mein ham Machine Learning model train karte hain
# 1. Dataset load karte hain (CSV file)
# 2. Data cleaning (duplicates/hissing values remove)
# 3. Categorical encoding (LabelEncoder)
# 4. Train-test split (80-20)
# 5. Random Forest model train karte hain
# 6. Model save karte hain pickle use karke
```

### HTML Templates
```python
# Templates hamare web pages hain
# - login.html: Login form
# - home.html: Main dashboard
# - predict.html: Flight details input
# - result.html: Prediction output
# - about.html: Project information
```

---

## ⚠️ Troubleshooting

### Error: "Model not loaded!"
```bash
# Solution: Train the model first
python train_model.py
```

### Error: "Module not found"
```bash
# Solution: Install required libraries
pip install -r requirements.txt
```

### Error: "Port already in use"
```bash
# Solution: Change port in app.py
app.run(port=5001)  # Use different port
```

### Error: "Template not found"
```bash
# Solution: Check templates folder is in correct location
# Folder structure should be:
# project/
#   app.py
#   templates/
#     *.html files
```

---

## 📚 Learning Outcomes

After completing this project, you will understand:
- ✅ Flask web application development
- ✅ Machine Learning model training and deployment
- ✅ Data preprocessing and feature engineering
- ✅ User authentication and session management
- ✅ HTML/CSS/Bootstrap for responsive design
- ✅ End-to-end ML project workflow
- ✅ Model persistence with pickle/joblib

---

## 🤝 Contributing

This project is for educational purposes. Feel free to:
- Add more features
- Improve the ML model
- Enhance the UI
- Add more data

---

## 📄 License

This project is open source and available for educational use.

---

## 👨‍💻 Author

**AI/ML Developer**

Made with ❤️ for college students and ML enthusiasts!

---

## 🙏 Acknowledgments

- scikit-learn documentation
- Flask documentation
- Bootstrap framework
- Open source community

---

**⭐ Don't forget to star this repository if you found it helpful!**
