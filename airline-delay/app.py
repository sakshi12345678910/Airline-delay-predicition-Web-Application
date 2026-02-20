# Import required libraries
from flask import Flask, render_template, request, redirect, url_for, session, flash
import pickle
import pandas as pd
import joblib
import json
import os
from datetime import datetime

# Initialize Flask application
app = Flask(__name__)

# Secret key for session management
app.secret_key = 'flight_delay_prediction_secret_key'

# Hardcoded admin credentials
USERNAME = 'admin'
PASSWORD = 'admin123'

# User data file
USERS_FILE = 'users.json'

# Load or initialize users
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

# Distance mapping (source -> destination -> distance)
# This will be loaded from the dataset
DISTANCE_MAP = {}

# Load trained model and required files
print("Loading ML model and encoders...")
try:
    model = pickle.load(open("model.pkl", "rb"))
    label_encoders = joblib.load("label_encoders.joblib")
    feature_columns = joblib.load("feature_columns.joblib")
    
    # Build distance map from dataset
    print("Building distance map...")
    df = pd.read_csv("data/flights.csv")
    distance_map = {}
    for _, row in df.iterrows():
        source = row['Source']
        dest = row['Destination']
        dist = row['Distance']
        key = (source, dest)
        if key not in distance_map:
            distance_map[key] = dist
    
    # Save distance map globally
    DISTANCE_MAP = distance_map
    print(f"Distance map created with {len(DISTANCE_MAP)} route(s)")
    
    print("Model and encoders loaded successfully!")
except Exception as e:
    print(f"Warning: {e}")
    print("   Please run 'python train_model.py' first!")
    model = None
    label_encoders = None
    feature_columns = None
    DISTANCE_MAP = {}


# =============================================================================
# ROUTE: Home / Login Page
# =============================================================================
@app.route("/", methods=["GET", "POST"])
def login():
    """
    Login page route.
    GET: Display login form
    POST: Validate credentials and redirect to home
    """
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        # Check if admin
        if username == USERNAME and password == PASSWORD:
            session["user"] = username
            session["role"] = "admin"
            flash("✅ Admin login successful!", "success")
            return redirect(url_for("home"))
        
        # Check registered users
        users = load_users()
        if username in users and users[username]["password"] == password:
            session["user"] = username
            session["role"] = "user"
            flash(f"✅ Welcome back, {username}!", "success")
            return redirect(url_for("home"))
        else:
            flash("❌ Invalid credentials! Try again.", "error")
    
    return render_template("login.html")


# =============================================================================
# ROUTE: Registration Page
# =============================================================================
@app.route("/register", methods=["GET", "POST"])
def register():
    """
    Registration page route.
    GET: Display registration form
    POST: Create new user account
    """
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        
        # Validation
        if password != confirm_password:
            flash("❌ Passwords do not match!", "error")
            return redirect(url_for("register"))
        
        # Check if username already exists
        users = load_users()
        if username in users:
            flash("❌ Username already exists!", "error")
            return redirect(url_for("register"))
        
        if username == USERNAME:
            flash("❌ Username 'admin' is reserved!", "error")
            return redirect(url_for("register"))
        
        # Save new user
        users[username] = {
            "email": email,
            "password": password
        }
        save_users(users)
        
        flash("✅ Registration successful! Please login.", "success")
        return redirect(url_for("login"))
    
    return render_template("register.html")


# =============================================================================
# ROUTE: Home Page
# =============================================================================
@app.route("/home")
def home():
    """
    Home page route.
    Displays welcome message, current date/time, and navigation menu.
    """
    # Check if user is logged in
    if "user" not in session:
        flash("⚠️ Please login first!", "warning")
        return redirect(url_for("login"))
    
    # Get current date and time
    now = datetime.now()
    current_date = now.strftime("%A, %d %B %Y")
    current_time = now.strftime("%H:%M:%S")
    
    return render_template(
        "home.html",
        date=current_date,
        time=current_time,
        username=session["user"]
    )


# =============================================================================
# ROUTE: Flight Details Page (Prediction Form)
# =============================================================================
@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Flight details input page route.
    GET: Display flight details form
    POST: Process form data and show prediction result
    """
    # Check if user is logged in
    if "user" not in session:
        flash("⚠️ Please login first!", "warning")
        return redirect(url_for("login"))
    
    if request.method == "POST":
        # Get form data
        airline = request.form.get("airline")
        source = request.form.get("source")
        destination = request.form.get("destination")
        departure_hour = int(request.form.get("departure_hour"))
        day_of_week = int(request.form.get("day_of_week"))
        
        # Auto-calculate distance based on source and destination
        distance = DISTANCE_MAP.get((source, destination))
        if distance is None:
            # If route not found, use average distance
            flash("⚠️ Route not found in database. Using estimated distance.", "warning")
            distance = 1500  # Default average distance
        
        # Prepare data for prediction
        try:
            # Encode categorical variables using saved encoders
            airline_encoded = label_encoders['Airline'].transform([airline])[0]
            source_encoded = label_encoders['Source'].transform([source])[0]
            destination_encoded = label_encoders['Destination'].transform([destination])[0]
            
            # Create DataFrame with proper feature order
            input_data = pd.DataFrame([[
                airline_encoded,
                source_encoded,
                destination_encoded,
                departure_hour,
                day_of_week,
                distance
            ]], columns=feature_columns)
            
            # Make prediction
            if model:
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0]
                
                # Prepare result message
                if prediction == 1:
                    result_text = "⚠️ Flight is DELAYED"
                    result_class = "delayed"
                    confidence = probability[1] * 100
                else:
                    result_text = "✅ Flight is ON TIME"
                    result_class = "ontime"
                    confidence = probability[0] * 100
                
                return render_template(
                    "result.html",
                    result=result_text,
                    result_class=result_class,
                    confidence=f"{confidence:.1f}",
                    airline=airline,
                    source=source,
                    destination=destination,
                    hour=departure_hour,
                    day=day_of_week,
                    distance=distance
                )
            else:
                flash("⚠️ Model not loaded!", "error")
                return redirect(url_for("predict"))
                
        except Exception as e:
            flash(f"❌ Error during prediction: {str(e)}", "error")
            return redirect(url_for("predict"))
    
    # GET request - show the form
    # Get unique values from encoders for dropdowns
    airlines = list(label_encoders['Airline'].classes_)
    sources = list(label_encoders['Source'].classes_)
    destinations = list(label_encoders['Destination'].classes_)
    
    return render_template(
        "predict.html",
        airlines=airlines,
        sources=sources,
        destinations=destinations
    )


# =============================================================================
# ROUTE: Logout
# =============================================================================
@app.route("/logout")
def logout():
    """
    Logout route.
    Clears session and redirects to login page.
    """
    session.pop("user", None)
    flash("👋 You have been logged out!", "info")
    return redirect(url_for("login"))


# =============================================================================
# ROUTE: About Page
# =============================================================================
@app.route("/about")
def about():
    """
    About page route.
    Displays project information.
    """
    if "user" not in session:
        flash("⚠️ Please login first!", "warning")
        return redirect(url_for("login"))
    
    return render_template("about.html")


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FLIGHT DELAY PREDICTION SYSTEM")
    print("=" * 60)
    print("Starting Flask Server...")
    print("Access the application at: http://127.0.0.1:5000")
    print("🔑 Default Login:")
    print("   Username: admin")
    print("   Password: admin123")
    print("=" * 60 + "\n")
    
    # Run the Flask app
    app.run(debug=True)
