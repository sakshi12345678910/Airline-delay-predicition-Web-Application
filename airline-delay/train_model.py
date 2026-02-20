# Import required libraries
import pandas as pd                          # For data manipulation
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import LabelEncoder         # For encoding categorical variables
from sklearn.ensemble import RandomForestClassifier   # ML Algorithm
from sklearn.metrics import accuracy_score            # For model evaluation
import pickle                                         # For saving model
import joblib                                         # For saving label encoders

# Step 1: Load the dataset
print("=" * 60)
print("STEP 1: LOADING DATASET")
print("=" * 60)
df = pd.read_csv("data/flights.csv")
print(f"Dataset loaded successfully!")
print(f"Total records: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Step 2: Display basic information about dataset
print("\n" + "=" * 60)
print("STEP 2: DATASET OVERVIEW")
print("=" * 60)
print("\n📌 Dataset Columns Explanation:")
print("   - Airline: Name of the airline (IndiGo, Air India, etc.)")
print("   - Source: Departure airport city")
print("   - Destination: Arrival airport city")
print("   - DepartureHour: Hour of departure (0-23)")
print("   - DayOfWeek: Day of week (1=Monday to 7=Sunday)")
print("   - Distance: Flight distance in kilometers")
print("   - Delayed: Target variable (0=On Time, 1=Delayed)")

print("\nFirst 5 rows:")
print(df.head())

print("\n📈 Data Types:")
print(df.dtypes)

print("\n📉 Missing Values:")
print(df.isnull().sum())

# Step 3: Data Cleaning
print("\n" + "=" * 60)
print("STEP 3: DATA CLEANING")
print("=" * 60)

# Remove any duplicate rows
df_clean = df.drop_duplicates()
print(f"Duplicates removed: {len(df) - len(df_clean)} rows")

# Handle missing values (if any)
if df_clean.isnull().sum().sum() > 0:
    df_clean = df_clean.dropna()
    print("Missing values handled")
else:
    print("No missing values found")

# Step 4: Feature Engineering & Encoding
print("\n" + "=" * 60)
print("STEP 4: FEATURE ENCODING")
print("=" * 60)

# Store label encoders for later use during prediction
label_encoders = {}

# Encode categorical columns
categorical_columns = ['Airline', 'Source', 'Destination']

for col in categorical_columns:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    label_encoders[col] = le
    print(f"Encoded '{col}': {list(le.classes_)}")

# Step 5: Prepare features and target
print("\n" + "=" * 60)
print("STEP 5: PREPARING FEATURES AND TARGET")
print("=" * 60)

# Features (X) - all columns except 'Delayed'
X = df_clean.drop('Delayed', axis=1)

# Target (y) - only 'Delayed' column
y = df_clean['Delayed']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Features: {list(X.columns)}")
print(f"Target distribution:")
print(f"   - On Time (0): {(y == 0).sum()}")
print(f"   - Delayed (1): {(y == 1).sum()}")

# Step 6: Split data into training and testing sets
print("\n" + "=" * 60)
print("STEP 6: TRAIN-TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% data for testing
    random_state=42,    # For reproducibility
    stratify=y          # Maintain class distribution
)

print(f"Data split successfully!")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Step 7: Train the Machine Learning Model
print("\n" + "=" * 60)
print("STEP 7: MODEL TRAINING")
print("=" * 60)

# Using Random Forest Classifier
# Random Forest is good for classification problems with mixed features
model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Maximum depth of trees
    random_state=42,       # For reproducibility
    n_jobs=-1              # Use all CPU cores
)

print("Training Random Forest model...")
model.fit(X_train, y_train)
print("Model training completed!")

# Step 8: Model Evaluation
print("\n" + "=" * 60)
print("STEP 8: MODEL EVALUATION")
print("=" * 60)

# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Feature importance
print("\n📈 Feature Importance:")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print(feature_importance.to_string(index=False))

# Step 9: Save the trained model and label encoders
print("\n" + "=" * 60)
print("STEP 9: SAVING MODEL")
print("=" * 60)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
print("Model saved as 'model.pkl'")

# Save label encoders for web prediction
joblib.dump(label_encoders, "label_encoders.joblib")
print("Label encoders saved as 'label_encoders.joblib'")

# Save feature order
feature_columns = list(X.columns)
joblib.dump(feature_columns, "feature_columns.joblib")
print("Feature columns saved as 'feature_columns.joblib'")

print("\n" + "=" * 60)
print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\n📝 NEXT STEPS:")
print("   1. Run 'python app.py' to start the Flask web server")
print("   2. Open browser and go to 'http://127.0.0.1:5000'")
print("   3. Login with username: admin, password: admin123")
print("   4. Enter flight details to get delay prediction")
print("=" * 60)
