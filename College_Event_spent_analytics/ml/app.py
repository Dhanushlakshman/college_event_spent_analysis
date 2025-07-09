import streamlit as st
import pandas as pd
import mysql.connector
from mysql.connector.pooling import MySQLConnectionPool
from mysql.connector import errorcode
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Database connection pool configuration
POOL_CONFIG = {
    "pool_name": "mypool",
    "pool_size": 5,
    "host": "localhost",
    "port": 3306,
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "Dhanush@12"),
    "database": "college_cost_analysis"
}


# Cache model loading
@st.cache_resource
def load_model():
    try:
        model = joblib.load("ml/saved_model/model_rf.pkl")
        scaler = joblib.load("ml/saved_model/scaler.pkl")
        # Get feature names from scaler if available
        X_columns = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else None
        return model, scaler, X_columns
    except FileNotFoundError:
        return None, None, None

def get_db_connection():
    try:
        db_pool = MySQLConnectionPool(**POOL_CONFIG)
        return db_pool.get_connection()
    except mysql.connector.Error as e:
        if e.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            raise Exception("Database access denied: Invalid credentials")
        elif e.errno == errorcode.ER_BAD_DB_ERROR:
            raise Exception("Database does not exist")
        else:
            raise Exception(f"Failed to connect to MySQL: {str(e)}")

def train_model():
    try:
        conn = get_db_connection()
        query = """
            SELECT dept, donation, events, food_spend, guest_amt, venue_cost,
                   equipment_rental, attendance, sponsorship, marketing_spend, student_spend
            FROM event_spending
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return None, None, None, "No data found in database"
        
        # Preprocess
        df = pd.get_dummies(df, columns=['dept'], drop_first=True)
        X = df.drop(columns=['student_spend'])
        y = df['student_spend']
        
        X_columns = X.columns.tolist()  # Save feature names
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train model with cross-validation
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        # Save model and scaler
        os.makedirs("ml/saved_model", exist_ok=True)
        joblib.dump(model, "ml/saved_model/model_rf.pkl")
        joblib.dump(scaler, "ml/saved_model/scaler.pkl")
        
        return model, scaler, X_columns, (f"Model trained with R²: {model.score(scaler.transform(X_test), y_test):.2f}\n"
                                        f"Cross-validation scores: {cv_scores.mean():.2f} (±{cv_scores.std() * 2:.2f})")
    except Exception as e:
        return None, None, None, f"Error training model: {str(e)}"

def predict_spending(inputs, model, scaler):
    try:
        # Input validation
        for key, value in inputs.items():
            if key != 'dept' and value < 0:
                raise ValueError(f"{key} cannot be negative")
        
        input_df = pd.DataFrame([inputs])
        input_df = pd.get_dummies(input_df, columns=['dept'], drop_first=True)
        
        # Align columns with training data
        model_columns = scaler.feature_names_in_
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model_columns]
        
        input_scaled = scaler.transform(input_df)
        return model.predict(input_scaled)[0]
    except Exception as e:
        return f"Error predicting: {str(e)}"

def plot_feature_importance(model, X_columns):
    if len(model.feature_importances_) != len(X_columns):
        raise ValueError(f"Feature importances ({len(model.feature_importances_)}) and X_columns ({len(X_columns)}) have different lengths")
    importance = pd.DataFrame({
        'feature': X_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance['feature'], importance['importance'])
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    plt.tight_layout()
    return fig

def main():
    st.title("College Event Spending Prediction")
    
    # Train or load model
    model, scaler, X_columns, train_status = train_model()
    if model is None or scaler is None:
        model, scaler, X_columns = load_model()
    
    st.write(train_status)
    
    if model is None or scaler is None or X_columns is None:
        st.error("Failed to load or train model. Please check the database connection and data availability.")
        return
    
    # Display feature importance
    st.subheader("Feature Importance")
    st.pyplot(plot_feature_importance(model, X_columns))
    
    # Input form
    with st.form("prediction_form"):
        st.subheader("Enter Event Details")
        dept = st.selectbox("Department", ["Civil", "Mechanical", "ECE", "EEE", "CSE", 
                                         "IT", "AI&DS", "Medical Electronics", "Cybersecurity"])
        donation = st.number_input("Donation Amount ($)", min_value=0.0, value=100000.0, step=1000.0)
        events = st.number_input("Number of Events", min_value=1, value=10, step=1)
        food_spend = st.number_input("Food Spend ($)", min_value=0.0, value=20000.0, step=1000.0)
        guest_amt = st.number_input("Guest Amount ($)", min_value=0.0, value=10000.0, step=1000.0)
        venue_cost = st.number_input("Venue Cost ($)", min_value=0.0, value=50000.0, step=1000.0)
        equipment_rental = st.number_input("Equipment Rental ($)", min_value=0.0, value=10000.0, step=1000.0)
        attendance = st.number_input("Attendance", min_value=0, value=300, step=10)
        sponsorship = st.number_input("Sponsorship Amount ($)", min_value=0.0, value=50000.0, step=1000.0)
        marketing_spend = st.number_input("Marketing Spend ($)", min_value=0.0, value=10000.0, step=1000.0)
        
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            inputs = {
                'dept': dept, 'donation': donation, 'events': events, 'food_spend': food_spend,
                'guest_amt': guest_amt, 'venue_cost': venue_cost, 'equipment_rental': equipment_rental,
                'attendance': attendance, 'sponsorship': sponsorship, 'marketing_spend': marketing_spend
            }
            prediction = predict_spending(inputs, model, scaler)
            if isinstance(prediction, str):
                st.error(prediction)
            else:
                st.success(f"Predicted Student Spending: ${prediction:,.2f}")

if __name__ == "__main__":
    main()