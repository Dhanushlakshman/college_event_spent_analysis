import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import errorcode
import os
import warnings

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Load environment variables
load_dotenv()

def get_db_connection(database=None):
    user = os.getenv("DB_USER", "root")
    password = os.getenv("DB_PASSWORD", "Dhanush@12")
    try:
        return mysql.connector.connect(
            host="localhost",
            port=3306,
            user=user,
            password=password,
            database=database if database else None
        )
    except mysql.connector.Error as e:
        if e.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            raise Exception("Database access denied: Invalid credentials")
        elif e.errno == errorcode.ER_BAD_DB_ERROR:
            raise Exception("Database does not exist")
        else:
            raise Exception(f"Failed to connect to MySQL: {str(e)}")

def create_database_and_table():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS college_cost_analysis")
        conn.commit()
        cursor.close()
        conn.close()

        conn = get_db_connection(database="college_cost_analysis")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS event_spending (
                id INT AUTO_INCREMENT PRIMARY KEY,
                dept VARCHAR(50),
                donation FLOAT,
                events INT,
                student_spend FLOAT,
                food_spend FLOAT,
                guest_amt FLOAT,
                venue_cost FLOAT,
                equipment_rental FLOAT,
                event_date DATE,
                attendance INT,
                sponsorship FLOAT,
                marketing_spend FLOAT
            )
        """)
        conn.commit()
        cursor.close()
        conn.close()
    except mysql.connector.Error as e:
        raise Exception(f"Error creating database/table: {str(e)}")

def validate_data(df):
    required_columns = [
        'dept', 'donation', 'events', 'student_spend', 'food_spend',
        'guest_amt', 'venue_cost', 'equipment_rental', 'event_date',
        'attendance', 'sponsorship', 'marketing_spend'
    ]
    
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Missing required columns: {missing}")
    
    numeric_cols = [col for col in required_columns if col not in ['dept', 'event_date']]
    for col in numeric_cols:
        if (df[col] < 0).any():
            raise ValueError(f"Negative values found in {col}")
    
    valid_depts = ["Civil", "Mechanical", "ECE", "EEE", "CSE", "IT", "AI&DS", 
                   "Medical Electronics", "Cybersecurity"]
    if not df['dept'].isin(valid_depts).all():
        raise ValueError("Invalid department values found")
    
    return df

def generate_synthetic_data(n=1000):
    np.random.seed(42)
    departments = ["Civil", "Mechanical", "ECE", "EEE", "CSE", "IT", "AI&DS", 
                   "Medical Electronics", "Cybersecurity"]
    
    data = {
        'dept': np.random.choice(departments, n),
        'donation': np.random.uniform(50000, 2000000, n),
        'events': np.random.randint(1, 50, n),
        'food_spend': np.random.uniform(10000, 100000, n),
        'guest_amt': np.random.uniform(5000, 50000, n),
        'venue_cost': np.random.uniform(20000, 150000, n),
        'equipment_rental': np.random.uniform(5000, 30000, n),
        'event_date': [(datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 730))).date() for _ in range(n)],
        'attendance': np.random.randint(100, 1000, n),
        'sponsorship': np.random.uniform(10000, 500000, n),
        'marketing_spend': np.random.uniform(5000, 50000, n)
    }
    
    df = pd.DataFrame(data)
    df['student_spend'] = (
        0.3 * df['donation'] +
        1000 * df['events'] +
        0.5 * df['food_spend'] +
        0.2 * df['guest_amt'] +
        0.4 * df['venue_cost'] +
        0.3 * df['equipment_rental'] +
        50 * df['attendance'] +
        0.2 * df['sponsorship'] +
        0.5 * df['marketing_spend'] +
        np.random.normal(0, 10000, n)
    ).clip(lower=10000)
    
    return validate_data(df)

def save_to_mysql(df, truncate=True):
    try:
        create_database_and_table()
        conn = get_db_connection(database="college_cost_analysis")
        cursor = conn.cursor()
        
        if truncate:
            cursor.execute("TRUNCATE TABLE event_spending")
        
        insert_query = """
            INSERT INTO event_spending (
                dept, donation, events, student_spend, food_spend, guest_amt,
                venue_cost, equipment_rental, event_date, attendance, sponsorship, marketing_spend
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        batch_size = 100
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            values = [tuple(row) for _, row in batch.iterrows()]
            cursor.executemany(insert_query, values)
        
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = f"synthetic_data_{timestamp}.csv"
        df.to_csv(fallback_path, index=False)
        raise Exception(f"Error saving to MySQL. Data saved to {fallback_path}. Reason: {str(e)}")

if __name__ == "__main__":
    try:
        df = generate_synthetic_data(1000)
        save_to_mysql(df, truncate=True)
        print("Data generation and storage completed successfully.")
        print(df.head())
    except Exception as e:
        print(f"Script execution failed: {str(e)}")
        raise
