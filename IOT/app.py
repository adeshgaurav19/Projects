# In your app.py file on your computer

from flask import Flask, request, jsonify
import sqlite3 # Using SQLite for simplicity
from datetime import datetime

app = Flask(__name__)

# --- Database Setup ---
def get_db_connection():
    # Connects to the database file, creating it if it doesn't exist
    conn = sqlite3.connect('iot_database.db')
    conn.row_factory = sqlite3.Row
    return conn

# A function to create the table if it's not already there
def setup_database():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reading_time TEXT NOT NULL,
            temperature REAL NOT NULL,
            humidity REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# --- The API Endpoint ---
@app.route('/submit_data', methods=['POST'])
def submit_data():
    """This function is the heart of your API."""
    try:
        # 1. Get the JSON data that the Pico sent
        data = request.get_json()
        print(f"Received data: {data}")

        # 2. Extract the values
        temp = data['temperature']
        hum = data['humidity']
        # Use the server's current time for this example
        timestamp = datetime.now().isoformat()

        # 3. Connect to the database and save the data
        conn = get_db_connection()
        conn.execute(
            "INSERT INTO sensor_readings (reading_time, temperature, humidity) VALUES (?, ?, ?)",
            (timestamp, temp, hum)
        )
        conn.commit()
        conn.close()

        # 4. Send a success response back to the Pico
        return jsonify({"message": "Data received successfully"}), 201

    except Exception as e:
        # If anything goes wrong, send an error response
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 400

# --- Main execution block ---
if __name__ == '__main__':
    setup_database()
    app.run(debug=True, host='0.0.0.0', port=5001)