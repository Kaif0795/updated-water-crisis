from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import sqlite3
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Database setup
db_path = "orders.db"
if not os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE orders (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, phone TEXT, address TEXT, amount INTEGER)''')
    conn.commit()
    conn.close()

# Sample dataset with major Indian cities
data = {
    'City': ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow'],
    'Rainfall': [800, 900, 750, 650, 700, 850, 950, 620, 670, 730],
    'Temperature': [30, 32, 31, 33, 34, 29, 28, 35, 36, 30],
    'Population_Growth': [1.2, 1.5, 1.3, 1.7, 2.0, 1.1, 1.0, 2.2, 2.5, 1.4],
    'Water_Availability': [500, 550, 480, 450, 460, 520, 580, 430, 440, 490]
}

df = pd.DataFrame(data)

# Train Model
model_path = "model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    X = df[['Rainfall', 'Temperature', 'Population_Growth']]
    y = df['Water_Availability']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction.html')
def prediction():
    return render_template('prediction.html')

@app.route('/sample_data.html')
def sample_data():
    return render_template('sample_data.html')

@app.route('/water_tanker.html')
def water_tanker():
    return render_template('water_tanker.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        city = request.json['city'].strip()
        city_data = df[df['City'].str.lower() == city.lower()]
        
        if city_data.empty:
            return jsonify({"response": "City not found in dataset. Please enter a valid city name."}), 400
        
        input_data = city_data[['Rainfall', 'Temperature', 'Population_Growth']].values
        predicted = model.predict(input_data)
        return jsonify({"response": f"Predicted Water Availability in {city}: {predicted[0]:.2f} liters"})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 500

@app.route('/get_data')
def get_data():
    return jsonify(df.to_dict(orient='records'))

@app.route('/order', methods=['POST'])
def order():
    try:
        name = request.form['name']
        phone = request.form['phone']
        address = request.form['address']
        amount = int(request.form['amount'])
        
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("INSERT INTO orders (name, phone, address, amount) VALUES (?, ?, ?, ?)", (name, phone, address, amount))
        conn.commit()
        conn.close()
        
        return jsonify({"response": "Order placed successfully!"}), 200
    except ValueError:
        return jsonify({"response": "Invalid amount. Please enter a valid number."}), 400
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
