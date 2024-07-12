from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model
with open('model_implement.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from request
    data = request.json
    df = pd.DataFrame(data, index=[0])
    
    # Replace missing values with 0 (or any other value as required)
    df.replace([np.nan, np.inf, -np.inf], 0, inplace=True)
    
    # Select only numerical columns
    numerical_columns = [col for col in df.columns if df[col].dtype != 'object']
    df = df[numerical_columns]
    
    # Make prediction
    prediction = model.predict(df)
    
    # Return the result as a JSON response
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
