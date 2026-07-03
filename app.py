import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load trained model
model = pickle.load(open('credit_card_model.pkl', 'rb'))

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction page
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    features = [float(x) for x in request.form.values()]

    # Convert input into array
    final_input = [np.array(features)]

    # Predict result
    prediction = model.predict(final_input)

    # Display output
    if prediction[0] == 1:
        result = "Credit Card Approved"
    else:
        result = "Credit Card Rejected"

    return render_template('result.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
