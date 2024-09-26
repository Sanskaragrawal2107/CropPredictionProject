from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the model and label encoders
model = joblib.load('crop_model.pkl')
state_label_encoder = joblib.load('state_label_encoder.pkl')
crop_label_encoder = joblib.load('crop_label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form and convert to float
        int_features = [float(request.form.get(feature)) for feature in ['N_SOIL', 'P_SOIL', 'K_SOIL', 'TEMPERATURE', 'HUMIDITY', 'ph', 'RAINFALL', 'CROP_PRICE']]
    except ValueError as e:
        return render_template('index.html', prediction_text=f'Error: Please enter valid numerical values. {str(e)}')

    # Get the STATE input
    STATE = request.form.get('STATE')

    # Encode 'STATE'
    try:
        STATE_encoded = state_label_encoder.transform([STATE])
    except ValueError:
        return render_template('index.html', prediction_text='Error: Invalid state name. Please ensure it is correct.')

    # Combine all features
    final_features = np.array([[*int_features, STATE_encoded[0]]])
    
    # Make the prediction
    prediction = model.predict(final_features)
    
    # Decode the predicted crop
    predicted_crop = crop_label_encoder.inverse_transform(prediction)

    return render_template('index.html', prediction_text=f'The predicted crop is: {predicted_crop[0]}')

@app.route('/allowed_states', methods=['GET'])
def allowed_states():
    states = state_label_encoder.classes_.tolist()  # Get the list of allowed states
    return jsonify(states)

if __name__ == "__main__":
    app.run(debug=True)
