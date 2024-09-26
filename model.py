import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load your dataset
df = pd.read_csv('crop_recommendation.csv')  # Replace with your actual dataset path

# Create label encoders
state_label_encoder = LabelEncoder()
crop_label_encoder = LabelEncoder()

# Fit encoders
df['STATE_ENCODED'] = state_label_encoder.fit_transform(df['STATE'])
df['CROP_ENCODED'] = crop_label_encoder.fit_transform(df['CROP'])

# Prepare features and target
X = df[['N_SOIL', 'P_SOIL', 'K_SOIL', 'TEMPERATURE', 'HUMIDITY', 'ph', 'RAINFALL', 'STATE_ENCODED', 'CROP_PRICE']]
y = df['CROP_ENCODED']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model and encoders
joblib.dump(model, 'crop_model.pkl')
joblib.dump(state_label_encoder, 'state_label_encoder.pkl')
joblib.dump(crop_label_encoder, 'crop_label_encoder.pkl')
