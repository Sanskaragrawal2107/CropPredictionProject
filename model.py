import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib


df = pd.read_csv('crop_recommendation.csv')  # Replace with your actual dataset path


state_label_encoder = LabelEncoder()
crop_label_encoder = LabelEncoder()


df['STATE_ENCODED'] = state_label_encoder.fit_transform(df['STATE'])
df['CROP_ENCODED'] = crop_label_encoder.fit_transform(df['CROP'])


X = df[['N_SOIL', 'P_SOIL', 'K_SOIL', 'TEMPERATURE', 'HUMIDITY', 'ph', 'RAINFALL', 'STATE_ENCODED', 'CROP_PRICE']]
y = df['CROP_ENCODED']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier()
model.fit(X_train, y_train)


joblib.dump(model, 'crop_model.pkl')
joblib.dump(state_label_encoder, 'state_label_encoder.pkl')
joblib.dump(crop_label_encoder, 'crop_label_encoder.pkl')
