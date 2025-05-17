import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load CSV file (use the custom dataset)
data = pd.read_csv("Admission_Predict_Custom.csv")

# Preprocess the data
df = data.copy()
df['Admission_Chance'] = [1 if chance > 0.85 else 0 for chance in df['Admission Chance']]
X = df[['GRE Score', 'TOEFL Score', 'CGPA']]
y = df['Admission_Chance']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialize and train the model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Save the model and scaler
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(sc, open("scaler.pkl", "wb"))