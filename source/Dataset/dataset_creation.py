import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()

# Define number of samples
n_samples = 100_000

# Possible values for categorical fields
gender_choices = ['Male', 'Female']
smoking_choices = ['Never', 'Former', 'Current']
diabetes_choices = ['No', 'Yes']
chest_pain_choices = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']

# Generate synthetic data
data = {
    "age": np.random.randint(29, 77, size=n_samples),  # normal age range for heart issues
    "gender": np.random.choice(gender_choices, size=n_samples),
    "smoking_status": np.random.choice(smoking_choices, size=n_samples),
    "systolic_bp": np.random.randint(90, 180, size=n_samples),
    "cholesterol": np.random.randint(150, 300, size=n_samples),
    "diabetes": np.random.choice(diabetes_choices, size=n_samples),
    "chest_pain_type": np.random.choice(chest_pain_choices, size=n_samples),
}

# Target variable: simple risk-based logic
def generate_target(row):
    risk = 0
    if row['age'] > 50:
        risk += 1
    if row['systolic_bp'] > 140:
        risk += 1
    if row['cholesterol'] > 240:
        risk += 1
    if row['diabetes'] == 'Yes':
        risk += 1
    if row['smoking_status'] == 'Current':
        risk += 1
    return 1 if risk >= 3 else 0

df = pd.DataFrame(data)
df["heart_disease"] = df.apply(generate_target, axis=1)

# Preview and save
print(df.head())
df.to_csv("synthetic_heart_disease.csv", index=False)