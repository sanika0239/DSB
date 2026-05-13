import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("Titanic-Dataset.csv")
print("Dataset Loaded Successfully!\n")

print("First 5 Rows of Dataset:\n")
print(df.head())

print("\n--- Checking Missing Values ---")
print(df.isnull().sum())

print("\n--- Dataset Dimensions ---")
print("Shape of Dataset:", df.shape)

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Dataset Info ---")
print(df.info())


print("\n--- Data Types Before Conversion ---")
print(df.dtypes)

df['Sex'] = df['Sex'].astype('category')
df['Embarked'] = df['Embarked'].astype('category')
df['Pclass'] = df['Pclass'].astype('category')

print("\n--- Data Types After Conversion ---")
print(df.dtypes)

print("\n--- before Encoding Categorical Variables ---")
print(df.head())

label_encoder = LabelEncoder()

df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])
df['Pclass'] = df['Pclass'].cat.codes

print("\n--- After Encoding Categorical Variables ---")
print(df.head())

scaler = StandardScaler()
numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']

df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\n--- After Normalization ---")
print(df.head())

print("\nProgram Executed Successfully!")