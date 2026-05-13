import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Academic_Performance_Dataset.csv")

print("\nFirst 5 Records:\n")
print(df.head())

print("\nMissing Values in Each Column:\n")
print(df.isnull().sum())


# 1. Gender (Categorical → Mode Imputation)
df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)

# 2. Age (Numeric → Median Imputation)
df["Age"].fillna(df["Age"].median(), inplace=True)

# 3. Scores & Study Hours (Numeric → Mean Imputation)
numeric_cols = ["Math_Score","Science_Score",
                "English_Score","Attendance_Percentage",
                "Study_Hours"]


for col in numeric_cols:
    df[col].fillna(df[col].mean(), inplace=True)

print("\nMissing Values After Cleaning:\n")
print(df.isnull().sum())

df["Gender"] = df["Gender"].replace({"M": "Male", "F": "Female"})


df.loc[df["Attendance_Percentage"] > 100, "Attendance_Percentage"] = 100
df.loc[df["Attendance_Percentage"] < 0, "Attendance_Percentage"] = 0

df.loc[df["Age"] <= 0, "Age"] = df["Age"].median()

print("\nData cleaned for inconsistencies.")





def treat_outliers_iqr(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    return np.where(column > upper, upper,
           np.where(column < lower, lower, column))

for col in numeric_cols:
    df[col] = treat_outliers_iqr(df[col])

print("\nOutliers treated using IQR Capping Method.")

plt.figure(figsize=(10,6))
sns.boxplot(data=df[numeric_cols])
plt.title("Boxplot After Outlier Treatment")
plt.show()



# Check skewness before transformation
print("\nSkewness Before Transformation:")
print(df["Study_Hours"].skew())



# Apply Log Transformation to reduce skewness
df["Study_Hours_Log"] = np.log1p(df["Study_Hours"])

print("\nSkewness After Log Transformation:")
print(df["Study_Hours_Log"].skew())


# Compare distributions
fig, ax = plt.subplots(1,2, figsize=(12,5))

sns.histplot(df["Study_Hours"], kde=True, ax=ax[0])
ax[0].set_title("Before Log Transformation")

sns.histplot(df["Study_Hours_Log"], kde=True, ax=ax[1])
ax[1].set_title("After Log Transformation")

plt.show()

print(df.head())