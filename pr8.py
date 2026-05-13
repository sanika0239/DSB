# =========================================================
# Titanic Dataset Analysis using Seaborn
# =========================================================

import seaborn as sns
import matplotlib.pyplot as plt

# =========================================================
# Load Titanic Dataset
# =========================================================

titanic = sns.load_dataset('titanic')

# Display first 5 rows
print(titanic.head())

# Dataset Information
print("\nDataset Shape:")
print(titanic.shape)

print("\nDataset Info:")
print(titanic.info())

# =========================================================
# Finding Patterns in Titanic Dataset
# =========================================================

# 1. Survival Count
plt.figure(figsize=(6,4))
sns.countplot(x='survived', data=titanic)

plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")

plt.show()

# =========================================================
# 2. Survival based on Gender
# =========================================================

plt.figure(figsize=(6,4))
sns.countplot(x='sex', hue='survived', data=titanic)

plt.title("Survival Based on Gender")
plt.xlabel("Gender")
plt.ylabel("Count")

plt.show()

# =========================================================
# 3. Survival based on Passenger Class
# =========================================================

plt.figure(figsize=(6,4))
sns.countplot(x='pclass', hue='survived', data=titanic)

plt.title("Survival Based on Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Count")

plt.show()

# =========================================================
# 4. Age Distribution
# =========================================================

plt.figure(figsize=(8,5))
sns.histplot(titanic['age'].dropna(), bins=30, kde=True)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.show()

# =========================================================
# 5. Fare Distribution (Required Question)
# =========================================================

plt.figure(figsize=(8,5))

sns.histplot(titanic['fare'], bins=30, kde=True)

plt.title("Fare Distribution of Titanic Passengers")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")

plt.show()