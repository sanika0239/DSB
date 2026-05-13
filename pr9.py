# =========================================================
# Data Visualization II - Titanic Dataset
# =========================================================

import seaborn as sns
import matplotlib.pyplot as plt

# =========================================================
# Load Titanic Dataset
# =========================================================

titanic = sns.load_dataset('titanic')

# Display first 5 rows
print(titanic.head())

# =========================================================
# Box Plot: Age Distribution with Gender and Survival
# =========================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    x='sex',
    y='age',
    hue='survived',
    data=titanic
)

plt.title("Age Distribution by Gender and Survival")
plt.xlabel("Gender")
plt.ylabel("Age")

plt.legend(title="Survived", labels=["No", "Yes"])

plt.show()