import pandas as pd

# Load the Iris dataset
df = pd.read_csv("Iris.csv")

# Display the first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Display dataset information
print("\nDataset information:")
print(df.info())

# Display dataset shape
print("\nDataset shape:")
print(df.shape)

# Display column names
print("\nColumn names:")
print(df.columns)

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())

# Display statistical summary
print("\nStatistical summary:")
print(df.describe())

# Display species distribution
print("\nSpecies distribution:")
print(df["Species"].value_counts())

import matplotlib.pyplot as plt
import seaborn as sns

# Graph 1: Species distribution
plt.figure(figsize=(8, 5))
sns.countplot(x="Species", data=df)
plt.title("Distribution of Iris Species")
plt.xlabel("Species")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("species_distribution.png")


# Graph 2: Sepal length vs Sepal width
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x="SepalLengthCm",
    y="SepalWidthCm",
    hue="Species",
    data=df
)
plt.title("Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.tight_layout()
plt.savefig("sepal_scatterplot.png")

# Show both graphs
# plt.show()

# Separate features (X) and target (y)

X = df[
    [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
]

y = df["Species"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Training labels:", y_train.shape[0])
print("Testing labels:", y_test.shape[0])

from sklearn.ensemble import RandomForestClassifier

# Create the machine learning model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

print("\nModel training completed successfully!")

from sklearn.metrics import accuracy_score

# Predict species for test data
y_pred = model.predict(X_test)

# Calculate model accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

from sklearn.metrics import confusion_matrix, classification_report

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Predict the species of a new flower
new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=[
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
)

prediction = model.predict(new_flower)

print("\nNew Flower Prediction:", prediction[0])