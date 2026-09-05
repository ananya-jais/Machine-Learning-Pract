import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

df = pd.read_csv("dataset.csv")

print("MUSHROOM EDIBILITY DATASET")
print("--------------------------")

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# --------------------------------------------------
# 2. Check Class Distribution
# --------------------------------------------------

print("\nCLASS DISTRIBUTION:")
print(df["Class"].value_counts())


# --------------------------------------------------
# 3. Remove SampleID
# --------------------------------------------------

# SampleID is only an identifier and is not useful
# for predicting mushroom edibility.

df = df.drop("SampleID", axis=1)


# --------------------------------------------------
# 4. Encode Categorical Data
# --------------------------------------------------

encoder = LabelEncoder()

for column in df.columns:
    df[column] = encoder.fit_transform(df[column].astype(str))


# --------------------------------------------------
# 5. Separate Features and Target
# --------------------------------------------------

X = df.drop("Class", axis=1)
y = df["Class"]


print("\nFEATURES:")
print(X.columns.tolist())

print("\nTARGET:")
print("Class")


# --------------------------------------------------
# 6. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTRAINING AND TESTING")
print("--------------------")

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# --------------------------------------------------
# 7. Naive Bayes Model
# --------------------------------------------------

model = GaussianNB()

model.fit(X_train, y_train)


# --------------------------------------------------
# 8. Prediction
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 9. Evaluation
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("-----------------")

print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Edible", "Poisonous"]
))


# --------------------------------------------------
# 10. Sample Predictions
# --------------------------------------------------

print("\nSAMPLE PREDICTIONS")
print("------------------")

for i in range(min(10, len(y_test))):

    actual = "Poisonous" if y_test.iloc[i] == 1 else "Edible"
    predicted = "Poisonous" if y_pred[i] == 1 else "Edible"

    print(
        f"Sample {i + 1}: "
        f"Actual = {actual}, "
        f"Predicted = {predicted}"
    )