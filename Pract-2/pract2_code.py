import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

df = pd.read_csv("Pract-2/dataset.csv")

print("DATASET")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

# --------------------------------------------------
# 2. Encode Data & Separate Features/Target
# --------------------------------------------------
# Drop SampleID first since it's just an ID
df = df.drop("SampleID", axis=1)

# Initialize the encoder
encoder = LabelEncoder()

# Translate ALL text columns into numerical codes (0, 1, 2...)
for column in df.columns:
    df[column] = encoder.fit_transform(df[column].astype(str))

# Now separate the translated data into Features (X) and Target (y)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]


# --------------------------------------------------
# 3. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTRAINING DATA:")
print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# --------------------------------------------------
# 4. Decision Tree using Entropy
# --------------------------------------------------

model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

model.fit(X_train, y_train)


# --------------------------------------------------
# 5. Prediction
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 6. Evaluation
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("-----------------")

print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --------------------------------------------------
# 7. Display Decision Tree
# --------------------------------------------------

plt.figure(figsize=(16, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=[str(c) for c in model.classes_],
    filled=True,
    rounded=True
)

plt.title("Decision Tree Classification Using Entropy")

plt.savefig(
    "decision_tree.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()