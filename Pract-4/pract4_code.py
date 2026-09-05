import numpy as np
from sklearn.neighbors import KNeighborsClassifier


# --------------------------------------------------
# 1. Training Data
# --------------------------------------------------

# Six training points
X = np.array([
    [4, 3],   # P1
    [3, 3],   # P2
    [5, 5],   # P3
    [2, 4],   # P4
    [8, 8],   # P5
    [7, 2]    # P6
])

# Classes of the six points
y = np.array([
    "B",  # P1
    "A",  # P2
    "A",  # P3
    "A",  # P4
    "B",  # P5
    "B"   # P6
])


# --------------------------------------------------
# 2. New Point
# --------------------------------------------------

Q = np.array([[4, 4]])


# --------------------------------------------------
# 3. Calculate Euclidean Distances
# --------------------------------------------------

print("K-NEAREST NEIGHBOUR CLASSIFICATION")
print("-----------------------------------")

print("\nTraining Points:")

points = ["P1", "P2", "P3", "P4", "P5", "P6"]

distances = []

for i in range(len(X)):
    distance = np.sqrt(np.sum((X[i] - Q[0]) ** 2))
    distances.append(distance)

    print(
        f"{points[i]} = {tuple(X[i])}, "
        f"Class = {y[i]}, "
        f"Distance = {distance:.4f}"
    )


# --------------------------------------------------
# 4. Sort Points According to Distance
# --------------------------------------------------

sorted_indices = np.argsort(distances)

print("\nPoints Sorted by Distance:")

for index in sorted_indices:
    print(
        f"{points[index]} -> "
        f"Distance = {distances[index]:.4f}, "
        f"Class = {y[index]}"
    )


# --------------------------------------------------
# 5. Classification for Different Values of K
# --------------------------------------------------

for k in [1, 3, 5]:

    model = KNeighborsClassifier(
        n_neighbors=k,
        metric="euclidean"
    )

    model.fit(X, y)

    prediction = model.predict(Q)[0]

    print(f"\nK = {k}")
    print(f"Predicted Class for Q = (4,4): {prediction}")

    # Display the K nearest neighbours
    nearest_indices = sorted_indices[:k]

    print("Nearest Neighbours:")

    for index in nearest_indices:
        print(
            f"  {points[index]} "
            f"(Class {y[index]}, "
            f"Distance = {distances[index]:.4f})"
        )


# --------------------------------------------------
# 6. Final Observation
# --------------------------------------------------

print("\nOBSERVATION")
print("-----------")
print("For K = 1, the predicted class is B.")
print("For K = 3, the predicted class is A.")
print("For K = 5, the predicted class is A.")

print("\nAs K increases, more neighbouring points")
print("participate in the classification.")