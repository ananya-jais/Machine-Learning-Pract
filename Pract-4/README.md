# Practical 4 – K-Nearest Neighbour (KNN) Classification

## Aim

To implement K-Nearest Neighbour (KNN) classification and classify a new point using different values of K.

## Given Data

Six training points P1 to P6 are given with their coordinates and respective classes A or B.

The new point to be classified is:

**Q = (4, 4)**

## Algorithm Used

### K-Nearest Neighbour (KNN)

KNN is a supervised Machine Learning classification algorithm that classifies a new data point based on the classes of its nearest neighbours.

Euclidean distance is used to calculate the distance between the points.

## Values of K

The point Q = (4, 4) is classified using:

- K = 1
- K = 3
- K = 5

## Results

| K Value | Predicted Class |
|---|---|
| K = 1 | B |
| K = 3 | A |
| K = 5 | A |

## Observation

For **K = 1**, only the nearest point is considered, so the predicted class is **B**.

For **K = 3** and **K = 5**, more neighbouring points are considered. Class **A** becomes the majority class, resulting in an **A** prediction.

This demonstrates that the choice of K can affect the classification result.

## Technologies Used

- Python
- NumPy
- Scikit-learn

## Files

| File | Description |
|---|---|
| `code.py` | Python implementation of KNN classification |
| `output.log` | Program output |

## Conclusion

K-Nearest Neighbour classification was successfully implemented for K = 1, 3 and 5. The experiment shows that changing the value of K can affect the predicted class of a data point.