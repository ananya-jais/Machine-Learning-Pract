# Practical 2 – Decision Tree Classification Using Entropy

## Aim

To implement a Decision Tree Classification algorithm using Entropy as the criterion for splitting the data into different classes.

## Objective

To classify data into target classes using a Decision Tree and evaluate its performance using classification metrics.

## Algorithm Used

### Decision Tree Classifier

A Decision Tree is a supervised Machine Learning algorithm that makes decisions by splitting the dataset based on feature values.

### Entropy

Entropy measures the impurity or uncertainty in the data. The decision tree selects splits that reduce entropy and provide better classification.

The model was implemented using:

- `DecisionTreeClassifier`
- `criterion = "entropy"`

## Steps

1. Load the dataset.
2. Preprocess and encode categorical data.
3. Separate features and target variable.
4. Split the data into training and testing sets.
5. Train the Decision Tree using Entropy.
6. Predict the classes for test data.
7. Evaluate the model using accuracy, confusion matrix and classification report.
8. Visualize the decision tree.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib

## Files

| File | Description |
|---|---|
| `pract2_code.py` | Python implementation of Decision Tree Classification |
| `dataset.csv` | Dataset used for classification |
| `output.log` | Program output |
| `decision_tree.png` | Visualization of the generated decision tree |

## Result

The Decision Tree successfully classified the test data into the required classes. The performance of the model was evaluated using accuracy, confusion matrix and classification report.

## Conclusion

Decision Tree Classification using Entropy was successfully implemented. Entropy helped the model select suitable feature splits for classification, and the resulting decision tree was visualized using Matplotlib.