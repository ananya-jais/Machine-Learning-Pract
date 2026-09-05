# Practical 3 – Mushroom Edibility Classification Using Naive Bayes

## Aim

To classify mushrooms as edible or poisonous using the Naive Bayes Machine Learning algorithm.

## Dataset

The dataset contains different characteristics of mushrooms such as:

- Cap Shape
- Cap Color
- Odor
- Gill Spacing
- Gill Color
- Stalk Shape
- Ring Number
- Habitat
- Population

The target variable is `Class`, which represents whether the mushroom is edible or poisonous.

## Algorithm Used

### Naive Bayes

Naive Bayes is a supervised classification algorithm based on Bayes' Theorem. It assumes that the features are conditionally independent of each other.

A Gaussian Naive Bayes classifier was used for this practical.

## Steps

1. Load the mushroom dataset.
2. Remove `SampleID` as it is only an identifier.
3. Encode categorical features into numerical values.
4. Separate features and target variable.
5. Split the dataset into training and testing sets.
6. Train the Gaussian Naive Bayes model.
7. Predict mushroom edibility.
8. Evaluate the model using accuracy, confusion matrix and classification report.

## Technologies Used

- Python
- Pandas
- Scikit-learn

## Files

| File | Description |
|---|---|
| `pract3_code.py` | Python implementation of Naive Bayes |
| `dataset.csv` | Mushroom dataset |
| `output.log` | Program output |

## Result

The Naive Bayes model successfully classified mushrooms into edible and poisonous classes. Its performance was evaluated using accuracy, confusion matrix and classification report.

## Conclusion

Naive Bayes was successfully implemented for mushroom edibility classification. The model was able to use mushroom characteristics to predict whether a mushroom belongs to the edible or poisonous class.