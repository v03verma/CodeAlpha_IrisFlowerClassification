# Iris Flower Classification

## Project Overview
This project uses Machine Learning to classify Iris flowers into three species based on their sepal and petal measurements.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Dataset
The dataset contains 150 flower records and four measurement features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Target classes:
- Iris-setosa
- Iris-versicolor
- Iris-virginica

## Project Workflow
1. Load and explore the dataset.
2. Check missing values and dataset statistics.
3. Visualize species distribution and feature relationships.
4. Split the dataset into training and testing sets.
5. Train a Random Forest Classifier.
6. Evaluate the model using accuracy, confusion matrix, and classification report.
7. Predict the species of a new flower.

## Model Performance
- Model: Random Forest Classifier
- Train-test split: 80:20
- Test accuracy: 90%

## Project Files
- `iris_classification.py` — Main Python code
- `Iris.csv` — Dataset
- `species_distribution.png` — Species distribution graph
- `sepal_scatterplot.png` — Sepal measurement scatterplot

## How to Run
1. Install Python.
2. Install the required libraries:

   `pip install pandas numpy matplotlib seaborn scikit-learn`

3. Keep `Iris.csv` in the same folder as the Python script.
4. Run the project:

   `python iris_classification.py`

## Author
Vinit