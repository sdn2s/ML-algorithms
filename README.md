# ML-algorithms
Basic Machine Learning Algorithms
## KNN
This project demonstrates how to use the K-Nearest Neighbors (KNN) algorithm to classify wines based on their chemical properties using the Wine Dataset from the scikit-learn library. The dataset classifies wines into three different categories.

### Features
Load and preprocess the Wine dataset.
Build a K-Nearest Neighbors (KNN) classifier.
Train the model on the training dataset.
Evaluate the model's accuracy on the test dataset.
Predict the wine category based on new chemical property inputs.
Requirements
Before running the application, ensure that you have the following libraries installed:

 - Python 3.x
 - numpy
 - pandas
 - scikit-learn
You can install the required dependencies by running:

```bash
pip install numpy pandas scikit-learn
Usage
To run the script and train the model, simply execute the following command in your terminal:
```
```bash
python wine_classification.py
Example Output
After training and evaluating the model, you will see outputs like:
```

Точность модели: 75.93%
Прогноз для новых данных: class_1
Model Architecture
The classifier used in this project is the K-Nearest Neighbors (KNN) model with 5 neighbors (n_neighbors=5), which classifies the input data based on the 5 nearest neighbors in the training dataset.

Dataset
The Wine Dataset is a well-known dataset that consists of 178 samples of wine, each described by 13 features (chemical properties), including:

Alcohol
Malic acid
Ash
Magnesium
Flavanoids
and more...
Each sample is categorized into one of three possible classes.

## Linear-regression (Housing Price Prediction)
This project demonstrates how to use Linear Regression to predict house prices based on various features using the Boston Housing Dataset from scikit-learn. The dataset contains features like crime rate, average number of rooms per dwelling, and proximity to employment centers, and the goal is to predict the median value of owner-occupied homes.

### Features
Load and preprocess the Boston Housing dataset.
Build a linear regression model.
Train the model on the training dataset.
Evaluate the model's accuracy using the Mean Squared Error (MSE) metric.
Predict the house price for new input data.
Requirements
Before running the application, ensure that you have the following libraries installed:

 - Python 3.x
 - numpy
 - pandas
 - scikit-learn
You can install the required dependencies by running:

```bash
pip install numpy pandas scikit-learn
Usage
To run the script and train the model, simply execute the following command in your terminal:
```
```bash
python boston_regression.py
Example Output
After training and evaluating the model, you will see outputs like:
```
```scss
Среднеквадратичная ошибка (MSE): 21.52
Предсказанная цена для новых данных: 25.57
Model Architecture
The model used in this project is the Linear Regression model, which attempts to fit a linear relationship between the input features and the target variable (house price).
```

Dataset
The Boston Housing Dataset contains 506 samples, each with 13 features related to the housing market, such as:

CRIM: Per capita crime rate by town
RM: Average number of rooms per dwelling
TAX: Property tax rate per $10,000
PTRATIO: Pupil-teacher ratio by town

Each sample is associated with the target variable PRICE, which represents the median value of owner-occupied homes in $1000's.