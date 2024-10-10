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