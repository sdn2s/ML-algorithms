import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Загрузка данных о винах
wine = load_wine()

# Преобразуем данные в DataFrame для удобства
data = pd.DataFrame(data=wine.data, columns=wine.feature_names)
data['target'] = wine.target

# Шаг 2: Разделение данных на тренировочную и тестовую выборки
X = data.drop('target', axis=1)  # Признаки (химические свойства)
y = data['target']  # Целевая переменная (классы вина)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Создание модели K-ближайших соседей и обучение
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Оценка точности модели на тестовой выборке
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy * 100:.2f}%")

# Прогнозирование для новых данных
new_data = np.array([[13.2, 2.77, 2.51, 18.9, 96.0, 2.61, 2.78, 0.29, 1.59, 5.05, 1.43, 3.17, 1510]])  # Пример новых данных
prediction = model.predict(new_data)
wine_class = wine.target_names[prediction]
print(f"Прогноз для новых данных: {wine_class[0]}")