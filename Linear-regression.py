import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Загрузка набора данных о домах в Бостоне
boston = load_boston()

# Преобразуем данные в DataFrame для удобства
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['PRICE'] = boston.target

# Разделение данных на тренировочную и тестовую выборки
X = data.drop('PRICE', axis=1)  # Признаки (характеристики домов)
y = data['PRICE']  # Целевая переменная (цена домов)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Создание и обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Оценка модели на тестовой выборке
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")

# Прогнозирование для новых данных
new_data = np.array([[0.00632, 18.0, 2.31, 0.0, 0.538, 6.575, 65.2, 4.0900, 1.0, 296.0, 15.3, 396.9, 4.98]])  # Пример новых данных
predicted_price = model.predict(new_data)
print(f"Предсказанная цена для новых данных: {predicted_price[0]:.2f}")