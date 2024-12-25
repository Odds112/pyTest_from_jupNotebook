import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pytest

np.random.seed(42)


xs1 = np.linspace(0, 10, 100)
ys1 = xs1 + np.random.random(100) * 2 - 1
with open("dataset1.csv", "w") as f:
    f.write("x,y\n")
    for x, y in zip(xs1, ys1):
        f.write(f"{x},{y}\n")


xs2 = np.linspace(0, 10, 100)
ys2 = 2 * xs2 + np.random.random(100) * 3 - 1.5
with open("dataset2.csv", "w") as f:
    f.write("x,y\n")
    for x, y in zip(xs2, ys2):
        f.write(f"{x},{y}\n")


xs3 = np.linspace(0, 10, 100)
ys3 = -xs3 + np.random.random(100) * 1.5 - 0.75
with open("dataset3.csv", "w") as f:
    f.write("x,y\n")
    for x, y in zip(xs3, ys3):
        f.write(f"{x},{y}\n")


model = LinearRegression()


train_data = pd.read_csv("dataset1.csv")
X_train = train_data["x"].values.reshape(-1, 1)
y_train = train_data["y"].values

model.fit(X_train, y_train)


predictions = model.predict(X_train)


mse = mean_squared_error(y_train, predictions)
print(f"Mean Squared Error on Dataset 1: {mse}")


xs_noisy = np.linspace(0, 10, 100)
ys_noisy = xs_noisy + np.random.random(100) * 2 - 1
ys_noisy[25:45] *= 2  # Добавление шума
with open("dataset_noisy.csv", "w") as f:
    f.write("x,y\n")
    for x, y in zip(xs_noisy, ys_noisy):
        f.write(f"{x},{y}\n")


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(xs1, ys1, label="Dataset 1")
plt.plot(xs1, predictions, color="red", label="Model Prediction")
plt.legend()
plt.title("Dataset 1 and Model")

plt.subplot(1, 2, 2)
plt.scatter(xs_noisy, ys_noisy, label="Noisy Dataset")
plt.legend()
plt.title("Noisy Dataset")
plt.show()


def test_model_on_clean_data():
    clean_data = pd.read_csv("dataset1.csv")
    predictions_clean = model.predict(clean_data["x"].values.reshape(-1, 1))
    mse_clean = mean_squared_error(clean_data["y"].values, predictions_clean)
    assert mse_clean < 1.0, "MSE на чистом датасете должно быть меньше 1.0"

def test_model_on_noisy_data():
    noisy_data = pd.read_csv("dataset_noisy.csv")
    predictions_noisy = model.predict(noisy_data["x"].values.reshape(-1, 1))
    mse_noisy = mean_squared_error(noisy_data["y"].values, predictions_noisy)
    assert mse_noisy > 1.0, "MSE на шумном датасете должно быть больше 1.0"


pytest.main(["-v", "-s"])
