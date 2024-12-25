import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def create_model():
    model = LinearRegression()
    return model


@pytest.fixture
def clean_data():
    return pd.read_csv("dataset1.csv")

@pytest.fixture
def noisy_data():
    return pd.read_csv("dataset_noisy.csv")

@pytest.fixture
def model(clean_data):
    model = create_model()
    X_train = clean_data["x"].values.reshape(-1, 1)
    y_train = clean_data["y"].values
    model.fit(X_train, y_train)
    return model

# Тест 1: Проверка обучения модели на чистом датасете
def test_model_training(clean_data):
    model = create_model()
    X_train = clean_data["x"].values.reshape(-1, 1)
    y_train = clean_data["y"].values
    model.fit(X_train, y_train)
    assert model.coef_ is not None, "Коэффициенты модели не должны быть None"

# Тест 2: Проверка качества предсказания на чистом датасете
def test_model_on_clean_data(model, clean_data):
    predictions = model.predict(clean_data["x"].values.reshape(-1, 1))
    mse = mean_squared_error(clean_data["y"].values, predictions)
    assert mse < 1.0, "MSE на чистом датасете должно быть меньше 1.0"

# Тест 3: Проверка качества предсказания на шумном датасете
def test_model_on_noisy_data(model, noisy_data):
    predictions = model.predict(noisy_data["x"].values.reshape(-1, 1))
    mse = mean_squared_error(noisy_data["y"].values, predictions)
    assert mse > 1.0, "MSE на шумном датасете должно быть больше 1.0"

# Тест 4: Проверка размера предсказаний
@pytest.mark.parametrize("dataset_name", ["dataset1.csv", "dataset_noisy.csv"])
def test_predictions_size(dataset_name, model):
    data = pd.read_csv(dataset_name)
    predictions = model.predict(data["x"].values.reshape(-1, 1))
    assert len(predictions) == len(data), "Размер предсказаний должен совпадать с размером входных данных"

# Тест 5: Проверка коэффициента модели
@pytest.mark.parametrize("expected_coefficient", [1.0, 2.0, -1.0])
def test_model_coefficient(expected_coefficient, clean_data):
    xs = clean_data["x"].values.reshape(-1, 1)
    ys = expected_coefficient * xs.ravel() + np.random.random(len(xs)) * 2 - 1
    model = create_model()
    model.fit(xs, ys)
    np.testing.assert_almost_equal(model.coef_[0], expected_coefficient, decimal=1, 
        err_msg="Коэффициент модели должен быть близок к ожидаемому")
