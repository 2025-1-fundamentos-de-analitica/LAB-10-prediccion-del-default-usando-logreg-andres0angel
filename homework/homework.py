# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
import gzip
import json
import pickle
import zipfile


def read_csv_from_zip(zip_path):
    with zipfile.ZipFile(zip_path, "r") as z:
        file_name = z.namelist()[0]
        with z.open(file_name) as f:
            return pd.read_csv(f)
        
def clean_data(data):
    """Limpieza de los datasets."""

    data.rename(columns={"default payment next month": "default"}, inplace=True)
    data.drop(columns=["ID"], inplace=True)
    data.dropna(inplace=True) 
    print(data.shape)
    data= data[(data["EDUCATION"]!=0) & (data["MARRIAGE"]!=0)] 
    print(data.shape)
    data["EDUCATION"] = data["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

    return data


def classifier_model():
    """Creación un pipeline para el modelo de clasificación."""
    
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    numerical_features = [
        "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", 
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    model = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(score_func=f_classif, k=10)), 
    ('classifier', classifier )
    ])

    return model

def optimize_hyperparameters(model, x_train, y_train):
    """Optimización los hiperparametros del pipeline usando validación cruzada."""

    param_grid = {
        "feature_selection__k": range(1, 11), 
        "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100], 
        "classifier__solver": ["liblinear"],
       
    }
    search = GridSearchCV(model, param_grid, n_jobs=-1, cv=10, scoring="balanced_accuracy", refit=True )
    search.fit(x_train, y_train)

    return search


def save_model(model):
    """Guardar el modelo (comprimido con gzip)."""

    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(model, f)

def metrics(model, x_train, y_train, x_test, y_test):
    """Calculo las metricas de precision, precision balanceada"""

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    metrics_train = {
        "type": "metrics",
        "dataset": "train",
        "precision": float(precision_score(y_train, y_train_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_train, y_train_pred)),
        "recall": float(recall_score(y_train, y_train_pred)),
        "f1_score": float(f1_score(y_train, y_train_pred))
    }

    metrics_test = {
        "type": "metrics",
        "dataset": "test",
        "precision": float(precision_score(y_test, y_test_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_test_pred)),
        "recall": float(recall_score(y_test, y_test_pred)),
        "f1_score": float(f1_score(y_test, y_test_pred))
    }

    return metrics_train, metrics_test

def confusion_matrix_metrics(model, x_train, y_train, x_test, y_test):
    """Calculo las matrices de confusion."""

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    cm_metrics_train = {
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {"predicted_0": int(cm_train[0, 0]), "predicted_1": int(cm_train[0, 1])},
        "true_1": {"predicted_0": int(cm_train[1, 0]), "predicted_1": int(cm_train[1, 1])}
    }

    cm_metrics_test = {
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {"predicted_0": int(cm_test[0, 0]), "predicted_1": int(cm_test[0, 1])},
        "true_1": {"predicted_0": int(cm_test[1, 0]), "predicted_1": int(cm_test[1, 1])}
    }

    return cm_metrics_train, cm_metrics_test


def save_metrics(metrics_train, metrics_test, cm_metrics_train, cm_metrics_test, file_path="files/output/metrics.json"):
    """Guarda las métricas en un archivo JSON"""
    metrics_data = [metrics_train, metrics_test, cm_metrics_train, cm_metrics_test]

    with open(file_path, "w") as f:
        for item in metrics_data:
            f.write(json.dumps(item) + "\n")
import os  # Asegúrate de importar esto si no lo tienes

def ensure_directories():
    """Crea las carpetas necesarias si no existen."""
    os.makedirs("files/models", exist_ok=True)
    os.makedirs("files/output", exist_ok=True)


train_zip_path = "files/input/train_data.csv.zip"
test_zip_path = "files/input/test_data.csv.zip"
       
train_data = clean_data(read_csv_from_zip(train_zip_path))
test_data = clean_data(read_csv_from_zip(test_zip_path))

x_train = train_data.drop(columns=["default"])  
y_train = train_data["default"]

x_test = test_data.drop(columns=["default"])
y_test = test_data["default"]


model = classifier_model()
best_model = optimize_hyperparameters(model, x_train, y_train)
print("Modelo después de optimización:", type(best_model))

ensure_directories()
save_model(best_model)

metrics_train, metrics_test = metrics(best_model, x_train, y_train, x_test, y_test)
cm_metrics_train, cm_metrics_test = confusion_matrix_metrics(best_model, x_train, y_train, x_test, y_test)
save_metrics(metrics_train, metrics_test, cm_metrics_train, cm_metrics_test)