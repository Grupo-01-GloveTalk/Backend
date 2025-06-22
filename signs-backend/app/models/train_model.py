# train_model.py

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Cargar el dataset
data_path = os.path.join("..", "data", "data_clean", "combined_data.csv")  # Ajusta si el nombre cambia
df = pd.read_csv(data_path)

# Seleccionar solo columnas de acelerómetro y giroscopio
features = df[['ax', 'ay', 'az', 'gx', 'gy', 'gz']]
labels = df['gesture']  # Ajusta si el nombre de columna es diferente

# Codificar las etiquetas de texto a números
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Separar datos
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)
