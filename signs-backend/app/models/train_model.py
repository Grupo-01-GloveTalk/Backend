# train_model.py
# Script para cargar, entrenar y guardar un modelo de clasificación de gestos a partir de datos de sensores (acelerómetro y giroscopio)

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # sube de models -> app -> signs-backend
data_path = BASE_DIR / "data" / "data_clean" / "dataset.csv"
df = pd.read_csv(data_path)

# Eliminar columnas no numéricas (timestamp)
features = df.drop(columns=["timestamp", "label"])
labels = df["label"]

# Codificar las etiquetas (HOLA, CHAU, A, etc.)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Separar los datos
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

# Normalización por fila (cada muestra)
X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
X_test = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)
#mm

# Crear el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(60,)),  # 60 features
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(labels_encoded)), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.2)

# Evaluar
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'\nPrecisión en el conjunto de prueba: {test_acc:.2f}')

# Guardar modelo entrenado
model.save("gesture_model.h5")
