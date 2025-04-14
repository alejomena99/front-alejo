# train_xor_model.py

import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
import os

# Datos del problema XOR
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

# Entrenar un modelo MLP (red neuronal simple)
model = MLPClassifier(hidden_layer_sizes=(4,), max_iter=1000, random_state=42)
model.fit(X, y)

# Guardar el modelo
model_dir = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "xor.joblib")
joblib.dump(model, model_path)

print(f"Modelo guardado en: {model_path}")
