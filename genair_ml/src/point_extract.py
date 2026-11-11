import os
import re
import numpy as np
from src.config import ROOT_DIR

root_dir = ROOT_DIR

def parsec_to_points(coeffs, X):
    powers = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    Z = np.sum(coeffs * X[:, None]**powers, axis=1)
    return Z

dataset = []
theta = np.linspace(0, np.pi, 50)
x = (1 - np.cos(theta)) / 2
x[0] = 1e-4

for root, dirs, files in os.walk(root_dir):

    for file in files:

        if file.endswith(".txt"):

            path = os.path.join(root, file)

            with open(path, "r") as f:
                text = f.read()

            dorso = re.search(r"Dorso \[c1\.\.c6\]:\s*\[(.*?)\]", text)
            pancia = re.search(r"Pancia \[c1\.\.c6\]:\s*\[(.*?)\]", text)

            if dorso and pancia:

                dorso_vals = [float(v) for v in dorso.group(1).split(",")]
                pancia_vals = [float(v) for v in pancia.group(1).split(",")]

                p_dorso = parsec_to_points(dorso_vals, x)
                p_pancia = parsec_to_points(pancia_vals, x)
                p_total = np.concatenate([p_dorso, p_pancia])

                dataset.append(p_total)

dataset = np.array(dataset)
min_val = dataset.min(axis=0)
max_val = dataset.max(axis=0)
dataset = (dataset - min_val) / (max_val - min_val + 1e-8)

print(min_val, max_val)

dataset_training = dataset[:1000]
dataset_validation = dataset[-621:]

print("Complete dataset: ", dataset.shape)
print("Training dataset: ", dataset_training.shape)
print("Validation dataset: ", dataset_validation.shape)