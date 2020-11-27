import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pickle
from PIL import Image

df = pickle.load(open('./structured_data.pickle', 'rb'))
print(df.info())

# Removing null
df.dropna(axis=0, inplace=True)

# Split training data and the ground truth
X = df.iloc[:, 1:].values  # structured images
y = df.iloc[:, 0].values  # gender

# Scaling
X_norm = X / X.max()
y_norm = np.where(y == 'female', 1, 0)  # female: 1, male: 0

np.savez('./norm_data.npz', X_norm, y_norm)