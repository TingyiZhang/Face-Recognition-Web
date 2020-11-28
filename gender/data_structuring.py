import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pickle

from PIL import Image
from glob import glob

FEMALE_PATH = glob('./crop_imgs/female/*.png')
MALE_PATH = glob('./crop_imgs/male/*.png')
DATA_PATH = FEMALE_PATH + MALE_PATH


def get_size(path):
    img = Image.open(path)
    return img.size[0]


def get_gender(path):
    return path.split('_')[1].split('/')[-1]


def resize_img(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    size = gray.shape[0]

    if size >= 100:
        # Shrink the image
        gray_re = cv2.resize(gray, (100, 100), cv2.INTER_AREA)
    else:
        # Enlarge the image
        gray_re = cv2.resize(gray, (100, 100), cv2.INTER_CUBIC)

    # Flatten the image
    return gray_re.flatten()


df = pd.DataFrame(data=DATA_PATH, columns=['path'])
df['size'] = df['path'].apply(get_size)

# Remove images that are too small
df = df[df['size'] > 60]
df['gender'] = df['path'].apply(get_gender)

# Resize all images
df['resized'] = df['path'].apply(resize_img)

# Expanding the flatten data
df_flatten = df['resized'].apply(pd.Series)

df_train = pd.concat((df['gender'], df_flatten), axis=1)

pickle.dump(df_train, open('./structured_data.pickle', 'wb'))