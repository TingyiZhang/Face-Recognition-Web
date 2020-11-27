import numpy as numpy
import cv2
import os

from glob import glob

FEMALE_PATH = glob('./Module-2/data/female/*.jpg')
MALE_PATH = glob('./Module-2//data/male/*.jpg')

HAAR = cv2.CascadeClassifier('./Module-2/model/haarcascade_frontalface_default.xml')


def extract_faces(path, gender, i):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = HAAR.detectMultiScale(gray, 1.5, 5)

    for x, y, w, h in faces:
        roi = img[y: y+h, x: x+w]
        cv2.imwrite('./crop_imgs/{}/{}_{}.png'.format(gender, gender, i), roi)


for i, path in enumerate(FEMALE_PATH):
    extract_faces(path, 'female', i)
    print('{}/{} female images processed'.format(i, len(MALE_PATH)))

for i, path in enumerate(MALE_PATH):
    extract_faces(path, 'male', i)
    print('{}/{} male images processed'.format(i, len(MALE_PATH)))