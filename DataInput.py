import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from detector import dnn_face_detection
import cv2

# NOTE this reads data as 224*224 colored images. Which is compatible with the CNN right now
#  The other one creates grayscale 100*100, which is not compatible with the CNN
#  MobileNetV2 gets 224*244 colored images.


# NOTE didn't add dataset5 to the default directories because it makes training slow
DIRECTORIES = ["./data/dataset3-artificial",
               "./data/dataset1-real",
               "./data/dataset2-medical",
               "./data/dataset4-artificial"]
CATEGORIES = ["with_mask", "without_mask"]


def load_data(save=False, directories=None):
    data = []
    labels = []
    if directories is None:
        directories = DIRECTORIES
    # read data from the files
    for directory in directories:
        for category in CATEGORIES:
            path = os.path.join(directory, category)
            for img in os.listdir(path):
                img_path = os.path.join(path, img)
                image = load_img(img_path,target_size=(224, 224))  # load the image
                image = img_to_array(image)
                image = preprocess_input(image)  # preprocessing the data for the mobileNetV2 CNN
                data.append(image)
                labels.append(category)

    # use one-hot encoding on the labels so we dont have a bunch of strings as our labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)

    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    # Shuffle data (to make sure different datasets are mixed together)
    idx = np.random.permutation(labels.shape[0])
    data, labels = data[idx, :], labels[idx, :]

    if save:
        np.save('data', data)
        np.save('label', labels)

    return data, labels
