from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report
import numpy as np
from matplotlib import pyplot as plt

INIT_LR = 1e-4  # initial learning rate
EPOCHS = 10  #
BS = 32  # Batch size


# ---------------------------------------------------------------
# Elie's stuff with some comment added to that. Didn't change his.


def train_and_test_model(data, labels, save_path=None):
    """

    :param data:
    :param labels:
    :param save_path: path to save the trained model
    :return: the trained model and training history
    """
    # Splitting data
    (trainX, testX, trainY, testY) = \
        train_test_split(data, labels,
                         test_size=0.20, stratify=labels,
                         random_state=42)

    # used this for data augmentation while training
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,  # sort of like affine
        horizontal_flip=True,
        fill_mode="nearest",  # how to fill points outside the image boundaries
        # brightness_range=[0.8, 1.2] # adding this ruined accuracy don't know why
    )

    baseModel = MobileNetV2(weights="imagenet",
                            include_top=False,  # because we're adding our own head to the model
                            input_tensor=Input(shape=(224, 224, 3)),
                            input_shape=(224, 224, 3)
                            )
    # input is BS*224*224*3 output is input is BS*7*7*1280

    # adding a DNN on top of the baseModes. This is what we train
    # base
    # -->
    # AveragePooling (Shrinks data by taking average in each 7*7 box (so data size with be 1/7 in x and y)
    # (it'll become BS*1*1*1280))
    # -->
    # Flatten (creates a 1d array (size: 1280))
    # -->
    # Dense + relu (a fully connected layer)
    # -->
    # dropout (used for preventing over-fitting) This causes training accuracy be less that validation accuracy
    # -->
    # Dense + softmax (a fully connected layer with softmax activation) for classification with two classes
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)

    model = Model(inputs=baseModel.input, outputs=headModel)

    # don't train the mobileNetV2
    for layer in baseModel.layers:
        layer.trainable = False

    # compile model, get ready to train
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    # training (after training we have the classifier. We just need to give it data)
    # H is the training history (loss and accuracy after each epoch(: iterating the whole dataset))
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BS,
        epochs=EPOCHS)

    # testing (it's called cross validation?)
    # getting the predicted probabilities
    predIdxs = model.predict(testX, batch_size=BS)

    # printing report
    print(classification_report(testY.argmax(axis=1), predIdxs.argmax(axis=1)))

    # save the model. #
    if save_path is not None:
        model.save(save_path, save_format="h5")

    return model, H


def plot_training_history(H, save_path=None):
    """

    :param H: Training History
    :param save_path: Path to save the training history plot
    """
    # plot training history (to look for over-fitting (and possible bugs))
    N = EPOCHS
    plt.figure()
    plt.axis("off")
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    ax1.plot(np.arange(0, N), H.history["val_loss"], label="validation_loss")
    ax1.legend()
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.title("Training Loss")

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(np.arange(0, N), H.history["accuracy"], label="train_accuracy")
    ax2.plot(np.arange(0, N), H.history["val_accuracy"], label="validation_accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Training Accuracy")
    plt.title("Training Accuracy")
    ax2.legend()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()
