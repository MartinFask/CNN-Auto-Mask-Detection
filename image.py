import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from detector import violaJones_face_detection, dnn_face_detection
import argparse


# face detection options (Just to make to code clean)
fd_options = {
    'ViolaJones': violaJones_face_detection,
    'DNN': dnn_face_detection
}
# default classifier model and detection method
DEFAULT_CLASSIFIER_PATH = "experiments/classifier5/classifier5.model"
DEFAULT_FACEDETECTION_METHOD = "DNN"
# default image
DEFAUL_IMAGE_PATH = "./data/IMG_1204.jpg"


# THIS FUNCTION GETS AN IMAGE AND MARKS THE FACES AS MASK OR NO MASK AND SHOWS THE RESULT
def detect_and_mark(classifier_path=None, fd_method=None, image_path=None, save_path=None, threshold=0.5):
    """

    :param classifier_path: path to the classifier model
    :param fd_method: 'ViolaJones' or 'DNN' default is ViolaJones
    :param image_path: path to the image to be marked
    :param save_path: path to save the marked image
    """

    # Load the model
    if classifier_path is None:
        model = load_model(DEFAULT_CLASSIFIER_PATH)
    else:
        model = load_model(classifier_path)

    # set the face detector function
    if fd_method is None:
        face_detect = fd_options[DEFAULT_FACEDETECTION_METHOD]
    else:
        face_detect = fd_options[fd_method]

    # set the path to the image
    if image_path is None:
        image_path = DEFAUL_IMAGE_PATH

    # read the image
    image = cv2.imread(image_path).copy()

    # detect the faces
    faces = face_detect(image)

    # loop over the detections and annotate the image
    for x1, y1, x2, y2 in faces:
        # crop the detected face from the frame
        face = image[y1:y2, x1:x2, :]
        # dnn sometimes gives faces with 0 dimensions to prevent error:
        if face.shape[0] == 0 or face.shape[1] == 0:
            continue
        # preprocessing the face for the mask classifier
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        # pass the face to the mask classifier and get the label
        predIdxs = model.predict(face)[0]

        # add the frames around the faces based on the prediction
        if predIdxs[1] > threshold: # no mask
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 16)
        else: # mask
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 16)

    # show the annotated image
    plt.axis("off")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

    # if save_path provided save the annotated image
    if save_path is not None:
        cv2.imwrite(save_path, image)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-cp", "--classifier_path", required=False, default=DEFAULT_CLASSIFIER_PATH,
                    help="path to the classifier model")
    ap.add_argument("-fd", "--face_detection_method", required=False, default=DEFAULT_FACEDETECTION_METHOD,
                    help="face detection method 'DNN' or 'ViolaJones'")
    ap.add_argument("-sp", "--save_path", required=False, default=None,
                    help="path to save the result")
    ap.add_argument("-im", "--image_path", required=False, default=DEFAUL_IMAGE_PATH,
                    help="path to the image")
    ap.add_argument("-th", "--threshold", required=False, default=0.5, type=float,
                    help="threshold")
    args = vars(ap.parse_args())
    print("face_detection_method is : {}".format(args['face_detection_method']))
    print("classifier is : {}".format(args['classifier_path']))

    detect_and_mark(classifier_path=args['classifier_path'],
                    fd_method=args['face_detection_method'],
                    image_path=args['image_path'],
                    save_path=args['save_path'])
