import os, cv2, numpy
from tensorflow.keras import utils as ku

path = '../data/dataset3-artificial'
size = 100
labels_obj = {
    'with_mask': 0,
    'without_mask': 1
}

categories = ['with_mask', 'without_mask']

data = list()
target = list()

for ctg in categories: #loops over the two folders, one of people with mask and one without masks
    folder = os.path.join(path, ctg) #gets the folder path for
    imgs = os.listdir(folder) #gets th elist of images in the folders path

    for i in imgs:
        i_path = os.path.join(folder, i) #gets image path
        img = cv2.imread(i_path) #uses cv2 to read the image

        try:
            resize = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),(size,size)) #turns the image to grey scale and resizes it to 100x100
            data.append(resize)
            target.append(labels_obj[ctg])
        except Exception as e:
            print(e)

data = numpy.array(data) / 255 #makes the images in the range from 0-1
data = numpy.reshape(data, (data.shape[0], size, size, 1)) #reshapes the numpy array as a 4d array to be used in the CNN
target = numpy.array(target)
target = ku.to_categorical(target)

numpy.save('../data', data)
numpy.save('target', target)
