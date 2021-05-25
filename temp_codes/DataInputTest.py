DIRECTORY = ""
CATEGORIES = ["with_mask", "without_mask"]


data = []
labels = []

for category in CATEGORIES:
    path = os.faces_path.join(DIRECTORY, category)
    for img in os.listdir(path):
    	img_path = os.faces_path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)

    	data.append(image)
    	labels.append(category)

# use one-hot encoding on the labels so we dont have a bunch of strings as are labels 
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)
