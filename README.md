# CV_project

# Running with pretrained models

Creating virtual environment using VirtualEnv:
```bash 
virtualenv -p python3 ./
source ./bin/activate
 ```

Installing requirements

```bash
pip3 install -r requirements.txt
```

**To Run the video** (can find the defaults in the code)

Optional arguments:  
-fd [face detection method] -->'DNN' or 'ViolaJones'   
-cp [path to the pretrained mask classifier]
-th [the probability threshold to detect no mask (alarm)]  (optimal value varies with different models)
```bash
python3 video.py 
``` 

an example with arguments:
```bash
python3 video.py -fd 'DNN' -cp "experiments/classifier5/classifier5.model" -th 0.5
```

**To run the image annotation**

 Optional arguments: (can find the defaults in the code)  
 -fd [face detection method] , -> 'DNN' or 'ViolaJones'  
 -cp [path to the pretrained mask classifier]   
 -sp [path to save the result] (example: './sampleoutput/name.jpg')   if sp is not provided the result will not be saved
 -im [path to the image to be marked]  
 -th [the probability threshold to detect no mask (alarm)]  (optimal value varies with different models)
 ```bash
python3 image.py 
```

an example with arguments:
```bash 
python3 image.py -fd 'DNN' -im './data/IMG_1204.jpg' -sp './testimage_result.jpg'
```

**To recreate the evaluation results**
```bash 
python3 evaluation.py
```

Current pretrained classifiers are:   
"experiments/classifier123/classifier123.model" (trained with dataset 1,2,3)
"experiments/classifier5/classifier5.model" (trained with dataset 5)  
"experiments/classifier_all/classifier_all.model" (trained with all data sets)
"experiments/classifier_real/classifier_real.model" (trained with dataset 1,2)  
"experiments/classifier3/classifier3.model" (trained with dataset 3)  
"experiments/classifier125/classifier125.model" (trained with dataset 1,2,5)

# Training Models

To load the data
```python
from DataInput import load_data
from CNN import train_and_test_model
data, labels = load_data(save=False, directories = ["./data/dataset1-real","./data/dataset2-medical"]) 
# you can use any of the data sets just add the path to the directories array.
# if directories is not given, the algorithm uses database 1 to 4
# default value for save is False
```
To train the model
```python
from CNN import train_and_test_model
model, history = train_and_test_model(data=data, labels=labels, save_path="<path>/model.model")
#if no save path given the model will not be automatically saved
# the function automatically splits data into test and train and prints the test results.
```
To plot training history
```python
from CNN import plot_training_history
plot_training_history(H=history, save_path="<path>/history.pdf")
# plots training and validation loss and accuracy
# we use this to investigate bugs and overfitting.
#if no save path given the plot will not be automatically saved
```
To test a saved model on test images (it prints the results)
```python
from image import test_model_on_test_images
test_model_on_test_images(classifier_path="<path to the model>", threshold = 0.5)
```
