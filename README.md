# YOLOv8-Fall-detection
#Training YOLOv8 on Falling Dataset to Enable Real-Time Fall Detection
# **Fall Detection using YOLOv8**

![alt text](https://github.com/pahaht/Falbot/blob/main/Images/fall-all.JPG )  

### *Overview*
This project focuses on training YOLOv8 on a Falling Dataset with the goal of enabling real-time fall detection.
It constitutes a comprehensive initiative aimed at harnessing the capabilities of YOLOv8, a cutting-edge object
detection model, to enhance the efficiency of fall detection in real-time scenarios.

### *Installation*
To install YOLOv8, please follow this link, which contains comprehensive 
documentation that will be beneficial to you: https://docs.ultralytics.com/

### *Dataset*
In this project, I used the Roboflow dataset :https://universe.roboflow.com/hero-d6kgf/yolov5-fall-detection


### *Training YOLOv8 based on Fall Dataset*
-To understand well the training process follow this link :https://docs.ultralytics.com/modes/train/#why-choose-ultralytics-yolo-for-training
-To customize YOLOv8 for the fall dataset, run the following command : 

<pre><code>
from ultralytics import YOLO 
# Load the model.
model = YOLO('yolov8n.pt') 
# Training.
results = model.train(
   data=r"D:\ALL DATASET\Fall_Dataset\data.yaml",    
   imgsz=640,
   epochs=100,
   batch=8,)
</code></pre>

### *Use of fall detection pre-trained model*

To find the weight(pre-trained model) after training YOLOv8 on a Fall dataset, use the best.pt file,
which is automatically stored in the runs/detect/train/weights directory.


### *Loading fall detection pre-trained model*
<pre><code>
from ultralytics import YOLO
model = YOLO(r'D:\My-Pretrain\f-e-100.pt') # f-e-100.pt it is the weight (best.pt I change to f-e-100)
</code></pre>

### *Model Evaluation*
#### *Correlogram*
<pre><code>
from IPython.display import Image
# Display the image
Image(filename=r'C:\Users\user\YOLOhuit-FALL-detection\runs\detect\train3\labels_correlogram.jpg', width=600)
</code></pre>

#### *Results*
<pre><code>
from IPython.display import Image
# Display the image
Image(filename=r'C:\Users\user\YOLOhuit-FALL-detection\runs\detect\train3\labels.jpg', width=600)
</code></pre>

![alt text](https://github.com/pahaht/Fall-detection-using-YOLOv8/blob/main/Images/results.JPG )  

<pre><code>
from IPython.display import Image
# Display the image
Image(filename=r'C:\Users\user\YOLOhuit-FALL-detection\runs\detect\train3\results.png', width=600)
</code></pre>

#### *Confusion matrix*
<pre><code>
from IPython.display import Image
# Display the image
Image(filename=r'C:\Users\user\YOLOhuit-FALL-detection\runs\detect\val5\confusion_matrix_normalized.png', width=600)
</code></pre>

#### *Curve*
<pre><code>
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_paths = [
    r'C:\Users\user\YOLOhuit-FALL-detection\runs\detect\val5\F1_curve.png',
    r'C:\Users\user\YOLOhuit-FALL-detection\runs\detect\val5\P_curve.png',
    r'C:\Users\user\YOLOhuit-FALL-detection\runs\detect\val5\PR_curve.png',
    r'C:\Users\user\YOLOhuit-FALL-detection\runs\detect\val5\R_curve.png'
]

fig, axs = plt.subplots(2, 2, figsize=(15, 15))
for i, ax in enumerate(axs.flatten()):
    img = mpimg.imread(image_paths[i])
    ax.imshow(img)
    ax.axis('off')  

plt.tight_layout()

plt.show()

</code></pre>


### *Prediction(1)*

<pre><code>
# Prediction fall detection
from PIL import Image
# from PIL
im1 = Image.open(r"C:\deep\c8.jpg")
results = model.predict(source=im1, save=True)  
</code></pre>

### *Display predicted image(1)*

<pre><code>
display(Image.open('runs/detect/predict5/c8.jpg'))
</code></pre>


### *Prediction(2)*
<pre><code>
im1 = Image.open(r"C:\deep\c12.jpg")
results = model.predict(source=im1, save=True) 
</code></pre>

### *Display predicted image(2)*
display(Image.open('runs/detect/predict5/c12.jpg'))

### *Prediction(3)*
<pre><code>
im1 = Image.open(r"C:\deep\f7.jpg")
results = model.predict(source=im1, save=True)  
</code></pre>

### *Display predicted image(3)*
<pre><code>
display(Image.open('runs/detect/predict5/f7.jpg'))
</code></pre>
