# IMAGE-TRANSFORMATIONS


## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
Import numpy module as np and pandas as pd.
### Step2:
Assign the values to variables in the program.
### Step3:
Get the values from the user appropriately.
### Step4:
Continue the program by implementing the codes of required topics.
### Step5:
Thus the program is executed in google colab.
## Program:
```
Developed By:Vasanthamukilan M
Register Number:212222230167
```
### i)Image Translation
```python
import cv2
import numpy as np
from matplotlib import pyplot as plt
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
image_url = 'rose.jpg'  
image = cv2.imread(image_url)
tx = 50
ty = 30  
translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]]) 
translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
print("Original Image:")
show_image(image)
print("Translated Image:")
show_image(translated_image)
```
### ii) Image Scaling
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
in_img=cv2.imread("flower1.jpg")
in_img=cv2.cvtColor(in_img,cv2.COLOR_BGR2RGB)
rows,cols,dim=in_img.shape
M=np.float32([[1.5,0 ,0],
              [0,1.8,0],
              [0,0,1]])
scaled_img=cv2.warpPerspective(in_img, M,(cols,rows))
plt.axis('off')
plt.imshow(scaled_img)
plt.show()
```

### iii)Image shearing

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
image_url = 'flower3.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)
shear_factor_x = 0.5  
shear_factor_y = 0.2 
shear_matrix = np.float32([[1, shear_factor_x, 0], [shear_factor_y, 1, 0]])
sheared_image = cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))
print("Original Image:")
show_image(image)
print("Sheared Image:")
show_image(sheared_image)
```
### iv)Image Reflection
```python
import cv2
import numpy as np
from matplotlib import pyplot as plt
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
image_url = 'flower2.jpg'
image = cv2.imread(image_url)
reflected_image_horizontal = cv2.flip(image, 1)
reflected_image_vertical = cv2.flip(image, 0)
reflected_image_both = cv2.flip(image, -1)
print("Original Image:")
show_image(image)
print("Reflected Horizontally:")
show_image(reflected_image_horizontal)
print("Reflected Vertically:")
show_image(reflected_image_vertical)
print("Reflected Both:")
show_image(reflected_image_both)
```
### v)Image Rotation
```python
import cv2
import numpy as np
from matplotlib import pyplot as plt
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
image_url = 'flower4.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)
angle = 45
height, width = image.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
print("Original Image:")
show_image(image)
print("Rotated Image:")
show_image(rotated_image)
```
### vi)Image Cropping
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
in_img = cv2.imread("flower5.jpg")
in_img = cv2.cvtColor(in_img,cv2.COLOR_BGR2RGB)
plt.imshow(in_img)
plt.show()
cropped_img=in_img[2000:3000 ,1000:2500]
plt.imshow(cropped_img)
plt.show()
```
## Output:
### i)Image Translation
<br>![download](https://github.com/Vasanthamukilan/IMAGE-TRANSFORMATIONS/assets/119559694/e3614dcd-cae4-45d4-9ec2-2c30bc631d41)<br>
<br>![download](https://github.com/Vasanthamukilan/IMAGE-TRANSFORMATIONS/assets/119559694/74d28a3c-67a9-415e-b344-33f5ff5c94b1)<br>
### ii) Image Scaling
<br>![download](https://github.com/Vasanthamukilan/IMAGE-TRANSFORMATIONS/assets/119559694/9a10e679-f3c7-4df4-acf9-8ba330f5f442)<br>
<br>![download](https://github.com/Vasanthamukilan/IMAGE-TRANSFORMATIONS/assets/119559694/c1649edb-ccbc-495f-b490-aa038582d301)<br>
### iii)Image shearing
<br>![download](https://github.com/Vasanthamukilan/IMAGE-TRANSFORMATIONS/assets/119559694/c70fd227-96f4-429d-9c7d-1f26a58fa825)<br>
<br> ![download](https://github.com/Vasanthamukilan/IMAGE-TRANSFORMATIONS/assets/119559694/5743770b-89fb-497b-9737-f32ee748dbc1)<br>
### iv)Image Reflection
<br>![download](https://github.com/Vasanthamukilan/IMAGE-TRANSFORMATIONS/assets/119559694/9494ad60-0cc9-4c80-b35b-1a0fb4a1a642)<br>
<br>![download](https://github.com/Vasanthamukilan/IMAGE-TRANSFORMATIONS/assets/119559694/a35b86ba-2464-44ec-b22c-f8b5b58a8d41)<br>
<br>![download](https://github.com/Vasanthamukilan/IMAGE-TRANSFORMATIONS/assets/119559694/a3fefe9c-4e00-4739-a53f-52b7b9e1d896)<br>
<br>![download](https://github.com/Vasanthamukilan/IMAGE-TRANSFORMATIONS/assets/119559694/f67bdf89-630c-4fd1-8682-cbe21094b685)<br>
### v)Image Rotation
<br>![download](https://github.com/Vasanthamukilan/IMAGE-TRANSFORMATIONS/assets/119559694/f4dc04f3-c9f0-451f-9dd5-728c5f00ad85)<br>
<br>![download](https://github.com/Vasanthamukilan/IMAGE-TRANSFORMATIONS/assets/119559694/d7fee572-ede4-4100-a295-0782c5d13eb1)<br>
### vi)Image Cropping
<br>![download](https://github.com/Vasanthamukilan/IMAGE-TRANSFORMATIONS/assets/119559694/c06100b0-4d91-45a1-b350-5dca1d2ff6de)<br>
<br>![download](https://github.com/Vasanthamukilan/IMAGE-TRANSFORMATIONS/assets/119559694/1737b1eb-3b83-4c40-ad1d-9fd003c7ccd3)<br>
## Result: 
Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
