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
Developed By: KARTHICK P
Register Number:212222100021
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
![WhatsApp Image 2024-03-22 at 8 54 52 AM](https://github.com/karthiop6/IMAGE-TRANSFORMATIOS/assets/160331179/3a950c17-f34d-4ef9-a8de-e2198d3e8b1a)
![WhatsApp Image 2024-03-22 at 8 55 16 AM](https://github.com/karthiop6/IMAGE-TRANSFORMATIOS/assets/160331179/47c7c16c-4860-4362-a658-061476080d7c)


### ii) Image Scaling
![WhatsApp Image 2024-03-22 at 8 58 04 AM](https://github.com/karthiop6/IMAGE-TRANSFORMATIOS/assets/160331179/7a0a5b1c-d2a1-4d0d-89ab-03bfdc0599bd)

![WhatsApp Image 2024-03-22 at 8 58 17 AM](https://github.com/karthiop6/IMAGE-TRANSFORMATIOS/assets/160331179/7244e522-9bca-4508-94ea-f88f4aaf3139)

### iii)Image sharing
![WhatsApp Image 2024-03-22 at 8 59 10 AM](https://github.com/karthiop6/IMAGE-TRANSFORMATIOS/assets/160331179/3fd33cbd-1b38-45d9-a1e6-2da715c44b4d)

![WhatsApp Image 2024-03-22 at 8 59 32 AM](https://github.com/karthiop6/IMAGE-TRANSFORMATIOS/assets/160331179/eb117dcf-a279-4037-bf89-5b496871703a)

### iv)Image Reflection
![WhatsApp Image 2024-03-22 at 9 02 50 AM](https://github.com/karthiop6/IMAGE-TRANSFORMATIOS/assets/160331179/2a0d1345-788d-41b2-ae87-f0e00241f362)
![WhatsApp Image 2024-03-22 at 9 03 12 AM](https://github.com/karthiop6/IMAGE-TRANSFORMATIOS/assets/160331179/91d9d00d-6799-4175-a4ab-900db89f84d9)
![WhatsApp Image 2024-03-22 at 9 03 05 AM](https://github.com/karthiop6/IMAGE-TRANSFORMATIOS/assets/160331179/e3fa1421-f341-4b55-8eef-f481d4e0cd2a)
![WhatsApp Image 2024-03-22 at 9 02 58 AM](https://github.com/karthiop6/IMAGE-TRANSFORMATIOS/assets/160331179/c78465b4-0f9e-40d0-9a1a-897970740239)

### v)Image rotation
![WhatsApp Image 2024-03-22 at 9 03 59 AM](https://github.com/karthiop6/IMAGE-TRANSFORMATIOS/assets/160331179/176c4bd1-8dd3-4f99-a309-9386c3977fe6)


### vi)Image Cropping
![WhatsApp Image 2024-03-22 at 9 04 49 AM](https://github.com/karthiop6/IMAGE-TRANSFORMATIOS/assets/160331179/a74047a0-4c0c-446c-975f-1c43e8ccc140)
![WhatsApp Image 2024-03-22 at 9 04 56 AM](https://github.com/karthiop6/IMAGE-TRANSFORMATIOS/assets/160331179/50f0a5f1-a858-4b64-bd9f-cb8d3321c707)

## Result: 
Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
