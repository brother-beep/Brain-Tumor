import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import matplotlib.pyplot as plt

model = load_model("brain-tumors.h5")

image = cv2.imread(r"E:\Tumor(Brain)\brain_tumor_dataset\pred\pred8.jpg")
imag = Image.fromarray(image)
imag =imag.resize((64,64))
img = np.array(imag)
input_img = np.expand_dims(img,axis=0)

result = model.predict(input_img)
print(result)