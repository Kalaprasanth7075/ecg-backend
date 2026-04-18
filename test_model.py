import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load saved model
model = load_model("ecg_mobilenet_model.h5")

IMG_SIZE = 224

# Change path to new ECG image
img_path = img_path = input("Enter full image path: ")

img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

predictions = model.predict(img_array)
class_index = np.argmax(predictions)

classes = ['AH', 'H_MI', 'MI', 'Normal']

print("Prediction Probabilities:", predictions)
print("Predicted Class:", classes[class_index])