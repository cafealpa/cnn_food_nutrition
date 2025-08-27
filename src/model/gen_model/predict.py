import os, json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image

image_path = "/Users/james/Desktop/dataset/21_korean/kfood_correct/떡/꿀떡/Img_047_0020.jpg"
img = image.load_img(image_path, target_size=(224, 224))  # 모델 입력 크기와 맞춤
img_array = image.img_to_array(img)  # (224, 224, 3)
img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3) - 배치 차원 추가
img_array = img_array / 255.0  # 정규화 (학습시와 동일)

indices_path = "indices-2025-08-25 16:26:29.604277.json"
indices_data = {}
with open(indices_path, "r") as f:    
    indices_data = json.load(f)

indices_data = {v: k for k, v in indices_data.items()}

model = load_model("cho_korean_food_classifier.keras")
# print(model)
predictions = model.predict(img_array)

predicted_class_index = np.argmax(predictions[0])
predicted_class_name = indices_data[predicted_class_index]
confidence = predictions[0][predicted_class_index]

print(f"예측된 클래스: {predicted_class_name}")
print(f"신뢰도: {confidence:.4f}")
print(f"클래스 인덱스: {predicted_class_index}")