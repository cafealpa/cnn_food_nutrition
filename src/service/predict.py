import json
import numpy as np
from tensorflow.keras.models import load_model

def predict(img_array, model_path, indices_path):
    indices_data = {}
    with open(indices_path, "r") as f:    
        indices_data = json.load(f)

    indices_data = {v: k for k, v in indices_data.items()}

    model = load_model(model_path)
    # print(model)
    predictions = model.predict(img_array)

    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = indices_data[predicted_class_index]
    confidence = predictions[0][predicted_class_index]

    returned_values = {
        "predict": [predicted_class_name],
        "confidence": f"{(confidence * 100):.4f}"
    }

    return returned_values