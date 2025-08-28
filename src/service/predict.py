import json
import numpy as np
from tensorflow.keras.models import load_model

DB_CONVERT_DIC = {
    "간장게장": "게장_간장",
    "감자채볶음": "감자볶음",
    "계란국": "달걀국",
    "계란말이": "달걀말이",
    "계란찜": "달걀찜",
    "계란후라이": "달걀후라이",
    "고추장진미채볶음": "오징어볶음",
    "곰탕_설렁탕": "설렁탕",
    "꽈리고추무침": "오이지무침_고추",
    "닭계장": "닭볶음탕",
    "도라지무침": "도라지생채",
    "떡국_만두국": "떡국_소고기",
    "떡꼬치": "떡강정",
    "북엇국": "북어국",
    "새우볶음밥": "볶음밥_새우",
    "소세지볶음": "소시지케첩볶음",
    "시래기국": "된장국_시래기",
    "양념게장": "게장_양념",
    "열무국수": "국수_열무김치",
    "젓갈": "양념오징어젓",
    "편육": "수육",
    "한과": "유과",
}

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
        "predict": [convert_class_name_db(predicted_class_name)],
        "confidence": f"{(confidence * 100):.4f}"
    }

    return returned_values

def convert_class_name_db(class_name):
    if class_name in DB_CONVERT_DIC.keys():
        return DB_CONVERT_DIC[class_name]
    else:
        return class_name

# if __name__ == "__main__":
#
#     print(type(DB_CONVERT_DIC.keys()))
#
#     test_list = [None, "abcd","과메기"]
#     test_list = test_list + list(DB_CONVERT_DIC.keys())
#
#     for name in test_list:
#         converted = convert_class_name_db(name)
#         print(f"{name} -> {converted}")