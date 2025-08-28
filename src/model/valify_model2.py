import os
import json
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# --- 설정 ---
# MODEL_PATH = './models/cho_korean_food_classifier-fine-20250827-125904.keras'
MODEL_PATH = './models/cho_korean_food_classifier-fine-20250827-161229.keras'
INDICES_JSON_PATH = './models/indices-fine-20250827-125904.json'

# CHECK_DIR = 'E:\\AIWork\\Data\\테스트\\valid'  # 예측할 이미지가 있는 폴더 경로
CHECK_DIR = 'E:\\AIWork\\Data\\테스트\\valid\\감자탕'  # 예측할 이미지가 있는 폴더 경로

# 1. 훈련 시 사용된 클래스 인덱스 불러오기
try:
    with open(INDICES_JSON_PATH, 'r', encoding='utf-8') as f:
        class_indices = json.load(f)
except FileNotFoundError:
    print(f"오류: 클래스 인덱스 파일({INDICES_JSON_PATH})을 찾을 수 없습니다.")
    exit()

class_labels = sorted(class_indices.keys(), key=lambda x: class_indices[x])

# 2. 모델 불러오기
try:
    model = load_model(MODEL_PATH)
    print(f"모델 로딩 성공: {MODEL_PATH}")
except (IOError, FileNotFoundError):
    print(f"오류: 모델 파일({MODEL_PATH})을 찾을 수 없습니다.")
    exit()

# 3. CHECK_DIR의 이미지에 대한 예측 수행
print(f"\n--- {CHECK_DIR} 폴더의 이미지 예측 시작 ---")

image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
if not os.path.exists(CHECK_DIR) or not os.path.isdir(CHECK_DIR):
    print(f"오류: 체크할 디렉토리({CHECK_DIR})를 찾을 수 없습니다.")
    exit()

image_paths = []
for root, _, files in os.walk(CHECK_DIR):
    for file in files:
        if os.path.splitext(file)[1].lower() in image_extensions:
            image_paths.append(os.path.join(root, file))

if not image_paths:
    print(f"{CHECK_DIR}에서 이미지를 찾을 수 없습니다.")
    exit()

print(f"총 {len(image_paths)}개의 이미지를 찾았습니다. 예측을 시작합니다.\n")

try:
    input_shape = model.input_shape[1:3]
except Exception:
    input_shape = (224, 224)
    print(f"모델 입력 크기를 자동으로 감지할 수 없어 기본값 {input_shape}를 사용합니다.")

# 정답률 계산을 위한 변수
dir_stats = {}
total_correct = 0

for img_path in image_paths:
    try:
        img = image.load_img(img_path, target_size=input_shape)
        img_array = image.img_to_array(img)
        img_array /= 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_batch, verbose=0)

        predicted_index = np.argmax(prediction[0])
        predicted_label = class_labels[predicted_index]
        predicted_confidence = np.max(prediction[0])

        # 실제 정답 (파일이 속한 디렉토리 이름)
        actual_label = os.path.basename(os.path.dirname(img_path))

        # 정답률 통계 업데이트
        if actual_label not in dir_stats:
            dir_stats[actual_label] = {'correct': 0, 'total': 0}
        
        dir_stats[actual_label]['total'] += 1
        is_correct = predicted_label == actual_label
        if is_correct:
            dir_stats[actual_label]['correct'] += 1
            total_correct += 1

        relative_path = os.path.relpath(img_path, CHECK_DIR)
        result_marker = "(정답)" if is_correct else "(오답)"
        print(f"파일: {relative_path}, 예측: {predicted_label}, 실제: {actual_label} {result_marker}")

    except Exception as e:
        print(f"파일 처리 중 오류 발생 {img_path}: {e}")

# --- 예측 결과 요약 ---
print("\n--- 디렉토리별 정답률 ---")

# 디렉토리 이름으로 정렬하여 출력
for dir_name, stats in sorted(dir_stats.items()):
    accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
    print(f"- {dir_name}: {accuracy:.2f}% ({stats['correct']}/{stats['total']})")

# --- 전체 정답률 ---
print("\n--- 전체 정답률 ---")
total_images = len(image_paths)
overall_accuracy = (total_correct / total_images) * 100 if total_images > 0 else 0
print(f">> {overall_accuracy:.2f}% ({total_correct}/{total_images})")
