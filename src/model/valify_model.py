import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# --- 설정 ---
# 훈련 시 생성된 모델과 인덱스 파일 경로를 정확하게 지정해야 합니다.
# 훈련 스크립트(korean_foods_model_cho.py)는 'models' 폴더 안에 모델과 json 파일을 저장합니다.
MODEL_PATH = './models/cho_korean_food_classifier-20250827-002505.keras'
INDICES_JSON_PATH = './indices-valid-20250827-002505.json'  # 예시: 실제 훈련된 인덱스 파일 경로로 변경
CHECK_DIR = 'E:\\AIWork\\Data\\테스트\\valid'  # 검증 데이터 경로

# 1. 훈련 시 사용된 클래스 인덱스 불러오기
try:
    with open(INDICES_JSON_PATH, 'r', encoding='utf-8') as f:
        class_indices = json.load(f)
except FileNotFoundError:
    print(f"오류: 클래스 인덱스 파일({INDICES_JSON_PATH})을 찾을 수 없습니다.")
    print("훈련 시 생성된 'indices-*.json' 파일의 정확한 경로를 설정해주세요.")
    exit()

# 인덱스 순서대로 클래스 이름 리스트 생성 (e.g., {'갈비구이': 0, '감자전': 1} -> ['갈비구이', '감자전'])
class_labels = sorted(class_indices.keys(), key=lambda x: class_indices[x])

# 2. 모델 불러오기
try:
    model = load_model(MODEL_PATH)
    print(f"모델 로딩 성공: {MODEL_PATH}")
except (IOError, FileNotFoundError):
    print(f"오류: 모델 파일({MODEL_PATH})을 찾을 수 없습니다.")
    print("훈련된 모델 파일의 정확한 경로를 설정해주세요.")
    exit()

# 3. 검증 데이터 생성기 설정 (훈련과 동일한 클래스 순서 지정)
validation_datagen = ImageDataGenerator(rescale=1. / 255.0)
validation_generator = validation_datagen.flow_from_directory(
    CHECK_DIR,
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical',
    shuffle=False,
    classes=class_labels  # <<< 핵심: 훈련 시의 클래스 순서를 동일하게 적용합니다.
)

# 4. 모델 평가
print("\n모델 평가를 시작합니다...")
loss, accuracy = model.evaluate(validation_generator)
print(f"\n평가 결과:")
print(f"  - Loss: {loss:.4f}")
print(f"  - Accuracy: {accuracy:.4f}")