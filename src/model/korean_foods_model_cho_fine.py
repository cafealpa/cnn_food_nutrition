import datetime
import json
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# --- 설정 변수 ---
# 경로, 이미지 크기, 배치 사이즈 등을 변수로 관리하여 유지보수를 용이하게 합니다.
TRAIN_DIR = 'E:\\AIWork\\Data\\테스트\\train'  # 사용자의 기존 경로 유지
VALID_DIR = 'E:\\AIWork\\Data\\테스트\\valid'  # 사용자의 기존 경로 유지
MODEL_SAVE_DIR = './models'
TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 150
INITIAL_EPOCHS = 100  # 초기 학습 에포크
FINE_TUNE_EPOCHS = 20  # 미세 조정 에포크

# 훈련 데이터 생성기: 데이터 증강 적용
train_datagen = ImageDataGenerator(
    rescale=1. / 255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

# 검증 데이터 생성기: 데이터 증강 없이 스케일링만 적용
valid_datagen = ImageDataGenerator(
    rescale=1. / 255.0,
)

validation_generator = valid_datagen.flow_from_directory(
    VALID_DIR,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
indices_json_file = os.path.join(MODEL_SAVE_DIR, f"indices-fine-{current_time}.json")

with open(indices_json_file, "w", encoding='utf-8') as f:
    json.dump(train_generator.class_indices, f, ensure_ascii=False, indent=4)

# 모델 불러오기 (사전 학습된 가중치 사용. 최사위 레이어 제거)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
# --- Dense 층 추가 및 Dropout 적용 ---
# 모델의 표현력을 높이고 과적합을 방지하기 위해 Dense 층과 Dropout을 추가합니다.
x = Dropout(0.5)(x)  # 50%의 뉴런을 랜덤하게 비활성화하여 과적합 방지
x = Dense(512, activation='relu')(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

print("--- 1단계: 상위 분류기 학습 시작 ---")
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

earlyStopping = EarlyStopping(monitor="val_loss", patience=10, verbose=1, restore_best_weights=True)
model_path = os.path.join(MODEL_SAVE_DIR, f"cho_korean_food_classifier-fine-{current_time}.keras")
modelCheckpoint = ModelCheckpoint(model_path, monitor="val_loss", verbose=1, save_best_only=True)

history = model.fit(train_generator,
                    epochs=INITIAL_EPOCHS,
                    validation_data=validation_generator,
                    steps_per_epoch=train_generator.samples // BATCH_SIZE,
                    validation_steps=validation_generator.samples // BATCH_SIZE,
                    callbacks=[earlyStopping, modelCheckpoint]
                    )

print("\n--- 2단계: 미세 조정(Fine-tuning) 시작 ---")
# 베이스 모델의 일부 상위 레이어의 동결을 해제합니다.
base_model.trainable = True

# MobileNetV2는 154개의 레이어로 구성되어 있습니다. 상위 54개 레이어만 학습 대상으로 설정합니다.
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# 매우 낮은 학습률로 모델을 다시 컴파일합니다.
# 사전 학습된 가중치가 급격하게 변하는 것을 방지하기 위함입니다.
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

total_epochs = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

model.fit(train_generator,
          epochs=total_epochs,
          initial_epoch=history.epoch[-1],  # 이전 학습이 끝난 지점부터 시작
          validation_data=validation_generator,
          steps_per_epoch=train_generator.samples // BATCH_SIZE,
          validation_steps=validation_generator.samples // BATCH_SIZE,
          callbacks=[earlyStopping, modelCheckpoint]  # 동일한 콜백 사용
          )

print(f"훈련 완료. 최적 모델이 {model_path} 에 저장되었습니다.")
