import datetime, json
from keras import Model
from keras.src.applications.mobilenet_v2 import MobileNetV2
from keras.src.layers import Dense, Dropout
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

train_dir = 'E:\\AIWork\\Data\\테스트\\train'
valid_dir = 'E:\\AIWork\\Data\\테스트\\valid'
# rabbit_train_dir = './train/rabbit'
# raccoon_train_dir = './train/raccoon'
# squirrel_train_dir = './train/squirrel'

train_datagen = ImageDataGenerator(
    rescale=1. / 255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

valid_datagen = ImageDataGenerator(
    rescale=1. / 255.0,
)

validation_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical',
    shuffle=False
)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

indices_train_json_file = f"indices-train-{current_time}.json"
with open(indices_train_json_file, "w", encoding='utf-8') as f:
    json.dump(train_generator.class_indices, f, ensure_ascii=False, indent=4)

indices_valid_json_file = f"indices-valid-{current_time}.json"
with open(indices_valid_json_file, "w", encoding='utf-8') as f:
    json.dump(validation_generator.class_indices, f, ensure_ascii=False, indent=4)

# 모델 불러오기 (사전 학습된 가중치 사용. 최사위 레이어 제거)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
# --- Dense 층 추가 및 Dropout 적용 ---
# 모델의 표현력을 높이고 과적합을 방지하기 위해 Dense 층과 Dropout을 추가합니다.
x = Dropout(0.5)(x)  # 50%의 뉴런을 랜덤하게 비활성화하여 과적합 방지
x = Dense(512, activation='relu')(x)
predictions = Dense(150, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 기본모델의 레이어 고정 (학습되지 않도록 설정)
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

earlyStopping = EarlyStopping(monitor="val_loss", patience=10, verbose=1, restore_best_weights=True)
modelCheckpoint = ModelCheckpoint(f"./models/cho_korean_food_classifier-{current_time}.keras", monitor="val_loss", verbose=1, save_best_only=True)

model.fit(train_generator,
          epochs=100,
          validation_data=validation_generator,
          steps_per_epoch=train_generator.samples // 8,  # // train_generator.batch_size,
          validation_steps=validation_generator.samples // 8,  # // validation_generator.batch_size
          callbacks=[earlyStopping, modelCheckpoint]
          )

# model.save(f'cho_korean_food_classifier-{current_time}.keras')
