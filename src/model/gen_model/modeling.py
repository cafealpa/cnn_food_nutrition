from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

data_dir = "/Users/james/Desktop/dataset/21_korean/kfood_correct_model_files"

img_size = (224, 224)
batch_size = 32
seed = 123

train_datagen = ImageDataGenerator(
    rescale=1. / 255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    seed=seed,
    shuffle=True
)

valid_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    seed=seed,
    shuffle=True
)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(150, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=['accuracy'])

import datetime, json
current_time = datetime.datetime.now()
indices_json_file = f"indices-{current_time}.json"

with open(indices_json_file, "w", encoding='utf-8') as f:
    json.dump(train_generator.class_indices, f, ensure_ascii=False, indent=4)

check_point_callback = ModelCheckpoint(filepath=f'food-{current_time}.keras', monitor='val_loss', save_best_only=True)
early_stopper_callback = EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')

history = model.fit(
    train_generator,  
    steps_per_epoch = train_generator.samples // train_generator.batch_size, 
    validation_data = valid_generator,
    validation_steps = valid_generator.samples // valid_generator.batch_size,
    epochs=10,
    callbacks=[early_stopper_callback, check_point_callback],
)

print(model.summary())