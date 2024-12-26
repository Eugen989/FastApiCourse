import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from keras.applications import VGG16

images_folder = './deepface/data/images/'
labels_folder = './deepface/data/labels/'

image_files = sorted([f for f in os.listdir(images_folder) if f.endswith('.jpg')])
data = {'image_path': [], 'race': []}

race_names = {
    0: "White",
    1: "Black",
    2: "Asian",
    3: "Indian",
    4: "Others"
}

for image_file in image_files:
    image_number = os.path.splitext(image_file)[0]
    label_file = os.path.join(labels_folder, f'{image_number}.txt')

    try:
        with open(label_file, 'r') as file:
            race = file.read().strip()
            data['image_path'].append(os.path.join(images_folder, image_file))
            data['race'].append(race)
    except FileNotFoundError:
        print(f'Файл меток для {image_file} не найден.')

parsed_races = []
for race in data['race']:
    race_values = list(map(float, race.split()))
    parsed_races.append(race_values)

data['race'] = parsed_races

scaler = MinMaxScaler()
normalized_races = scaler.fit_transform(data['race'])

y_categorical = to_categorical(np.argmax(normalized_races, axis=1), num_classes=len(race_names))

X_train, X_test, y_train, y_test = train_test_split(data['image_path'], y_categorical, test_size=0.2, random_state=42)

def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return image_array

X_train_images = np.array([load_and_preprocess_image(img) for img in X_train])
X_test_images = np.array([load_and_preprocess_image(img) for img in X_test])

X_train_split, X_val, y_train_split, y_val = train_test_split(X_train_images, y_train, test_size=0.2, random_state=42)

model = Sequential()
model.add(Flatten(input_shape=(224, 224, 3)))
model.add(Dense(128, activation='relu'))
model.add(Dense(len(race_names), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train_split, y_train_split, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

test_loss, test_accuracy = model.evaluate(X_test_images, y_test)
print(f"Тестовая точность: {test_accuracy:.4f}")

image_path = X_test[0]
image_array = load_and_preprocess_image(image_path)

predicted_class = model.predict(np.expand_dims(image_array, axis=0))
race_index = np.argmax(predicted_class)

print("Предполагаемая раса для тестового изображения:")
print(race_names[race_index])

model = VGG16(weights='imagenet')

model.save('MyModel.keras')

model.summary()
