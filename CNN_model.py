from deepface import DeepFace
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

model = DeepFace.build_model('VGG-Face')
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

X_train, X_test, y_train, y_test = train_test_split(data['image_path'], normalized_races.tolist(), test_size=0.2, random_state=42)

image_path = X_train[0]
image = Image.open(image_path)
image = image.resize((224, 224))
image_array = np.array(image)

print("Загруженное изображение:")
print(image_array)
print("Нормализованные метки для этого изображения:")
print(y_train[0])

race_index = np.argmax(y_train[0])
race_name = race_names[race_index]
print("Предполагаемая раса для этого изображения:")
print(race_name)

from keras.applications import VGG16
from keras.models import load_model

model = VGG16(weights='imagenet')

model.save('vgg16_model.keras')

model.summary()

loaded_model = load_model('vgg16_model.keras')

