import os
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import Normalizer
from imgaug import augmenters as iaa
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import random

# Initialize FaceNet model and Normalizer
MyFaceNet = FaceNet()
l2_normalizer = Normalizer('l2')

# Augmentation function
def augment_image(image):
    augmenter = iaa.Sequential([
        iaa.Affine(rotate=(-10, 10)),  # Small rotations
        iaa.Fliplr(0.5),               # Horizontal flips
        iaa.Multiply((0.9, 1.1)),      # Brightness adjustments
    ])
    return augmenter.augment_image(image)

# Extract face using MTCNN
def extract_face(image, required_size=(160, 160)):
    detector = MTCNN()
    results = detector.detect_faces(image)
    if results:
        x, y, width, height = results[0]['box']
        x, y = max(0, x), max(0, y)
        face = image[y:y+height, x:x+width]
        face = cv2.resize(face, required_size)
        return face
    return None

# Get embedding using FaceNet
def get_embedding(face):
    face = face.astype('float32')
    face = np.expand_dims(face, axis=0)
    embedding = MyFaceNet.embeddings(face)
    return l2_normalizer.transform(embedding)[0]

# Compute average embedding
def compute_average_embedding(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face = extract_face(image)
    if face is None:
        print(f"No face detected in image: {image_path}")
        return None

    # Generate augmented images only if a valid face is detected
    augmented_images = [augment_image(face) for _ in range(10)]
    embeddings = [get_embedding(img) for img in augmented_images]
    return np.mean(embeddings, axis=0)

# Load images and create embeddings
def load_and_encode_images(folder_path, label):
    embeddings = []
    for file in os.listdir(folder_path):
        if file.endswith(('jpg', 'jpeg', 'png')):
            person_name = os.path.splitext(file)[0]
            print(f"Processing {file} as {label}")
            image_path = os.path.join(folder_path, file)
            embedding = compute_average_embedding(image_path)
            if embedding is None:
                continue
            embeddings.append({'name': person_name, 'label': label, 'embedding': embedding})
    return embeddings

# Recognize face
def recognize_face(embedding, database, threshold=0.3):
    max_similarity = -1
    identified = {'name': 'x', 'label': 'x'}
    for record in database:
        db_embedding = record['embedding']
        similarity = cosine_similarity([embedding], [db_embedding])[0][0]
        if similarity > max_similarity and similarity > threshold:
            max_similarity = similarity
            identified = {'name': record['name'], 'label': record['label']}
    return identified

# Main
if __name__ == "__main__":
    students_folder = r'Data Base\students'
    observers_folder = r'Data Base\observers'

    # Load and encode images from both folders (training phase)
    students_data = load_and_encode_images(students_folder, 'S')
    observers_data = load_and_encode_images(observers_folder, 'O')

    # Combine data into a single database
    database = students_data + observers_data

    # Create models directory if it doesn't exist
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, 'finalFace.pkl')

    # Save the database to a file
    with open(model_path, 'wb') as f:
        pickle.dump(database, f)

    print("Training complete. Database saved.")