# Formation face's knowledge database of a person
import os
import cv2
import numpy as np
from helper import ROOT_FACEKNOWLEDGE_DATABASE, crop_image
from face_detection import FaceDetection
from face_embedding import FaceEmbedding

class FaceKnowledgeFormation():
    
    def __init__(self, person_name):
        self.person_name = person_name
        self.face_detection = FaceDetection()
        self.face_embedding = FaceEmbedding()
        self.db_folder = ROOT_FACEKNOWLEDGE_DATABASE
        os.makedirs(self.db_folder, exist_ok=True)
        
    def create_faceknowledge_database(self, image_paths):
        person_face_knowledge = []
        # detection face on the images
        face_detection_results = self.face_detection.detect_multiple_images(image_paths)
        for idx, result in enumerate(face_detection_results):
            # Load the face image correspond with the face detection result
            image_face = cv2.imread(image_paths[idx])
            # if face detected on the image
            if result.boxes is not None:
                image_face_crop = crop_image(result, image_face)
                print(f'Face detected on image: {image_paths[idx]}')
                face_embedding = self.face_embedding.embedding_face(image_face_crop)
                person_face_knowledge.append(face_embedding)
            else:
                print(f'Face NOT found on image: {image_paths[idx]}')
        print(f'Extracted total: {len(person_face_knowledge)} face knowledge of this person.')
        # save the face embedding of that person
        np.save(f'{ROOT_FACEKNOWLEDGE_DATABASE}/{self.person_name}_embedding', person_face_knowledge)