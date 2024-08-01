# Formation face's knowledge database of a person
import os
import cv2
import numpy as np
from helper import ROOT_FACEKNOWLEDGE_DATABASE
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
                # bounding boxes coordinate (left, top, right, bottom)
                boxes = result.boxes.xyxy
                # only crop the largest face area
                final_box = boxes[0]
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    area_box = (x2 - x1) * (y2 - y1)
                    area_final_box = (final_box[2] - final_box[0]) * (final_box[3] - final_box[1])
                    # compare the bounding box area
                    if area_box > area_final_box:
                        final_box = box
                # crop the image with padding
                image_face_crop = image_face[int(final_box[1] - 50):int(final_box[3] + 25), int(final_box[0] - 50):int(final_box[2] + 50)]
                print(f'Face detected on image: {image_paths[idx]}')
                face_embedding = self.face_embedding.embedding_face(image_face_crop)
                person_face_knowledge.append(face_embedding)
            else:
                print(f'Face NOT found on image: {image_paths[idx]}')
        print(f'Extracted total: {len(person_face_knowledge)} face knowledge of this person.')
        # save the face embedding of that person
        np.save(f'{ROOT_FACEKNOWLEDGE_DATABASE}/{self.person_name}_embedding', person_face_knowledge)