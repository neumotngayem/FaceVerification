import cv2
import os
import numpy as np
from config import Config
from face_detection import FaceDetection
from face_embedding import FaceEmbedding
from sklearn.metrics.pairwise import cosine_similarity
from helper import ROOT_FACEKNOWLEDGE_DATABASE, crop_image

class FaceVerification():
    def __init__(self):
        self.db_folder = ROOT_FACEKNOWLEDGE_DATABASE
        self.face_detection = FaceDetection()
        self.face_embedding = FaceEmbedding()
        self.config = Config().parse_config()
        self.matching_threshold = float(self.config['matching_thres'])
        
    def finding_in_database(self, face_embedding):
        if os.path.isdir(self.db_folder):
            list_face_register = os.listdir(self.db_folder)
            found_person = None
            found_similarity_percentage = -1
            for database_name in list_face_register:
                # file name format {person_name}_embedding
                person_name = database_name.split('_')[0]
                cosine_similarity_list = []
                registerd_face_embedding = np.load(os.path.join(self.db_folder, database_name))
                print(f'--------- Compare with {person_name} ---------')
                for face_stored_embedding in registerd_face_embedding:
                    similarity_percentage = cosine_similarity(face_embedding.reshape(1, -1), face_stored_embedding.reshape(1, -1))
                    cosine_similarity_list.append(similarity_percentage)
                    print(f'Similarity percentage: {similarity_percentage}%')  
                avg_cosine_similarity = np.average(cosine_similarity_list)
                print(f'Average similarity percentage: {similarity_percentage}%')
                # Take the most similarity person
                if avg_cosine_similarity > self.matching_threshold and avg_cosine_similarity > found_similarity_percentage:
                    found_person = person_name
                    found_similarity_percentage = avg_cosine_similarity
                    print(" ****** Matched ******")
            return found_person, found_similarity_percentage
        print('No face registered yet!')
        return None, -1
                
    def verification(self, image_path):
        image_face = cv2.imread(image_path)
        result = self.face_detection.detect_single_image(image_face)
        if result.boxes is not None:
            image_face_crop = crop_image(result, image_face)
            print(f'Face detected on image: {image_path}')
            face_embedding = self.face_embedding.embedding_face(image_face_crop)
            person_name, matching_similarity = self.finding_in_database(face_embedding)
            return person_name, matching_similarity
        else:
            print(f'Face NOT found on image: {image_path}')
        return None, -1
