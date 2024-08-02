import argparse
import os
from face_knowledge_formation import FaceKnowledgeFormation
from face_verification import FaceVerification
from helper import CASE_REGISTER, CASE_VERIFICATION


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='FaceVerficationSystem',
                        description='Register the face or verifcation the face with the registered faces')
    parser.add_argument('-t', '--type')      # Type of function register: r, verifcation: v
    parser.add_argument('-p', '--path')      # Path to the images folder in case register, path to image in case verifcation
    parser.add_argument('-n', '--name')      # Name of register person
    args = parser.parse_args()
    action_type = args.type
    path = args.path
    if action_type == CASE_REGISTER:
        person_name = args.name
        raw_images_fname = os.listdir(path)
        image_paths = [os.path.join(path, fname) for fname in raw_images_fname]
        print('--------- Register face ---------')
        print(f'Person name: {person_name} - face image paths: {image_paths}')
        faceknowledge_formation = FaceKnowledgeFormation(person_name)
        faceknowledge_formation.create_faceknowledge_database(image_paths)
        print('Register sucessuflly!')
    if action_type == CASE_VERIFICATION:
        print('--------- Face verification ---------')
        face_verification = FaceVerification()
        found_person, found_similarity_percentage = face_verification.verification(path)
        if found_person:
            print(f'Holla: {found_person} !')
        else:
            print("Sorry, we can't verify you yet!")