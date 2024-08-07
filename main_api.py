import base64
from fastapi import FastAPI, Form, UploadFile
from config import Config
import os
import time
import shutil
from face_knowledge_formation import FaceKnowledgeFormation
from face_verification import FaceVerification
from helper import DETECTION_OUTPUT_FILENAME
from fastapi.middleware.cors import CORSMiddleware
config = Config().parse_config()
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/register")
async def face_register(files: list[UploadFile], person_name: str = Form()):
    try:
        #save to temp folder
        temporal_folder = os.path.join(config['temp_uploading_folder'], str(float(time.time())))
        os.makedirs(temporal_folder, exist_ok=True)
        for file in files:
            file_path = os.path.join(temporal_folder, file.filename)
            with open(file_path, "wb") as f:
                f.write(file.file.read())
        raw_images_fname = os.listdir(temporal_folder)
        image_paths = [os.path.join(temporal_folder, fname) for fname in raw_images_fname]
        print('--------- Register face ---------')
        print(f'Person name: {person_name} - face image paths: {image_paths}')
        faceknowledge_formation = FaceKnowledgeFormation(person_name)
        faceknowledge_formation.create_faceknowledge_database(image_paths)
        shutil.rmtree(temporal_folder)
        return {"message": "Face registered successfully"}
    except Exception as e:
       return {"message": e.args}
    
@app.post("/verification")
async def verification(file: UploadFile):
    try:
        request_time = float(time.time())
        temporal_folder = os.path.join(config['temp_uploading_folder'], str(request_time))
        os.makedirs(temporal_folder, exist_ok=True)
        file_path = os.path.join(temporal_folder, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        print('--------- Face verification ---------')
        face_verification = FaceVerification(request_time)
        found_person, found_similarity_percentage = face_verification.verification(file_path)
        detection_output_path = os.path.join(temporal_folder, DETECTION_OUTPUT_FILENAME)
        with open(detection_output_path, "rb") as imagefile:
            base64_detection_image = base64.b64encode(imagefile.read())
        return_data = {"person_name": found_person, "similarity_percentage": found_similarity_percentage, "base64_detection_image": base64_detection_image}
        shutil.rmtree(temporal_folder)
        return {"message": "Face verification successfully", "data": return_data}
    except Exception as e:
        return {"message": e.args}