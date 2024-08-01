from config import Config
from ultralytics import YOLO

class FaceDetection():
    def __init__(self):
        self.config = Config().parse_config()
        model_path = self.config['face_detection_model']
        #Loading face detection model
        self.face_detection_model = YOLO(model_path)
        
    def detect_multiple_images(self, image_paths):
        return self.face_detection_model(image_paths)
    
    def detect_single_image(self, image_path):
        return self.face_detection_model(image_path)[0]