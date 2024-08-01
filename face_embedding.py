from config import Config
from models_def.embedding_net import EmbeddingNet
from models_def.triplet_net import TripletNet
import torch
from torchvision import transforms
from torchvision.transforms import v2

class FaceEmbedding():
    def __init__(self):
        embedding_net = EmbeddingNet()
        self.embedding_net = TripletNet(embedding_net)
        self.config = Config().parse_config()
        model_path = self.config['embedding_model']
        # load model weight
        self.embedding_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu') ))
        # state model for inference mode
        self.embedding_net.eval()
        # Transform image from BRG to RGB, cast to float type and resize the image
        self.transform = transforms.Compose([v2.ToTensor(), v2.Lambda(lambda image: image[[2, 1, 0]]), v2.ToDtype(torch.float32, scale=True), v2.Resize((128,128))])
        
    def embedding_face(self, image_face_crop):
        with torch.no_grad():
            data_trans = self.transform(image_face_crop).unsqueeze(0)
            # embedding model inference
            face_embedding = self.embedding_net.get_embedding(data_trans).data.cpu().numpy()[0]
        return face_embedding  