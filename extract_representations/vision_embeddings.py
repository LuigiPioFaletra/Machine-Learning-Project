from transformers import ViTFeatureExtractor, ViTModel
import torch

class VisionEmbeddings:
    '''
    This class is intended to extract embeddings from vision models.
    It uses ViT (Vision Transformer) as a default model.
    '''
    
    def __init__(self, model_name='google/vit-base-patch16-224', device='cuda'):
        
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        
        self.device = device
        self.model.to(self.device)
        
        self.model_name = model_name
        
        # eval mode
        self.model.eval()
        
    def extract(self, image):
        '''
        Extract embeddings from an image.
        
        Args:
            image (PIL.Image): Image to extract embeddings from.
        
        Returns:
            torch.Tensor: Embeddings of the image.
        '''
        
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()