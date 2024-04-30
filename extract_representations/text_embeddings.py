from transformers import AutoTokenizer, AutoModel
import torch

class TextEmbeddings:
    '''
    This class is intended to extract embeddings from text models.
    It uses BERT as a default model.
    '''
    
    def __init__(self, model_name='bert-base-uncased', device='cuda', max_length=64):
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        self.device = device
        self.model.to(self.device)
        
        self.model_name = model_name
        self.max_length = max_length
        
        # eval mode
        self.model.eval()
        
    def extract(self, text):
        '''
        Extract embeddings from a text.
        
        Args:
            text (str): Text to extract embeddings from.
        
        Returns:
            torch.Tensor: Embeddings of the text.
        '''
        
        inputs = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()