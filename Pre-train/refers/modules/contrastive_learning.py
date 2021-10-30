import torch.nn as nn
from torchvision import models
from transformers import AutoModel, AutoTokenizer
from torch.nn import Linear
import pdb

class TextEncoder(nn.Module):
    """
    For the text encoder, we use the BERT base encoder offered by the `Transformers library
    <https://arxiv.org/abs/1910.03771>`_ and initialize it with the `ClinicalBERT model
    <https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT>`_
    pretrained on the MIMIC clinical notes.

    At pretraining time we freeze the embeddings and the first 6 layers of this BERT encoder,
    and only fine-tune the last 6 layers for our contrastive task.
    """
    def __init__(self, feature_size = 768):
        super(TextEncoder, self).__init__()
        self.text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.fc = Linear(feature_size, feature_size*3)

    def forward(self, x):
        # pdb.set_trace()
        outputs = self.text_encoder(**x)
        feature = outputs[1]
        feature = self.fc(feature)
        
        return feature