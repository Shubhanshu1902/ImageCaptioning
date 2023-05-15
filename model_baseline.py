import torch
import torchvision.models as models
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,embed_size):
        super(Encoder,self).__init__()
        self.inception = models.inception_v3(pretrained = True, aux_logits = True)
        self.inception.fc = nn.Linear(self.inception.fc.in_features,embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.5)
        
    def forward(self,images):
        features = self.inception(images)
        for name,param in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        return self.dropout(self.relu(features))
    
class Decoder(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers):
        super(Decoder,self).__init__()
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers)
        self.linear = nn.Linear(hidden_size,vocab_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self,feature,captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((feature.unsqueeze(0),embeddings),dim = 0)
        hiddens,_ = self.lstm(embeddings)
        return self.linear(hiddens)
    
class CNNToRNN(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers):
        super(CNNToRNN,self).__init__()
        self.encoder = Encoder(embed_size)
        self.decoder = Decoder(embed_size,hidden_size,vocab_size,num_layers)
        
    def forward(self,images,captions):
        features = self.encoder(images)
        outputs = self.decoder(features,captions)
        return outputs
    
