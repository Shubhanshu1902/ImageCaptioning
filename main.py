import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from fastai.vision.all import *
from fastai.text.all import *
from pathlib import Path

import torch
torch.cuda.empty_cache()
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision

import torchtext
from torchtext.data import get_tokenizer   # for tokenization
from collections import Counter     # for tokenizer

import torchvision.transforms as T
import torchvision.models as models

import matplotlib.pyplot as plt
# import matplotlib.image as Image
import PIL
from PIL import Image

# for the bleu scores
from nltk.translate import bleu
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu

# Get descriptor dictionary
def getDescriptors(captionDatafile):
    df = pd.read_csv(captionDatafile)
    print(f"No of captions: {len(df)}")

    descriptors = {}
    for i in range(len(df)):
        img_id = df.iloc[i, 0]
        sentence = ("<start> " + df.iloc[i, 1] + " <end>").split()
        if img_id not in descriptors:
            descriptors[img_id] = [sentence]
            
        else:
            descriptors[img_id].append(sentence)
            
    return descriptors

class textVocab:
    def __init__(self):
        self.indexToSentence = {0:"<PAD>", 1:"<start>", 2:"<end>", 3:"<UNK>"}
        self.sentenceToIndex = {b:a for a, b in self.indexToSentence.items()}           
        self.min_freq = 1
        self.tokenizer = get_tokenizer("basic_english")
        self.token_counter = Counter()

        
    def __len__(self):
        return len(self.indexToSentence)
    
    def tokenize(self, text):
        return self.tokenizer(text)
    
    def numericalize(self, text):
        tokens_list = self.tokenize(text)
        
        ans = []
        for token in tokens_list:
            if token in self.sentenceToIndex.keys():
                ans.append(self.sentenceToIndex[token]) 
            else:
                ans.append(self.sentenceToIndex["<UNK>"])
        return ans
    
    def build_vocab(self, sentence_list):
        word_count = 4
        
        # for each sentence
        for sentence in sentence_list:
            tokens = self.tokenizer(sentence)
            self.token_counter.update(tokens)
            
            for token in tokens:
                if self.token_counter[token] >= self.min_freq and token not in self.sentenceToIndex.keys():
                    self.sentenceToIndex[token] = word_count
                    self.indexToSentence[word_count] = token
                    word_count += 1
                    
class customDataset(Dataset):
    def __init__(self, image_dir, captions_file, transform=None, min_freq=5):
        self.image_dir = image_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform
        self.img_ids = self.df["image"]
        self.sentences = self.df["caption"]
        self.vocab = textVocab()
        self.vocab.build_vocab(self.sentences.tolist())
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        img_id = self.img_ids[idx] 
        img_path = os.path.join(self.image_dir, img_id)

        img = Image.open(img_path).convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
            
        vec = []
        
        vec += [self.vocab.sentenceToIndex["<start>"]]   # tagging
        vec += self.vocab.numericalize(sentence)  # numericalization
        vec += [self.vocab.sentenceToIndex["<end>"]]     # tagging
       
        return img, torch.tensor(vec), img_id
    
class Collate_fn:
    def __init__(self, pad_value, batch_first=False):
        self.pad_value = pad_value       
        self.batch_first = batch_first    
        
    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)    
        captions = [item[1] for item in batch]
        captions = pad_sequence(captions, batch_first = self.batch_first, padding_value = self.pad_value)
        img_ids = [item[2] for item in batch]
        return imgs, captions, img_ids    

def train_val_split(dataset,val_ratio,batch_size,pad_value = 0):
    size = len(dataset)
    indices = list(range(size))
    split = int(np.floor(val_ratio * size))
    train_indices, val_indices = indices[split:], indices[:split]
    np.random.seed(42)
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    
    dls = torch.utils.data.DataLoader(dataset, 
                                           batch_size=batch_size, shuffle=False,
                                           collate_fn = Collate_fn(pad_value=pad_value, batch_first = True),
                                           sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, shuffle=False,
                                                    batch_size=batch_size,
                                                    collate_fn = Collate_fn(pad_value=pad_value, batch_first = True),
                                                    sampler=valid_sampler)
    
    return dls,validation_loader

# create utility function to print images
def show_image(img, title=None):
    
    # unnormalize
    img[0] *= 0.229
    img[1] *= 0.224
    img[2] *= 0.225
    img[0] += 0.485
    img[1] += 0.456
    img[2] += 0.406
    
    img = img.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    if title is not None:
        plt.title(title)
        
    plt.pause(0.001)   

# function to save model
def save_model(model, num_epochs,embed_sz,dataset,enc_dim,dec_dim):
    model_state = {
        'num_epochs' : num_epochs,
        'embed_sz' : embed_sz,
        'vocab_sz' : len(dataset.vocab),
        'enc_dim' : enc_dim,
        'dec_dim' : dec_dim,
        'state_dict' : model.state_dict()
    }
    
    torch.save(model_state, 'imageCaptioning.pth')
    
def train(epochs,print_each,model,criterion,optimizer,dls,vocab_sz,validation_loader,dataset,device):
    model.train()
    for epoch in range(1, epochs+1):
        for idx, (img, captions, img_ids) in enumerate(iter(dls)):
            img, captions = img.to(device), captions.to(device)
            optimizer.zero_grad()
            pred_caps, attentions = model(img, captions)
            targets = captions[:, 1:]
            loss = criterion(pred_caps.view(-1, vocab_sz), targets.reshape(-1))
            loss.backward()    # update the NN weights
            optimizer.step()
            if (idx + 1) % print_each == 0:
                print("Epoch: {} loss: {:.5f}".format(epoch, loss.item()))
                model.eval()
                with torch.no_grad():
                    itr = iter(validation_loader)
                    img, _, _ = next(itr)
                    features = model.encoder(img[0:1].to(device))
                    pred_caps, alphas = model.decoder.generate_caption(features, vocab=dataset.vocab)
                    caption = ' '.join(pred_caps)
                    print(caption)
                    show_image(img[0])
                model.train()
        
        save_model(model, epoch)
        