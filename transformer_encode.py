import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
from torch.autograd import Variable 
import numpy as np
import matplotlib.pyplot as plt 
import copy


class Embeddings(nn.Module):
    
    def __init__(self, d_modile, vocab):
        
        super(Embeddings, self).__init__()
        
        self.lut = nn.Embedding(vocab, d_model)
        
        self.d_model = d_model 
        
    def forward(self, x):
        
        return self.lut(x)*math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0)/d_model))
        # odd columns for sin
        pe[:, 0::2] = torch.sin(position * div_term)
        # even columns for cos 
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad = False)
        return self.dropout(x)

def subsequent_mask(size):
    attn_shape = (1, size, size)

    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # convert 0 to 1 and 1 to 0
    return torch.from_numpy(1- subsequent_mask)

# keep diagonal lower one and right 
print(np.triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], k=-1))
# keep diagonal and up right 
print(np.triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], k=0))
# keep diagonal up right only
print(np.triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], k=1))

size = 5
sm = subsequent_mask(size)
print("sm:", sm)

plt.figure(figsize=(5,5))
plt.imshow(subsequent_mask(20)[0])


def attention(query, keu, value, mask = None, dropout = None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


d_model = 512
vocab = 1000
x = Variable(torch.LongTensor([[100,2,421,508],[491,998,1,221]]))
emb = Embeddings(d_model, vocab)

embr = emb(x)
d_model = 512
dropout = 0.1
max_len = 60
x = embr 

pe = PositionalEncoding(d_model, dropout, max_len)
pe_result = pe(x)
print("pe_result:", pe_result)

query = key = value = pe_result 
mask = Variable(torch.zeros(2,4,4))
attn, p_attn = attention(query, key, value,mask=mask)
print("attn: ", attn)
print(attn.shape)
print("p_attn:", p_attn)
print(p_attn.shape)



