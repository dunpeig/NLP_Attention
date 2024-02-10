import torch
import torch.nn as nn
import math 
from torch.autograd import Variable 
import numpy as np

class Embeddings(nn.Module):
    
    def __init__(self, d_modile, vocab):
        
        super(Embeddings, self).__init__()
        
        self.lut = nn.Embedding(vocab, d_model)
        
        self.d_model = d_model 
        
    def forward(self, x):
        
        return self.lut(x)*math.sqrt(self.d_model)

embedding = nn.Embedding(10, 3)
input = torch.LongTensor([[1,2,3,4],[4,3,2,9]])
print(embedding(input))

embedding = nn.Embedding(10, 3, padding_idx = 0)
input = torch.LongTensor([[0,2,0,5]])
print(embedding(input))

d_model = 512
vocab = 1000
x = Variable(torch.LongTensor([[100,2,421,508],[491,998,1,221]]))
emb = Embeddings(d_model, vocab)
embr = emb(x)
print('embr:', embr)
    
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

m = nn.Dropout(p=0.2)
input1 = torch.randn(4,5)
output = m(input1)
output 

x = torch.tensor([1,2,3,4])
y = torch.unsqueeze(x, 0)
print(y)
z = torch.unsqueeze(x,1)
print(z)

d_model = 512
dropout = 0.1
max_len = 60

x = embr 
pe = PositionalEncoding(d_model, dropout, max_len)
pe_result = pe(x)
print("pe_result:", pe_result)


import matplotlib.pyplot as plt 
plt.figure(figsize=(15,5))
pe = PositionalEncoding(20, 0)
y = pe(Variable(torch.zeros(1, 100, 20)))
plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
plt.legend(["dim %d"%p for p in [4,5,6,7]])



