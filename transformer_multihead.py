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



def attention(query, key, value, mask = None, dropout = None):
    d_k = query.size(-1)
    #
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim = -1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()

        assert embedding_dim % head == 0

        self.d_k = embedding_dim // head 

        self.head = head 

        self.embedding_dim = embedding_dim

        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)

        self.attn = None 

        self.dropout = nn.Dropout(p = dropout)

    def forward(self, query, key, value, mask = None):

        if mask is not None:

            mask = mask.unsqueeze(1)

        batch_size = query.size(0)

        query, key, value = [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1,2)
                             for model, x in zip(self.linears, (query, key, value))]
        
        x, self.attn = attention(query, key, value, mask=mask, dropout = self.dropout)
        x = x.transpose(1,2).contiguous().view(batch_size, -1, self.head * self.d_k)
        return self.linears[-1](x)
    
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

head = 8
embedding_dim = 512
dropout = 0.2

query = key = value = pe_result 
mask = Variable(torch.zeros(2,4,4))

mha = MultiHeadedAttention(head, embedding_dim, dropout)
mha_result = mha(query, key, value, mask)
print(mha_result)
print(mha_result.shape)




x = torch.randn(4,4)
print(x.size())
y = x.view(16)
print(y.size())
z = x.view(-1,8)
print(z.size())

a = torch.randn(1,2,3,4)
print(a.size())
# swap the 2nd and 3rd dimension
b = a.transpose(1,2)
print(b.size())
c = a.view(1,3,2,4)
print(c.size())
print(torch.equal(b,c))


x = torch.randn(2,3)
print(x)
print(torch.transpose(x, 0,1))

