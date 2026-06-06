import torch
import torch.nn as nn
import torch.nn.functional as F

with open('input.txt', 'r') as f:
    text = f.read()

tokens = sorted(list(set(text)))
vocab_size = len(tokens)

stoi = {s:i for i,s in enumerate(tokens)}
itos = {i:s for i,s in enumerate(tokens)}

encode = lambda s: [stoi[ch] if ch in tokens else 1 for ch in s]
decode = lambda l: ''.join([itos[i] for i in l])

