import torch
import torch.nn as nn
import urllib.request
from torch.nn import functional as F

#hyperparameters
batch_size = 64 #independent sequences processed in parallel
block_size = 256 #maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384 #embedding dimension
n_head = 6 #number of attention heads
n_layer = 6 #number of layers
dropout = 0.2

torch.manual_seed(1337)
# Fetching Sample data
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
local_filename = "input.txt"
urllib.request.urlretrieve(url, local_filename)

# Reading the whole file in a go
with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()


char = sorted(list(set(text))) #All unique characters in the text
vocab_size = len(char)
stoi = {ch:i for i,ch in enumerate(char)} #Creating mapping from char to int
itos = {i:ch for i,ch in enumerate(char)}
encode = lambda s: [stoi[c] for c in s] #Take a string and return list of integers
decode = lambda l: ''.join([itos[i] for i in l]) #Take a list of integers and return string

#Train and Test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) #Splitting data
train_data = data[:n] #90% data is for training
val_data = data[n:] #10% is for validation

#loading data in batches
def get_batch(split):
  #generate a small batch of data of inputs x and targets y
  data = train_data if split=='train' else val_data
  ix = torch.randint(len(data)-block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x,y

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

class Head(nn.Module):
  """ one head of self-attention """
  
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x)   #(B,T,head_size)
    q = self.query(x) #(B,T,head_size)
    #comping the attention scores 
    wei = q @ k.transpose(-2,-1) * C**-0.5 #(B,T,T)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1) #(B,T,T)
    wei = self.dropout(wei)
    #performing the weighted aggregation of the values
    v = self.value(x) #(B,T,head_size)
    out = wei @ v #(B, T, T) @ (B, T, head_size) --> (B, T, head_size)
    return out

class MultiHeadAttention(nn.Module):
  """ multiple heads of self-attention in parallel """
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.proj(out)
    return out

class FeedForward(nn.Module):
  """ a simple linear layer followed by a non-linearity """
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embd, 4 * n_embd),
        nn.ReLU(),
        nn.Linear(4 * n_embd, n_embd),
        nn.Dropout(dropout),
    )

  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  """ Transformer block: communication followed by computation """
  def __init__(self, n_embd, n_head):
    #n_embd: embedding dimension, n_head: number of heads we'd like
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)
  
  def forward(self, x):
    x = x + self.sa(self.ln1(x)) # Applying one head attention
    x = x + self.ffwd(self.ln2(x)) # Applying feed-forward layer
    return x

class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.positional_embedding_table = nn.Embedding(block_size, n_embd)
    # self.sa_head = MultiHeadAttention(4, n_embd//4) #4 heads of 8-dimension self-attention
    # self.ffwd = FeedForward(n_embd)
    # self.blocks = nn.Sequential(
    #     Block(n_embd, 4),
    #     Block(n_embd, 4),
    #     Block(n_embd, 4),
    #     nn.LayerNorm(n_embd),
    # )
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
    self.lm_head = nn.Linear(n_embd, vocab_size)
  
  def forward(self, idx, targets=None):
    B, T = idx.shape
    token_embd = self.token_embedding_table(idx) #(B, T, C) Batch, Time, Chanel
    pos_embd = self.positional_embedding_table(torch.arange(T, device=device)) #(T,C)
    x = token_embd + pos_embd #Batch, Time, Chanel
    # x = self.sa_head(x) # Applying one head attention (Batch, Time, Chanel)
    # x = self.ffwd(x) # Applying feed-forward layer (Batch, Time, Chanel)
    x = self.blocks(x)
    logits = self.lm_head(x) #Batch, Time, Vocab_size
    
    if targets is None:
        loss = None
    else:
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
    return logits, loss

  def generate(self, idx, max_new_tokens): 
    #idx is B,T array of indices in current context
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:] #crop to the last block_size tokens
      logits, loss = self(idx_cond) #get predictions
      logits = logits[:, -1, :] # focus only on last time step becomes B,C
      probs = F.softmax(logits, dim=-1) #applying softmax to get probability
      idx_next = torch.multinomial(probs, num_samples=1) #sample from distribution
      idx = torch.cat((idx, idx_next), dim=1) #append sample index to running sequence
    return idx
  
model = BigramLanguageModel()
model = model.to(device)

#printing the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

#creating a Pytorch optimzer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    #sampling a batch of data
    xb, yb = get_batch('train')
    
    #evaluating the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#generating from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))