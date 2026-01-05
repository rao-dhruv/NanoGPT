import torch
import torch.nn as nn
import urllib.request
from torch.nn import functional as F

#hyperparameters
batch_size = 32 #independent sequences processed in parallel
block_size = 8 #maximum context length for predictions
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

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

def get_batch(split):
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

class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
  
  def forward(self, idx, targets=None):
    logits = self.token_embedding_table(idx) #Batch, Time, Chanel
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
      logits, loss = self(idx) #get predictions
      logits = logits[:, -1, :] # focus only on last time step becomes B,C
      probs = F.softmax(logits, dim=-1) #applying softmax to get probability
      idx_next = torch.multinomial(probs, num_samples=1) #sample from distribution
      idx = torch.cat((idx, idx_next), dim=1) #append sample index to running sequence
    return idx
  
model = BigramLanguageModel(vocab_size)
model = model.to(device)

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