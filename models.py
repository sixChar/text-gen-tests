import torch
import math
import torch.nn.functional as F
import numpy as np



class LSTMModel(torch.nn.Module):
  def __init__(self):
    super(LSTMModel, self).__init__()
    
    self.linear_in = torch.nn.Linear(128, 256)
    self.lstm = torch.nn.LSTM(256, 256, 2)
    self.linear_out = torch.nn.Linear(256, 128)

  def forward(self, x):
    x = F.relu(self.linear_in(x))
    x, (last_hid, last_cell) = self.lstm(x)
    x = F.softmax(self.linear_out(x), dim=-1)

    return x


'''
  Not exactly a full transformer as in the "Attention Is All You Need" paper but more something
  just to use the linear attention 
'''
class LinTransformerModel(torch.nn.Module):
  def __init__(self, d_io=128, d_model=256, num_layers=3):
    super(LinTransformerModel, self).__init__()
    
    # input layer
    self.in_ff = torch.nn.Linear(d_io, d_model)

    self.pe = PositionalEncoding(d_model)

    self.trans_layers = []

    for i in range(num_layers):
      attn = LinearAttention(d_model)
      hid1 = torch.nn.Linear(d_model, d_model)
      hid2 = torch.nn.Linear(d_model, d_model)
      self.trans_layers.append((attn, hid1, hid2))

    #output layer
    self.out_ff = torch.nn.Linear(d_model, d_io)

  def forward(self, x):
    x = F.relu(self.in_ff(x))
    x = self.pe(x)

    for (attn, hid1, hid2) in self.trans_layers:
      x_t = attn(x) + x
      x_t = F.relu(hid1(x_t))
      x_t = hid2(x_t)
      x = x_t + x

    return F.softmax(self.out_ff(x), dim=-1)


'''
  Linear attention ... I think. The paper does a very bad job explaining where some of the transposes
  come from and how the dimesnions are.
'''
class LinearAttention(torch.nn.Module):
  def __init__(self, d_model, d_key=None):
    super(LinearAttention, self).__init__()

    if not d_key:
      d_key = d_model
    
    self.W_k = torch.nn.Linear(d_model, d_key, bias=False)
    self.W_q = torch.nn.Linear(d_model, d_key, bias=False)
    self.W_v = torch.nn.Linear(d_model, d_model, bias=False)
    self.d_key = d_key
    self.d_model = d_model


  def forward(self, x):
    Q = self.kernel_func(self.W_q(x))
    K = self.kernel_func(self.W_k(x))
    V = self.W_v(x)

    s = torch.zeros(self.d_key, self.d_model)
    # batch_size x d_key
    z = torch.zeros(x.shape[1], self.d_key)
   
    y = []

    for i in range(x.shape[0]):
      KtV = torch.transpose(K[i,:,:], 0, 1).matmul(V[i,:,:])
      s = s.add(KtV)
      z = z.add(K[i,:,:])
      
      # numer : batch x d_model
      numer = Q[i,:,:].matmul(s)

      # Q_i : batch x 1 x d_key
      Q_i = Q[i,:,:].unsqueeze(1)

      # z_i : batch x d_key x 1
      z_i = z.unsqueeze(2)
    
      # denom : batch
      denom = Q_i.bmm(z_i).squeeze(2)
      y.append((numer/denom).unsqueeze(0))
    return torch.cat(y, dim=0)
      

  def kernel_func(self, x):
    return F.elu(x) + 1

  
'''
  Positional encoding layer
'''
class PositionalEncoding(torch.nn.Module):
  def __init__(self, d_model, max_len=5000):
    super(PositionalEncoding, self).__init__()
    
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)

  def forward(self, x):
    return x + self.pe[:x.size(0)]


'''
  Convert a string to a numpy array of tokens.
  
  token_func: function to convert chars into non-negative integers
  num_chars: number of integers in the range of token_func
'''
def text_to_tokens(text, token_func=ord, num_chars=128):
  tokens = []
  for ch in text:
    token = np.zeros((1,num_chars))
    if token_func(ch) < num_chars:
      token[:, token_func(ch)] = 1
    else:
      token[:, token_func('_')]
    tokens.append(token)
  return np.concatenate(tokens, axis=0).astype(np.float32)
    

'''
  Convert a numpy array of tokens to a string.
  inv_token_func: inverse of the function used to generate tokens, converts tokens to chars.
'''
def tokens_to_text(tokens, inv_token_func=chr):
  indexs = np.argmax(tokens, axis=-1)
  chs = []
  for i in range(indexs.shape[0]):
    chs.append(inv_token_func(indexs[i]))
  return ''.join(chs)



'''
  Use a model to generate num_chars characters using start_text as the initial input
'''
def generate_text(model, start_text, num_chars):
  tokens = np.expand_dims(text_to_tokens(start_text), axis=1)
  for i in range(num_chars):
    with torch.no_grad():
      x = torch.from_numpy(tokens)
      y = model.forward(x)
      last_prob = y[-1,0,:].numpy()

    next_token_index = np.random.choice(np.arange(tokens.shape[-1]), p=last_prob)
    next_token = np.zeros((1, 1, tokens.shape[-1]), np.float32)
    next_token[:, :, next_token_index] = 1
    tokens = np.concatenate([tokens, next_token], axis=0)
  return tokens_to_text(tokens[:,0,:])


if __name__=="__main__":
  model = LSTMModel()

  test_x = torch.rand(1,20,128)
  test_y = model.forward(test_x)

  print(generate_text(model, 'Today ', 256))


