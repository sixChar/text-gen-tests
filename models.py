import torch
import math
import torch.nn.functional as F
import numpy as np






'''
  A fast weight programmer model which uses an lstm slow net and a feed forward fast net.
'''
class LSTM_FF_FWPModel(torch.nn.Module):
  def __init__(self, num_ins, num_outs):
    super().__init__()
    slow_model_gen = lambda ins, outs: LSTMSlowNet(ins, outs)
    fast_model_gen = lambda ins, outs: FFFastNet(ins, outs)

    self.model = FWPModel(slow_model_gen, fast_model_gen, num_ins, num_outs)

  def parameters(self):
    return self.model.parameters()

  def forward(self, x):
    return self.model(x)


'''
  A fast weight programmer model which uses a feed forward network for both the slow and fast nets.
'''
class FF_FF_FWPModel(torch.nn.Module):
  def __init__(self, num_ins, num_outs):
    super().__init__()
    slow_model_gen = lambda ins, outs: FFSlowNet(ins, outs)
    fast_model_gen = lambda ins, outs: FFFastNet(ins, outs)

    self.model = FWPModel(slow_model_gen, fast_model_gen, num_ins, num_outs)

  def parameters(self):
    return self.model.parameters()

  def forward(self, x):
    return self.model(x)
 

'''
  Fast Weight Programmer model, uses a slow net to update the weights of a fast net that produces output.

  __init__:
    @slow_model_generator: function that returns a pytorch module that maps inputs to some vectors
                           which will be used to update the weights of the fast model.
    @fast_model_generator: function that returns a model which maps the inputs to ouputs and receives
                           weight updates from the slow model.
'''
class FWPModel(torch.nn.Module):
  def __init__(self, slow_model_generator, fast_model_generator, num_ins, num_outs): 
    super().__init__()
    self.fast_model = fast_model_generator(num_ins, num_outs)
    self.num_fast_params = self.fast_model.get_num_params()


    self.update_vec_size = math.ceil(math.sqrt(self.num_fast_params))
    # The slow net needs to output enough values that they can be split into 4 vectors, two pairs, whose 
    # outer product is the size of the number of elements in the fast models parameters as well as an 
    # update strength parameter for both the positive and negative updates.
    slow_net_outs =  4 * self.update_vec_size + 2

    self.slow_model = slow_model_generator(num_ins, slow_net_outs)


  def parameters(self):
    return set(self.slow_model.parameters()) | set(self.fast_model.parameters())


  '''
    x: tensor of shape (seq length, batch size, num ins)
  '''
  def forward(self, x):
    
    weight_updates = self.slow_model(x)
    outs = []
    self.fast_model.clear_weight_updates()
    for i in range(x.shape[0]):
      update_i = weight_updates[i]
      
      # The vectors whose outer product form the add update  
      add_upd_1 = update_i[:, :self.update_vec_size]
      add_upd_2 = update_i[:, self.update_vec_size: 2*self.update_vec_size]
      
      # The vectors whose outer product form the sub update
      sub_upd_1 = update_i[:, 2*self.update_vec_size: 3*self.update_vec_size]
      sub_upd_2 = update_i[:, 3*self.update_vec_size: 4*self.update_vec_size]

      # The strength factor to multiply each respective update by
      add_upd_strength = torch.sigmoid(update_i[:, 4*self.update_vec_size]).unsqueeze(1)
      sub_upd_strength = torch.sigmoid(update_i[:, 4*self.update_vec_size + 1]).unsqueeze(1)
      
      # The full add and sub weight updates flattened and cut to size
      add_update = torch.transpose(add_upd_1, 0, 1).matmul(add_upd_2).reshape((1,-1))[:,:self.num_fast_params] \
                    * add_upd_strength
      sub_update = torch.transpose(sub_upd_1, 0, 1).matmul(sub_upd_2).reshape(1,-1)[:,:self.num_fast_params] \
                    * sub_upd_strength
      self.fast_model.add_weight_update(add_update - sub_update)

      outs.append(self.fast_model(x[i]).unsqueeze(0))
    return torch.cat(outs,dim=0)




'''
  LSTM slow net for fast weight programmer (basically just a regular lstm model but used differently so 
  it's seperated)
'''
class LSTMSlowNet(torch.nn.Module):
  def __init__(self, num_ins, num_outs, num_hid=256):
    super().__init__()
    self.in_layer = torch.nn.Linear(num_ins, num_hid)

    self.lstm = torch.nn.LSTM(num_hid, num_hid, num_layers=2)
    
    self.out_layer = torch.nn.Linear(num_hid, num_outs)


  def forward(self, x):
    x = F.relu(self.in_layer(x))
    x, _ = self.lstm(x)
    return self.out_layer(x)
    

'''
  Basic feed forward slow net for fast weight programmer
'''
class FFSlowNet(torch.nn.Module):
  def __init__(self, num_ins, num_outs, hid_shapes=(256,512,1024)):
    super().__init__()
    self.layers = [torch.nn.Linear(num_ins, hid_shapes[0])]
    for hid1,hid2 in zip(hid_shapes[:-1], hid_shapes[1:]):
      self.layers.append(torch.nn.Linear(hid1, hid2))

    self.layers.append(torch.nn.Linear(hid_shapes[-1], num_outs))
    

  def forward(self, x):
    for layer in self.layers[:-1]:
      x = F.relu(layer(x))
    return self.layers[-1](x)



'''
  Basic feed forward fast net for fast weight programmer
'''
class FFFastNet(torch.nn.Module):
  def __init__(self, num_ins, num_outs, hid_shapes=(128,128,128,128)):
    super().__init__()
    self.layers = []
    self.num_params = 0
    for layer_in,layer_out in zip((num_ins,) + hid_shapes, hid_shapes + (num_outs,)):
      self.layers.append(FWPFastLinear(layer_in, layer_out))
      self.num_params += self.layers[-1].get_num_params()
    

  def forward(self, x):
    for layer in self.layers[:-1]:
      x = F.relu(layer(x)) 
    return F.softmax(self.layers[-1](x), dim=-1)

  def parameters(self):
    return set.union(*(set(layer.parameters()) for layer in self.layers))
  
  def add_weight_update(self, update):
    start = 0
    for layer in self.layers:
      layer.add_weight_update(update[:, start:start+layer.get_num_params()])
      start += layer.get_num_params()
  
  def clear_weight_updates(self):
    for layer in self.layers:
      layer.clear_weight_updates()

  def get_num_params(self):
    return self.num_params

'''
  Linear layer for a fast weight programmer fast net
'''
class FWPFastLinear(torch.nn.Module): 
  def __init__(self, num_ins, num_outs):
    super().__init__()
    self.register_parameter('W0', torch.nn.Parameter(torch.rand(num_ins, num_outs)-0.5))
    self.register_parameter('b0', torch.nn.Parameter(torch.rand(num_outs)-0.5))

    self.dW = torch.zeros_like(self.W0)
    self.db = torch.zeros_like(self.b0)

    self.num_params = num_ins * num_outs + num_outs

  def forward(self, x):
    W = self.W0 + self.dW
    b = self.b0 + self.db

    return x.matmul(W) + b

  def add_weight_update(self, update):
    if update.size()[0] != 1:
      raise ValueError("Only batch size of 1 supported for FWPFastLinear")

    dw_ = update[:, :self.dW.numel()].reshape(self.W0.size())
   
    self.dW += dw_ 
    self.db += update[:, self.dW.numel(): self.dW.numel() + self.db.numel()].reshape(self.b0.size())


  def clear_weight_updates(self):
    self.dW = torch.zeros_like(self.dW)
    self.db = torch.zeros_like(self.db)

  def get_num_params(self):
    return self.num_params


'''
  Calculates the number of parameters used by a network with the matrices whose shapes are listed in mat_
  shapes.
'''
def calc_params(mat_shapes, use_bias=True):
  num_params = sum(s1 * s2 for (s1,s2) in mat_shapes)

  if use_bias:
    num_params += sum(shape[1] for shape in mat_shapes)
  return num_params



class LSTMModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    
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
    super().__init__()
    
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
    super().__init__()

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
    super().__init__()
    
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
  test_x = torch.rand(10,1,4) 

  test_y = torch.rand(10,1,4) 
  
  slow_gen = lambda ins, outs: FFSlowNet(ins,outs) 
  fast_gen = lambda ins, outs: FFFastNet(ins,outs)

  model = FWPModel(slow_gen, fast_gen, 4, 4) 
  
  y_ = model(test_x)
  print(y_)

  loss = y_.square().mean()

  loss.backward()





