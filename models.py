import torch

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


