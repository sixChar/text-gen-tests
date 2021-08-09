from dotenv import  dotenv_values
from models import *
import numpy as np





DATA_PATH = dotenv_values(".env")['BLOG_DATA_PATH']
TEST_TEXT = 'Today '


def find_nth(s, find_ch, n):
  ch_count = 0
  for i, ch in enumerate(s):
    if ch == find_ch:
      ch_count += 1
    if ch_count == n:
      return i
  return -1


def load_data(text_only=False):
  texts = []
  heads = []
  with open(DATA_PATH, 'r') as fp:
    next(fp) # Skip header which looks like: (id, gender, age, topic, sign, date, text)
    for line in fp:
      # 6th comma seperates the id, gender, etc. from the text
      text_start = find_nth(line, ',', 6)
      
      header_data = line[:text_start].split(',')
      text_data = line[text_start + 1:]

      texts.append(text_data)
      heads.append(header_data)
  if text_only:
    return texts
  else:
    return heads, texts


def get_batch(texts, seq_length, batch_size=8, batch_axis=1):
  batch_xs = []
  batch_ys = []
  for i in range(batch_size):
    text = texts[np.random.randint(0, len(texts))]
    tokens = text_to_tokens(text)
   
    # pad tokens so there are seq_length + 1 of them 
    if tokens.shape[0] < seq_length + 1:
      tokens = np.pad(tokens, ((seq_length + 1 - tokens.shape[0],0), (0,0)))
    elif tokens.shape[0] > seq_length + 1:
      start = np.random.randint(0, tokens.shape[0] - (seq_length+1))
      tokens = tokens[start: start+seq_length+1, :]
    
    tokens_x = tokens[:-1, :]
    tokens_y = tokens[1:, :]
    batch_xs.append(np.expand_dims(tokens_x, axis=batch_axis))
    batch_ys.append(np.expand_dims(tokens_y, axis=batch_axis))

  return (np.concatenate(batch_xs, axis=batch_axis).astype(np.float32), 
          np.concatenate(batch_ys, axis=batch_axis).astype(np.float32))


def train_model(model, data, epochs, min_seq=16, max_seq=512):
  num_steps = int(len(data) * epochs)

  loss = torch.nn.MSELoss()
  opt = torch.optim.Adam(model.parameters(), lr=1e-4)

  for step in range(num_steps):
    batch_xs, batch_ys = get_batch(data, np.random.randint(min_seq, max_seq))
    x = torch.from_numpy(batch_xs)
    y_ = torch.from_numpy(batch_ys)

    y = model.forward(x)

    l = loss(y,y_)

    opt.zero_grad()
    l.backward()
    opt.step()

    if step % 100 == 0:
      test_gen = generate_text(model, TEST_TEXT, 256)
      print("Step: %i (%.2f pct) \nSample: %s\n"%(step, (step/num_steps), test_gen))

  


if __name__=="__main__":
  data = load_data(text_only=True)

  model = LSTMModel()

  train_model(model, data, 4)


