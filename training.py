from dotenv import  dotenv_values
from models import *
import numpy as np
import os


CKPT_DIR_PATH = './model_ckpts' 
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


def get_batch(texts, seq_length, batch_size=4, batch_axis=1):
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


def train_model(model, data, epochs=1, min_seq=16, max_seq=512, save_model=False, load_model=True, 
                ckpt_fname=None, steps_p_save=100, steps_p_test=100):
  num_steps = int(len(data) * epochs)
  if save_model or load_model:
    # Create save folder if it doesn't exist
    if not os.path.isdir(CKPT_DIR_PATH):
      print("Checkpoint directory not found. Creating directory: %s"%CKPT_DIR_PATH)
      os.mkdir(CKPT_DIR_PATH)
    # If checkpoint name not given use model class name as default.
    if not ckpt_fname:
      ckpt_fname = model.__class__.__name__ + '.pt'
    # Path to the checkpoint file to save to/ load from.
    ckpt_path = CKPT_DIR_PATH + '/' + ckpt_fname

  loss = torch.nn.MSELoss()
  opt = torch.optim.Adam(model.parameters(), lr=1e-4)

  start_step = 0
  if load_model:
    if os.path.isfile(ckpt_path):
      checkpoint = torch.load(ckpt_path)
      model.load_state_dict(checkpoint['model_state_dict'])
      opt.load_state_dict(checkpoint['optimizer_state_dict'])
      start_step = checkpoint['step']
      print('Loaded model from %s.'%ckpt_path)
    else:
      print('CANNOT LOAD MODEL. NO CHEKCPOINT FILE AT %s'%ckpt_path)
      

  for step in range(start_step, num_steps):
    batch_xs, batch_ys = get_batch(data, np.random.randint(min_seq, max_seq))
    x = torch.from_numpy(batch_xs)
    y_ = torch.from_numpy(batch_ys)

    y = model.forward(x)

    l = loss(y,y_)

    opt.zero_grad()
    l.backward()
    opt.step()

    if step % steps_p_test == 0:
      test_gen = generate_text(model, TEST_TEXT, 256)
      print("Step: %i (%.2f pct) \nSample: %s\n"%(step, (step/num_steps), test_gen))

    if save_model and step % steps_p_save == 0 and step != start_step:      
      print("Saving model ... ", end='')
      torch.save({'step':step,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': opt.state_dict(),
                  'loss': loss
                 }, ckpt_path)
      print("done.")
        


if __name__=="__main__":
  data = load_data(text_only=True)

  model = LinTransformerModel()

  train_model(model, data, save_model=True)


