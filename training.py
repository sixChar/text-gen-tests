from dotenv import  dotenv_values





DATA_PATH = dotenv_values(".env")['BLOG_DATA_PATH']


def load_data(text_only=False):
  lines = []
  with open(DATA_PATH, 'r') as fp:
    next(fp) # Skip header which looks like: (id, gender, age, topic, sign, date, text)
    for line in fp:
      line_data = line.split(',')
      if text_only:
        line_data = line_data[-1]
      lines.append(line_data)
  return lines


if __name__=="__main__":
  data = load_data(text_only=True)

  print(len(data))
  print(len(data[10]))
  print(len(data[100]))
  print(len(data[10000]))
  print(len(data[100000]))





