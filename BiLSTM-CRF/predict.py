import torch
import config
import numpy as np
import logging
from data_utils import Vocabulary
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

id2label={value:key for key,value in config.label2id.items()}
vocab = Vocabulary(config)
data = np.load(config.vocab_path,allow_pickle=True)
word2id = data["word2id"][()]
logging.info("-------- Vocabulary Loaded! --------")


text = input("input:")
example=[word2id.get(char,0) for char in text]
example=torch.LongTensor(example).to(device)

model = torch.load(config.model_dir)
model.to(device)

example=example.unsqueeze(0)
tmp=model.forward(example)
print(tmp.shape)

res = model.crf.decode(tmp)
res=list(itertools.chain.from_iterable(res))
result=[id2label[r] for r in res]
print(result)