import os
import json
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter

class Processor:
    def __init__(self, config):
        self.data_dir = config.data_dir
        self.files = config.files

    def data_process(self):
        for file_name in self.files:
            self.get_examples(file_name)

    def get_examples(self, mode):
        """
        将json文件每一行中的文本分离出来，存储为words列表
        标记文本对应的标签，存储为labels
        words示例：['下', '腹', '部', '息', '不', '适']
        labels示例：['B-解剖部位', 'I-解剖部位', 'I-解剖部位','O', 'O']
        """
        input_dir = self.data_dir + str(mode) + '.json'
        output_dir = self.data_dir + str(mode) + '.npz'
        if os.path.exists(output_dir):
            return
        with open(input_dir, 'r', encoding='utf-8') as f:
            word_list = []
            label_list = []
            length=[]
            labels = ['解剖部位', '影像检查', '疾病和诊断', '手术', '药物',
                      '实验室检验']
            dict = {"解剖部位": 0, "影像检查": 0, "疾病和诊断": 0, "手术": 0, "药物": 0, "实验室检验": 0}
            # 先读取到内存中，然后逐行处理
            for line in f.readlines():
                # loads()：用于处理内存中的json对象，strip去除可能存在的空格
                json_line = json.loads(line.strip())#
                text = json_line['text']
                words = list(text)
                # 如果没有label，则返回None
                label_entities = json_line.get('entity_list', None)
                labels = ['O'] * len(words)
                if label_entities is not None:
                    for line in label_entities:
                        length.extend([len(line['entity'])])
                        start=line['entity_index']['begin']
                        end=line['entity_index']['end']
                        assert ''.join(words[start:end]) == line['entity']
                        if start == end-1:
                            labels[start] = 'S-' + line['entity_type']
                        else:
                            labels[start] = 'B-' + line['entity_type']
                            labels[start + 1:end] = ['I-' + line['entity_type']] * (len(line['entity']) - 1)
                            dict[line['entity_type']] += 1
                word_list.append(words)
                label_list.append(labels)

            resnum=Counter(length)
            # 保存成二进制文件
            #实体种类分布
            # with open('./data/'+str(mode)+'entity'+'.txt','w',encoding='utf-8') as f:
            #     f.write(json.dumps(dict,ensure_ascii=False))
            # #实体长度分布
            # with open('./data/'+str(mode)+'fre'+'.txt','w',encoding='utf-8') as f:
            #     f.write(json.dumps(resnum,ensure_ascii=False))
            np.savez_compressed(output_dir, words=word_list, labels=label_list)
            logging.info("-------- {} data process DONE!--------".format(mode))

class Vocabulary:
    """
    构建词表
    """
    def __init__(self, config):
        self.data_dir = config.data_dir
        self.files = config.files
        self.vocab_path = config.vocab_path
        self.max_vocab_size = config.max_vocab_size
        self.word2id = {}
        self.id2word = None
        self.label2id = config.label2id
        self.id2label = config.id2label

    def __len__(self):
        return len(self.word2id)

    def vocab_size(self):
        return len(self.word2id)

    def label_size(self):
        return len(self.label2id)

    # 获取词的id
    def word_id(self, word):
        return self.word2id[word]

    # 获取id对应的词
    def id_word(self, idx):
        return self.id2word[idx]

    # 获取label的id
    def label_id(self, word):
        return self.label2id[word]

    # 获取id对应的词
    def id_label(self, idx):
        return self.id2label[idx]

    def get_vocab(self):
        """
        进一步处理，将word和label转化为id
        word2id: dict,每个字对应的序号
        idx2word: dict,每个序号对应的字
        保存为二进制文件
        """
        # 如果有处理好的，就直接load
        if os.path.exists(self.vocab_path):
            data = np.load(self.vocab_path, allow_pickle=True)
            # '[()]'将array转化为字典
            self.word2id = data["word2id"][()]
            self.id2word = data["id2word"][()]
            logging.info("-------- Vocabulary Loaded! --------")
            return
        # 如果没有处理好的二进制文件，就处理原始的npz文件
        word_freq = {}
        for file in self.files:
            data = np.load(self.data_dir + str(file) + '.npz', allow_pickle=True)
            word_list = data["words"]
            # 常见的单词id最小
            for line in word_list:
                for ch in line:
                    if ch in word_freq:
                        word_freq[ch] += 1
                    else:
                        word_freq[ch] = 1
        index = 0
        sorted_word = sorted(word_freq.items(), key=lambda e: e[1], reverse=True)
        # 构建word2id字典
        for elem in sorted_word:
            self.word2id[elem[0]] = index
            index += 1
            if index >= self.max_vocab_size:
                break
        # id2word保存
        self.id2word = {_idx: _word for _word, _idx in list(self.word2id.items())}
        # 保存为二进制文件
        np.savez_compressed(self.vocab_path, word2id=self.word2id, id2word=self.id2word)
        logging.info("-------- Vocabulary Build! --------")

class NERDataset(Dataset):
    def __init__(self, words, labels, vocab, label2id):
        self.vocab = vocab
        self.dataset = self.preprocess(words, labels)
        self.label2id = label2id

    def preprocess(self, words, labels):
        """convert the data to ids"""
        processed = []
        for (word, label) in zip(words, labels):
            word_id = [self.vocab.word_id(w_) for w_ in word]
            label_id = [self.vocab.label_id(l_) for l_ in label]
            processed.append((word_id, label_id))
        logging.info("-------- Process Done! --------")
        return processed

    def __getitem__(self, idx):
        word = self.dataset[idx][0]
        label = self.dataset[idx][1]
        return [word, label]

    def __len__(self):
        return len(self.dataset)

def get_long_tensor(texts, labels, batch_size):

    token_len = max([len(x) for x in texts])
    text_tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    label_tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    mask_tokens = torch.ByteTensor(batch_size, token_len).fill_(0)

    for i, s in enumerate(zip(texts, labels)):
        text_tokens[i, :len(s[0])] = torch.LongTensor(s[0])
        label_tokens[i, :len(s[1])] = torch.LongTensor(s[1])
        mask_tokens[i, :len(s[0])] = torch.tensor([1] * len(s[0]), dtype=torch.uint8)

    return text_tokens, label_tokens, mask_tokens

def collate_fn(batch):

    texts = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    lens = [len(x) for x in texts]
    batch_size = len(batch)

    input_ids, label_ids, input_mask = get_long_tensor(texts, labels, batch_size)

    return [input_ids, label_ids, input_mask, lens]



























