import logging
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import StepLR

from model import BiLSTM_CRF
from data_utils import Processor,Vocabulary,NERDataset,collate_fn
import utils
import config
from train import train
from test import test

def run(word_train, word_dev, label_train, label_dev,vocab,device):
    train_dataset = NERDataset(word_train, label_train, vocab, config.label2id)
    dev_dataset = NERDataset(word_dev, label_dev, vocab, config.label2id)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                            shuffle=False, collate_fn=collate_fn)

    model = BiLSTM_CRF(embedding_size=config.embedding_size,
                       hidden_size=config.hidden_size,
                       drop_out=config.drop_out,
                       vocab_size=vocab.vocab_size(),
                       target_size=vocab.label_size())
    model.to(device)

    optimizer=optim.Adam(model.parameters(),lr=config.lr)
    scheduler=StepLR(optimizer,step_size=config.lr_step,gamma=config.lr_gamma)
    for p in model.crf.parameters():
        _ = torch.nn.init.uniform_(p, -1, 1)
    train(train_loader,dev_loader,vocab, model, optimizer, scheduler, device)
    with torch.no_grad():
        test_loss, f1 = test(config.test_dir, vocab, device)
    return test_loss, f1
if __name__=='__main__':
    #日志文件
    utils.set_logger(config.log_dir)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('device: {}'.format(device))

    #处理json数据，建立词表
    processor = Processor(config)
    processor.data_process()
    vocab=Vocabulary(config)
    vocab.get_vocab()

    #加载训练集和验证集
    word_train, label_train = utils.load_data(config.train_dir)
    word_dev, label_dev = utils.load_data(config.dev_dir)

    #训练模型
    run(word_train, word_dev, label_train, label_dev,vocab,device)








