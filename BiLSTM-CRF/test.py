import numpy as np
import logging
import config
import torch
from torch.utils.data import DataLoader
from data_utils import NERDataset,collate_fn
from train import dev

def test(dataset_dir, vocab, device):
    """test model performance on the final test set"""
    data = np.load(dataset_dir, allow_pickle=True)
    word_test = data["words"]
    label_test = data["labels"]
    # build dataset
    test_dataset = NERDataset(word_test, label_test, vocab, config.label2id)
    # build data_loader
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=True, collate_fn=collate_fn)
    # Prepare model
    if config.model_dir is not None:
        # model
        model = torch.load(config.model_dir)
        model.to(device)
        logging.info("--------Load model from {}--------".format(config.model_dir))
    else:
        logging.info("--------No model to test !--------")
        return
    metric = dev(test_loader, vocab, model, device, mode='test')
    f1 = metric['f1']
    test_loss = metric['loss']
    logging.info("final test loss: {}, f1 score: {}".format(test_loss, f1))
    val_f1_labels = metric['f1_labels']
    for label in config.labels:
        logging.info("f1 score of {}: {}".format(label, val_f1_labels[label]))
    return test_loss, f1