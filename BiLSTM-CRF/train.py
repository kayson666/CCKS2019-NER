import torch
from tqdm import tqdm
import logging
import config
from metric import f1_score, bad_case
from torch.utils.tensorboard import SummaryWriter
import os
import time

log_dir='./runs/'
writer = SummaryWriter(os.path.join(log_dir, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))


def train(train_loader, dev_loader, vocab, model, optimizer, scheduler, device):
    best_val_f1=0.0
    patience_counter = 0
    if os.path.exists(config.model_dir):
        # 加载模型
        model = torch.load(config.model_dir)
        model.to(device)
        logging.info("--------Load model from {}--------".format(config.model_dir))
    for epoch in range(1,config.epoch_num):
        model.train()
        # step number in one epoch: 336
        train_loss = 0.0
        for idx, batch_samples in enumerate(tqdm(train_loader)):
            x, y, mask, lens = batch_samples
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            model.zero_grad()
            tag_scores, loss = model.forward_with_crf(x, mask, y)
            train_loss += loss.item()
            # 梯度反传
            loss.backward()
            # 优化更新
            optimizer.step()
            optimizer.zero_grad()
        # scheduler
        scheduler.step()
        train_loss = float(train_loss) / len(train_loader)
        logging.info("epoch: {}, train loss: {}".format(epoch, train_loss))

        #验证
        with torch.no_grad():
            # dev loss calculation
            metric = dev(dev_loader, vocab, model, device)
            val_f1 = metric['f1']
            dev_loss = metric['loss']
            logging.info("epoch: {}, f1 score: {}, " "dev loss: {}".format(epoch, val_f1, dev_loss))
            improve_f1 = val_f1 - best_val_f1
            if improve_f1 > 1e-5:
                best_val_f1 = val_f1
                torch.save(model, config.model_dir)
                logging.info("--------Save best model!--------")
                if improve_f1 < config.patience:
                    patience_counter += 1
                else:
                    patience_counter = 0
            else:
                patience_counter += 1
            # Early stopping and logging best f1 如果5个epoch的增加均小于patience值，跳出
            if (patience_counter >= config.patience_num and epoch > config.min_epoch_num) or epoch == config.epoch_num:
                logging.info("Best val f1: {}".format(best_val_f1))
                break
        writer.add_scalar('Training/training loss',train_loss ,epoch)#tensorboard --logdir "./runs"启动
        writer.add_scalar('Validation/loss', dev_loss, epoch)
        writer.add_scalar('Validation/f1', val_f1, epoch)
        logging.info("Training Finished!")



def dev(data_loader, vocab, model, device, mode='dev'):
    """test model performance on dev-set"""
    model.eval()
    true_tags = []
    pred_tags = []
    sent_data = []
    dev_losses = 0
    for idx, batch_samples in enumerate(data_loader):
        sentences, labels, masks, lens = batch_samples
        sent_data.extend([[vocab.id2word.get(idx.item()) for i, idx in enumerate(indices) if mask[i] > 0]
                          for (mask, indices) in zip(masks, sentences)])
        sentences = sentences.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        y_pred = model.forward(sentences)
        labels_pred = model.crf.decode(y_pred, mask=masks)
        targets = [itag[:ilen] for itag, ilen in zip(labels.cpu().numpy(), lens)]
        true_tags.extend([[vocab.id2label.get(idx) for idx in indices] for indices in targets])
        pred_tags.extend([[vocab.id2label.get(idx) for idx in indices] for indices in labels_pred])
        # 计算梯度
        _, dev_loss = model.forward_with_crf(sentences, masks, labels)
        dev_losses += dev_loss.item()
    assert len(pred_tags) == len(true_tags)
    if mode == 'test':
        assert len(sent_data) == len(true_tags)

    # logging loss, f1 and report
    metrics = {}
    if mode == 'dev':
        f1 = f1_score(true_tags, pred_tags, mode)
        metrics['f1'] = f1
    else:
        bad_case(true_tags, pred_tags, sent_data)
        f1_labels, f1 = f1_score(true_tags, pred_tags, mode)
        metrics['f1_labels'] = f1_labels
        metrics['f1'] = f1
    metrics['loss'] = float(dev_losses) / len(data_loader)
    return metrics











