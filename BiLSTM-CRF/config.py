
log_dir='./log/train.log'
data_dir='./data/'
files = ['train', 'dev','test']
vocab_path=data_dir + 'vocab.npz'

train_dir ='./data/'  + 'train.npz'
dev_dir = './data/' + 'dev.npz'
test_dir = './data/' + 'test.npz'


case_dir='./data/'  + 'bad_case.txt'
dev_split_size=0.1
max_vocab_size=1000000
batch_size=16
epoch_num=30
embedding_size=128
hidden_size=256
drop_out=0.2

lr = 0.001
lr_step = 5
lr_gamma = 0.8

min_epoch_num = 5
patience = 0.0002
patience_num = 2

model_dir = './save/' + 'model.pth'
# labels = ['疾病和诊断', '药物', '实验室检验', '手术', '药物',
#           '解剖部位']
labels = ['解剖部位', '影像检查', '疾病和诊断', '手术', '药物',
          '实验室检验']

label2id = {
    "O": 0,
    "B-解剖部位": 1,
    "B-影像检查": 2,
    "B-疾病和诊断": 3,
    'B-手术': 4,
    'B-药物': 5,
    'B-实验室检验': 6,
    "I-解剖部位": 7,
    "I-影像检查": 8,
    "I-疾病和诊断": 9,
    'I-手术': 10,
    'I-药物': 11,
    'I-实验室检验': 12,
    "S-解剖部位": 13,
    "S-影像检查": 14,
    "S-疾病和诊断": 15,
    'S-手术': 16,
    'S-药物': 17,
    'S-实验室检验': 18
}

id2label = {_id: _label for _label, _id in list(label2id.items())}

#tensorboard --logdir "./runs"