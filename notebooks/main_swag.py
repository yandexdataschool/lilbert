import os
import sys
# sys.path.append('..')

import numpy as np
import random
import torch
from pytorch_pretrained_bert import BertForMultipleChoice

from lib import data_processors, utils
from lib.bert import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from lib.train_eval import train, evaluate, train_swag
from lib import swag_utils

params = {
    'data_dir': '../data/SWAG',
    'output_dir': '../output',
    'cache_dir': '../model_cache',
    'task_name': 'swag',
    'bert_model': 'bert-base-uncased',
    'max_seq_length': 12,
    # 'max_seq_length': 128,
    'train_batch_size': 1,
    # 'train_batch_size': 32,
    'eval_batch_size': 8,
    'learning_rate': 2e-5,
    'warmup_proportion': 0.1,
    'num_train_epochs': 1,
    'seed': 1331
}

random.seed(params['seed'])
np.random.seed(params['seed'])
torch.manual_seed(params['seed'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


tokenizer = BertTokenizer.from_pretrained(
    params['bert_model'], do_lower_case=True)

# train_examples = processor.get_train_examples(params['data_dir'])
# eval_examples = processor.get_dev_examples(params['data_dir'])
train_examples = swag_utils.read_swag_examples(os.path.join(params['data_dir'], 'train.csv'), is_training=True)

model = BertForMultipleChoice.from_pretrained(
    params['bert_model'], cache_dir=params['cache_dir'], num_choices=4).to(device)

# model = BertForSequenceClassification.from_pretrained(
#     params['bert_model'], cache_dir=params['cache_dir'], num_labels=num_labels).to(device)

model, result = train_swag(model, train_examples, params, tokenizer)
