import numpy as np
import random
import torch
from pytorch_pretrained_bert import BertForMultipleChoice
import os
from pytorch_pretrained_bert.tokenization import BertTokenizer

from lib import data_processors, tasks
from lib.bert import BertForSequenceClassification
from lib.train_eval import train, evaluate, predict

params = {
    'data_dir': '../../data/SWAG',
    'output_dir': '../../output',
    'cache_dir': '../../model_cache',
    'task_name': 'swag',
    'bert_model': 'bert-base-uncased',
    'max_seq_length': 12,
    'train_batch_size': 1,
    'eval_batch_size': 8,
    'learning_rate': 2e-5,
    'warmup_proportion': 0.1,
    'num_train_epochs': 1,
    'seed': 1331,
    'device': torch.device(
        'cuda' if torch.cuda.is_available()
        else 'cpu')
}

random.seed(params['seed'])
np.random.seed(params['seed'])
torch.manual_seed(params['seed'])

processor = tasks.processors[params['task_name']]()
tokenizer = BertTokenizer.from_pretrained(
    params['bert_model'], do_lower_case=True)

train_examples = processor.get_train_examples(params['data_dir'])
dev_examples = processor.get_dev_examples(params['data_dir'])

model = BertForMultipleChoice.from_pretrained(
    params['bert_model'],
    cache_dir=params['cache_dir'], num_choices=4).to(params['device'])

EPOCH_NUM = 1

params['num_train_epochs'] = 1
checkpoint_files = {
    'config': 'bert_config.json',
    'model_weigths': 'model_{}_epoch_{}.pth'.format(
        params['task_name'], EPOCH_NUM)
}

model, result = train(model, tokenizer, params,
                      train_examples,
                      valid_examples=dev_examples,
                      checkpoint_files=checkpoint_files)
print(result)
