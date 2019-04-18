import os
import sys

import numpy as np
import random
import torch

from pytorch_pretrained_bert.tokenization import BertTokenizer

from lib import data_processors, tasks
from lib.train_eval import train


def run_train_task(model, params, task_name, num_epochs):
    params['task_name'] = task_name
    params['num_labels'] = tasks.num_labels[params['task_name']]
    params['label_list'] = tasks.label_lists[params['task_name']]
    params['data_dir'] = tasks.data_dirs[params['task_name']]

    processor = tasks.processors[params['task_name']]()
    tokenizer = BertTokenizer.from_pretrained(
        params['bert_model'],
        do_lower_case=True)

    train_examples = processor.get_train_examples(params['data_dir'])
    dev_examples = processor.get_dev_examples(params['data_dir'])

    for epoch_num in range(1, num_epochs + 1):
        params['num_train_epochs'] = 1
        checkpoint_files = {
            'config': 'bert_config.json',
            'file_to_save': 'model_{}_epoch_{}.pth'.format(
                params['task_name'], epoch_num)
        }
        model, result = train(model, tokenizer, params,
                              train_examples,
                              valid_examples=dev_examples,
                              checkpoint_files=checkpoint_files)
        print('saved to model_{}_epoch_{}.pth'.format(
                params['task_name'], epoch_num))

    return model
