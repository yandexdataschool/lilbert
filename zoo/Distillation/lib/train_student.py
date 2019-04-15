import os
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.modeling import (BertConfig, WEIGHTS_NAME, CONFIG_NAME)
from lib.bert import BertForSequenceClassification
from tqdm import tqdm, trange, tqdm_notebook
from torch.nn import CrossEntropyLoss, KLDivLoss
import torch.nn.functional as F

# another file for student model
WEIGHTS_NAME_STUDENT = 'pytorch_model_student.bin'
CONFIG_NAME_STUDENT = 'bert_config_student.json'

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

from lib import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def eval_teacher_soft_targets(teacher_model, eval_dataset, label_list, params):

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=params['train_batch_size'])

    teacher_model.eval()
     
    all_logist = []
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc='Evaluating'):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = teacher_model(input_ids, segment_ids, input_mask)
        all_logist.append(logits)
        
    return torch.cat(all_logist, 0)
    

    
def train_student(model, teacher_model, train_examples, label_list, params, tokenizer, temperature=10, alpha=0.7,
                  all_logits_teacher=None):
    train_steps_per_epoch = int(len(train_examples) / params['train_batch_size'])
    num_train_optimization_steps = train_steps_per_epoch * params['num_train_epochs']

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=params['learning_rate'],
                         warmup=params['warmup_proportion'],
                         t_total=num_train_optimization_steps)
    
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    
    train_features = utils.convert_examples_to_features(
        train_examples, label_list, params['max_seq_length'], tokenizer)
    
    print("***** Running training *****")
    print("Num examples:",  len(train_examples))
    print("Batch size:  ", params['train_batch_size'])
    print("Num steps:   ", num_train_optimization_steps)
    all_input_ids = torch.tensor([f.input_ids for f in train_features],
                                 dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features],
                                  dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features],
                                   dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features],
                                 dtype=torch.long)
    
    if all_logits_teacher is None:
        eval_teacher_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        all_logits_teacher = eval_teacher_soft_targets(teacher_model, eval_teacher_dataset, label_list, params)
    
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_logits_teacher)
    
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=params['train_batch_size'])

    model.train()
    for epoch_num in range(int(params['num_train_epochs'])):
        print('Epoch: {}'.format(epoch_num + 1))
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, teacher_logits = batch
            
            logits_model = model(input_ids, segment_ids, input_mask)
        
            loss_first = KLDivLoss()(F.log_softmax(logits_model / temperature), F.softmax(teacher_logits / temperature))
            loss_second = CrossEntropyLoss()(logits_model.view(-1, model.num_labels), label_ids.view(-1))
            loss = loss_first * (temperature ** 2) * 2 * alpha + (1. - alpha) * loss_second
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

    # Save a trained model and the associated configuration
#     model_to_save = model.module if hasattr(model, 'module') else model
#     output_model_file = os.path.join(params['output_dir'], WEIGHTS_NAME_STUDENT) #### another file
#     torch.save(model_to_save.state_dict(), output_model_file)
#     output_config_file = os.path.join(params['output_dir'], CONFIG_NAME_STUDENT) #### another file
#     with open(output_config_file, 'w') as f:
#         f.write(model_to_save.config.to_json_string())

#     # Load a trained model and config that you have fine-tuned
#     config = BertConfig(output_config_file)
#     model = BertForSequenceClassification(config, num_labels=model.num_labels)
#     model.load_state_dict(torch.load(output_model_file))
#     model.to(device)
    result = {
        'loss': tr_loss / nb_tr_steps,
        'global_step': global_step
    }
    
#     return model, result
    return result