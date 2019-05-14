import os
import torch
import random
import numpy as np
from tqdm import tqdm, trange, tqdm_notebook
from torch.utils.data import (
    DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.modeling import BertConfig
from torch.nn import CrossEntropyLoss, KLDivLoss
import torch.nn.functional as F

from lib.train_eval import evaluate
from lib import feature_processors, metrics
from lib.bert import BertForSequenceClassification

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def eval_teacher_soft_targets(teacher_model, tokenizer, params, examples):

    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    
    features = feature_processors.convert_examples_to_features(examples, params['label_list'],
                                                                 params['max_seq_length'], tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in features],
                             dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features],
                              dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features],
                               dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features],
                             dtype=torch.long)

    eval_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)


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
    
    

    
def train_student(model, teacher_model, tokenizer, params, 
                  train_examples, 
                  valid_examples=None,
                  name = None,
                  checkpoint_files={'config': 'bert_config.json',
                            'model_weigths': 'model_trained.pth'},
                  temperature=3, alpha=0.9,
                  all_logits_teacher=None):
    
    if name is not None:
        checkpoint_config = checkpoint_files['config'][:-5] + '_' + name + '.json'
        checkpoint_model_weigths = checkpoint_files['model_weigths'][:-4] + '_' + name + '.pth'
    else:
        checkpoint_config = checkpoint_files['config'][:-5] + '_student.json'
        checkpoint_model_weigths = checkpoint_files['model_weigths'][:-4] + '_student.pth'
    
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
   
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
    
    train_features = feature_processors.convert_examples_to_features(
        train_examples,
        params['label_list'],
        params['max_seq_length'],
        tokenizer)
    
    
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
        for step, batch in enumerate(tqdm_notebook(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(params['device']) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, teacher_logits = batch
            
            logits_model = model(input_ids, segment_ids, input_mask)
        
            loss_first = KLDivLoss()(F.log_softmax(logits_model / temperature), F.softmax(teacher_logits / temperature))
            loss_second = CrossEntropyLoss()(logits_model.view(-1, model.num_labels), label_ids.view(-1))
            loss = loss_first * (temperature ** 2) * alpha + (1. - alpha) * loss_second
#             loss = loss_first * alpha + (1. - alpha) * loss_second
            loss.backward()
          
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        train_result = {
            'train_loss': tr_loss / nb_tr_steps,
            'train_global_step': global_step,
        }
        print(train_result)
        if valid_examples is not None:
            valid_result, valid_prob_preds = evaluate(
                model, tokenizer, params, valid_examples)
            print(valid_result)
            model.train()


#     Save a trained model and the associated configuration
    if not os.path.exists(params['output_dir']):
        os.makedirs(params['output_dir'])

    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(params['output_dir'], checkpoint_model_weigths)
    torch.save(model_to_save.state_dict(), output_model_file)
#     output_config_file = os.path.join(params['output_dir'], checkpoint_config) #### another file
#     with open(output_config_file, 'w') as f:
#         f.write(model_to_save.config.to_json_string())

#     # Load a trained model and config that you have fine-tuned
#     config = BertConfig(output_config_file)
#     model = BertForSequenceClassification(config, num_labels=model.num_labels)
#     model.load_state_dict(torch.load(output_model_file))
#     model.to(device)

    result = {
        'train_loss': tr_loss / nb_tr_steps,
        'train_global_step': global_step
    }
    if valid_examples is not None:
        result['eval_loss'] = valid_result['eval_loss']
        result['eval_accuracy'] = valid_result['eval_accuracy']
    return model, result


def train_student_mae_logits(model, teacher_model, tokenizer, params, 
                             train_examples, 
                             valid_examples=None,
                             name = None,
                             checkpoint_files={'config': 'bert_config.json',
                                               'model_weigths': 'model_trained.pth'},
                             all_logits_teacher=None):
    
    if name is not None:
        checkpoint_config = checkpoint_files['config'][:-5] + '_' + name + '.json'
        checkpoint_model_weigths = checkpoint_files['model_weigths'][:-4] + '_' + name + '.pth'
    else:
        checkpoint_config = checkpoint_files['config'][:-5] + '_student.json'
        checkpoint_model_weigths = checkpoint_files['model_weigths'][:-4] + '_student.pth'
        
        
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    
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
    
    train_features = feature_processors.convert_examples_to_features(
        train_examples,
        params['label_list'],
        params['max_seq_length'],
        tokenizer)
    print("***** Running training *****")
    print("Num examples:",  len(train_examples))
    print("Batch size:  ", params['train_batch_size'])
    print("Num steps:   ", num_train_optimization_steps)
    all_input_ids = torch.tensor(
        [f.input_ids for f in train_features],
         dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in train_features],
         dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in train_features],
         dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in train_features],
         dtype=torch.long)
    
    if all_logits_teacher is None:
        eval_teacher_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        all_logits_teacher = eval_teacher_soft_targets(teacher_model, eval_teacher_dataset, label_list, params)
    
    train_data = TensorDataset(all_input_ids, 
                               all_input_mask, 
                               all_segment_ids, 
                               all_label_ids, 
                               all_logits_teacher)
            
            
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler,
        batch_size=params['train_batch_size'])

    model.train()
    for epoch_num in range(int(params['num_train_epochs'])):
        print('\nEpoch: {}'.format(epoch_num + 1))
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(params['device']) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, teacher_logits = batch
            logits_model = model(input_ids, segment_ids, input_mask)
            loss = torch.nn.L1Loss()(logits_model, teacher_logits)
            
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        train_result = {
            'train_loss': tr_loss / nb_tr_steps,
            'train_global_step': global_step,
        }
        print(train_result)
        if valid_examples is not None:
            valid_result, valid_prob_preds = evaluate(
                model, tokenizer, params, valid_examples)
            print(valid_result)
            model.train()

    if not os.path.exists(params['output_dir']):
        os.makedirs(params['output_dir'])
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(params['output_dir'],
                                     checkpoint_model_weigths)
    torch.save(model_to_save.state_dict(), output_model_file)
#     output_config_file = os.path.join(params['output_dir'],
#                                       checkpoint_files['config'])
#     with open(output_config_file, 'w') as f:
#         f.write(model_to_save.config.to_json_string())

    train_result = {
        'train_loss': tr_loss / nb_tr_steps,
        'train_global_step': global_step,
    }
    
    return model, train_result

