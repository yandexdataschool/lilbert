import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import (
    DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.modeling import BertConfig

from lib import feature_processors, metrics
from lib.bert import BertForSequenceClassification


def train(model, tokenizer, params,
          train_examples,
          valid_examples=None,
          checkpoint_files={'config': 'bert_config.json',
                            'model_weigths': 'model_trained.pth'}):
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
    train_data = TensorDataset(all_input_ids,
                               all_input_mask,
                               all_segment_ids,
                               all_label_ids)
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
            input_ids, input_mask, segment_ids, label_ids = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids)
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

#     if not os.path.exists(params['output_dir']):
#         os.makedirs(params['output_dir'])
#     model_to_save = model.module if hasattr(model, 'module') else model
#     output_model_file = os.path.join(params['output_dir'],
#                                      checkpoint_files['file_to_save'])
#     torch.save(model_to_save.state_dict(), output_model_file)
#     output_config_file = os.path.join(params['output_dir'],
#                                       checkpoint_files['config'])
#     with open(output_config_file, 'w') as f:
#         f.write(model_to_save.config.to_json_string())

#     train_result = {
#         'train_loss': tr_loss / nb_tr_steps,
#         'train_global_step': global_step,
#     }
    
    return model, train_result


def predict(model, tokenizer, params, valid_examples):
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    
    eval_features = feature_processors.convert_examples_to_features(
            valid_examples,
            params['label_list'],
            params['max_seq_length'],
            tokenizer)
    all_input_ids = torch.tensor(
        [f.input_ids for f in eval_features],
         dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in eval_features],
         dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in eval_features],
         dtype=torch.long)
    eval_data = TensorDataset(all_input_ids,
                              all_input_mask,
                              all_segment_ids)

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data,
                                 sampler=eval_sampler,
                                 batch_size=params['eval_batch_size'])

    model.eval()
    softmax = torch.nn.Softmax(dim=-1)
    test_preds = []
    for input_ids, input_mask, segment_ids in tqdm(
            eval_dataloader, desc='Evaluating'):
        logits = model(
            input_ids.to(params['device']),
            segment_ids.to(params['device']),
            input_mask.to(params['device'])).detach().cpu()
        test_preds += list(softmax(logits).numpy())

    return np.array(test_preds)


def evaluate(model, tokenizer, params, valid_examples):
    print("***** Running evaluation *****")
    print("Num examples: ", len(valid_examples))
    print("Batch size:   ", params['eval_batch_size'])
    
    prob_preds = predict(model, tokenizer, params, valid_examples)
    true_labels = np.array([int(example.label) 
                            for i, example in enumerate(valid_examples)])
    result = {
        'eval_loss': metrics.log_loss(true_labels, prob_preds),
        'eval_accuracy': metrics.accuracy(true_labels, prob_preds),
        'eval_f1_score': metrics.f1_score(true_labels, prob_preds),
        'eval_matthews_corrcoef': metrics.matthews_corrcoef(true_labels, prob_preds)
    }
    return result, prob_preds
