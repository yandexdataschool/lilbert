import torch
from sklearn.cluster import MiniBatchKMeans
import math
from tqdm import tqdm_notebook as tqdm
class QuantizedLayer(torch.nn.Module):
    def __init__(self, layer=None, n_clusters=8,  size=None):
        super(QuantizedLayer, self).__init__()
        self.n_bits = math.ceil(math.log2(n_clusters)) # int64
        self.code_size = 63 // self.n_bits  # 8 clusters to 63 bits
        
        if layer is None:
            if size is None:
                raise ValueError("During random init, size must be passed.")
            self.matrix_size = size
            
            centroids = torch.randn(n_clusters).view(-1,1)
            centroids_idx = torch.randint(low=0, high=n_clusters, size=size).view(-1)
            self.bias=torch.nn.Parameter(torch.randn(size[0]))
        else:
            self.matrix_size = layer.weight.size()
            algo = MiniBatchKMeans(n_clusters)
            points = layer.weight.view(-1, 1).detach().cpu().numpy()
            algo.fit(points)
            
            centroids = torch.Tensor(algo.cluster_centers_)
            centroids_idx = torch.LongTensor(algo.predict(points))
            if hasattr(layer, 'bias'):
                self.bias = torch.nn.Parameter(layer.bias)
            else:
                self.bias = None
            
        pad = torch.zeros(-len(centroids_idx) % self.code_size).long()
        to_code = torch.cat([centroids_idx, pad]).view(-1, self.code_size)
        self.codes = torch.nn.Parameter(
            torch.sum(to_code.long() *\
                      torch.LongTensor([(2 ** self.n_bits) ** i for i in range(self.code_size)]), dim=-1),
                                        requires_grad=False)
        self.emb = torch.nn.Embedding.from_pretrained(centroids)
        
        
    def forward(self, input_):
        decoded = self.codes.view(-1, 1) //\
            torch.LongTensor([(2 ** self.n_bits) ** i for i in range(self.code_size)]).to(input_.device) %\
            torch.LongTensor([(2 ** self.n_bits) for _ in range(self.code_size)]).to(input_.device)
        decoded = decoded.view(-1)[:self.matrix_size.numel()]
        weight = self.emb(decoded)
        weight = weight.view(self.matrix_size)
        return torch.functional.F.linear(input_, weight, self.bias)
    
    
def quantize_transformer(model, params, n_clusters=8,
                         intermediate_training=False,
                         tokenizer=None,
                         train_examples=None,
                         dev_examples=None
                         ):
    """
    Input: model -- BERT model to quantize
            params -- parameters of the model
            n_clusters -- number of clusters in which the layers will be quantized
            intermediate_training -- train model after quantizing some blocks
    """
    device = params['device']
    if intermediate_training:
        # raise ValueError("Not implemented yet!")
        blocks = [
            [6, 3, 7, 8],
            [4, 5, 0, 11],
            [9, 2, 10, 1]
        ]
        for i, block in enumerate(blocks):
            for transformer_layer_ind in tqdm(block):
                model.bert.encoder.layer[transformer_layer_ind].attention.self.query = \
                    QuantizedLayer(model.bert.encoder.layer[transformer_layer_ind].attention.self.query, n_clusters).to(
                        device)

                model.bert.encoder.layer[transformer_layer_ind].attention.self.key = \
                    QuantizedLayer(model.bert.encoder.layer[transformer_layer_ind].attention.self.key, n_clusters).to(
                        device)

                model.bert.encoder.layer[transformer_layer_ind].attention.self.value = \
                    QuantizedLayer(model.bert.encoder.layer[transformer_layer_ind].attention.self.value, n_clusters).to(
                        device)

                model.bert.encoder.layer[transformer_layer_ind].attention.output.dense = \
                    QuantizedLayer(model.bert.encoder.layer[transformer_layer_ind].attention.output.dense,
                                   n_clusters).to(device)

                model.bert.encoder.layer[transformer_layer_ind].intermediate.dense = \
                    QuantizedLayer(model.bert.encoder.layer[transformer_layer_ind].intermediate.dense, n_clusters).to(
                        device)

                model.bert.encoder.layer[transformer_layer_ind].output.dense = \
                    QuantizedLayer(model.bert.encoder.layer[transformer_layer_ind].output.dense, n_clusters).to(device)

            EPOCH_NUM = i

            params['num_train_epochs'] = 1
            checkpoint_files = {
                'config': 'bert_config.json',
                'file_to_save': 'model_{}_epoch_{}.pth'.format(
                    params['task_name'], EPOCH_NUM)
            }

            model, result = train(model, tokenizer, params,
                                  train_examples,
                                  valid_examples=dev_examples,
                                  checkpoint_files=checkpoint_files)

    else:
        for transformer_layer_ind in tqdm(range(12)):
            model.bert.encoder.layer[transformer_layer_ind].attention.self.query = \
            QuantizedLayer(model.bert.encoder.layer[transformer_layer_ind].attention.self.query, n_clusters).to(device)

            model.bert.encoder.layer[transformer_layer_ind].attention.self.key = \
            QuantizedLayer(model.bert.encoder.layer[transformer_layer_ind].attention.self.key, n_clusters).to(device)


            model.bert.encoder.layer[transformer_layer_ind].attention.self.value = \
            QuantizedLayer(model.bert.encoder.layer[transformer_layer_ind].attention.self.value, n_clusters).to(device)


            model.bert.encoder.layer[transformer_layer_ind].attention.output.dense =\
            QuantizedLayer(model.bert.encoder.layer[transformer_layer_ind].attention.output.dense, n_clusters).to(device)


            model.bert.encoder.layer[transformer_layer_ind].intermediate.dense =\
            QuantizedLayer(model.bert.encoder.layer[transformer_layer_ind].intermediate.dense, n_clusters).to(device)


            model.bert.encoder.layer[transformer_layer_ind].output.dense =\
            QuantizedLayer(model.bert.encoder.layer[transformer_layer_ind].output.dense, n_clusters).to(device)
    
