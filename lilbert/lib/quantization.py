import torch
from sklearn.cluster import MiniBatchKMeans
import math
from tqdm import tqdm_notebook as tqdm
from transformer_modification_utils import replace_transformer_layers


class QuantizedLayer(torch.nn.Module):
    """
    Layer creates quantized version of passed Linear layer
    """

    def __init__(self, layer, n_clusters=8, random_init=False):
        """
        Input: layer -- Linear layer to quantize
               n_clusters -- number of clusters to quantize
               random_init -- perform random initialization
        """
        super(QuantizedLayer, self).__init__()
        # number of bits to store one cluster index
        self.n_bits = math.ceil(math.log2(n_clusters))
        # number of clusters indexes to store in one int64 values
        self.code_size = 63 // self.n_bits
        self.matrix_size = layer.weight.shape

        # masks to reconstruct matrix from compressed form
        self.division_mask = torch.LongTensor([(2 ** self.n_bits) ** i
                                               for i in range(self.code_size)])
        self.mod_mask = torch.LongTensor([(2 ** self.n_bits)
                                          for _ in range(self.code_size)])

        # weight initialization
        if random_init:
            centroids = torch.randn(n_clusters).view(-1, 1)
            centroids_idx = torch.randint(
                low=0, high=n_clusters, size=self.matrix_size).view(-1)
        else:
            algo = MiniBatchKMeans(n_clusters)
            points = layer.weight.view(-1, 1).detach().cpu().numpy()
            algo.fit(points)

            centroids = torch.Tensor(algo.cluster_centers_)
            centroids_idx = torch.LongTensor(algo.predict(points))

        # bias initialization
        if hasattr(layer, 'bias'):
            self.bias = torch.nn.Parameter(layer.bias)
        else:
            self.bias = None

        # fake clusters to store matrix of shape (-1, self.code_size)
        pad = torch.zeros(-len(centroids_idx) % self.code_size).long()
        clusters_matrix = torch.cat(
            [centroids_idx, pad]).view(-1, self.code_size)
        self.codes = torch.nn.Parameter(
            torch.sum(clusters_matrix.long() *
                      torch.LongTensor([(2 ** self.n_bits) ** i
                                        for i in range(self.code_size)]),
                      dim=-1),
            requires_grad=False)
        self.codes_embedding = torch.nn.Embedding.from_pretrained(centroids)

    def forward(self, input_):
        # clusters indexes from codes
        decoded_matrix = self.codes.view(-1, 1) //\
            self.division_mask.to(input_.device) %\
            self.mod_mask.to(input_.device)
        decoded_clusters = decoded_matrix.view(-1)[:self.matrix_size.numel()]
        # reconstructed matrix
        weight = self.codes_embedding(decoded_clusters).view(self.matrix_size)
        return torch.functional.F.linear(input_, weight, self.bias)


def quantize_transformer(model, params,
                         n_clusters=8,
                         random_init=False,
                         intermediate_training=False,
                         tokenizer=None,
                         train_examples=None,
                         dev_examples=None
                         ):
    """
    Input: model -- BERT model to quantize
            params -- parameters of the model
            n_clusters -- number of clusters to quantize
            intermediate_training -- train model after quantizing some blocks
    """
    device = params['device']
    if intermediate_training:
        blocks = [
            [6, 3, 7, 8],
            [4, 5, 0, 11],
            [9, 2, 10, 1]
        ]
        for i, current_blocks in enumerate(blocks):
            replace_transformer_layers(model,
                                       QuantizedLayer,
                                       blocks=current_blocks,
                                       n_clusters=n_clusters,
                                       random_init=random_init)
            model = model.to(device)

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
        replace_transformer_layers(
            model, QuantizedLayer, n_clusters=n_clusters, random_init=random_init)
        model = model.to(device)
