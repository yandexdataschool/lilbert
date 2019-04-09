import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm


class SVDLinear(nn.Module):
    def __init__(self, in_features, out_features, hidden_size, bias=True):
        super(SVDLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_size = hidden_size

        self.u = Parameter(torch.Tensor(out_features, self.hidden_size))
        self.s = Parameter(torch.Tensor(self.hidden_size))
        self.v = Parameter(torch.Tensor(self.hidden_size, in_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.init_parameters()

    def init_weights(self, weight=None):
        if weight is None:
            init.kaiming_uniform_(self.u, a=math.sqrt(5))
            init.uniform_(self.s, a=math.sqrt(5))
            init.kaiming_uniform_(self.v, a=math.sqrt(5))
        else:
            u, s, v = np.linalg.svd(weight)
            del self.u, self.s, self.v
            self.u = Parameter(torch.Tensor(u[:, :self.hidden_size]))
            self.s = Parameter(torch.Tensor(s[:self.hidden_size]))
            self.v = Parameter(torch.Tensor(v[:self.hidden_size, :]))

    def init_parameters(self, weight=None):
        self.init_weights(weight)

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(
                torch.mm(self.u, torch.mm(torch.diag(self.s), self.v))
            )
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        output = F.linear(input, self.v)
        output = F.linear(output, torch.diag(self.s))
        output = F.linear(output, self.u, self.bias)
        return output


def find_hidden_size(in_features, out_features, compression_rate):
    """
    Returns hidden size of matrices for known compression rate
    """
    return int(in_features * out_features /
               (compression_rate * (in_features + out_features + 1)))


def find_compression_rate(in_features, out_features, hidden_size):
    """
    Returns matrix compression rate for known hidden size
    """
    return in_features * out_features / \
           (hidden_size * (in_features + out_features + 1))


def linear_to_svd(linear_layer, hidden_size=None, compression_rate=None):
    """
    Returns SVDLinear layer for linear layer with hidden size
    equal to hidden_size parameter
    if hidden_size is None hidden_size is calculated by the compression_rate
    """
    if hidden_size is None and compression_rate is None:
        raise ValueError("At least one parameter (hidden_size or compression rate) should be not None")

    dense_weight = linear_layer.weight.cpu().data.numpy()
    in_features = linear_layer.in_features
    out_features = linear_layer.out_features

    if hidden_size is None:
        hidden_size = find_hidden_size(
            in_features, out_features, compression_rate
        )

    svd_linear = SVDLinear(in_features, out_features, hidden_size)
    svd_linear.init_weights(dense_weight)

    return svd_linear


def change_transformer_linears_to_svd(model, params, hidden_size=None, compression_rate=None):
    """
    Changes all linear layers to SVDLinear layers with hidden size
    equal to hidden_size parameter
    if hidden_size is None hidden_size for each layer is calculated by the compression_rate
    """
    if hidden_size is None and compression_rate is None:
        raise ValueError("At least one parameter (hidden_size or compression rate) should be not None")

    encoder_layers = list(model.children())[0].encoder.layer
    device = params['device']

    for bert_layer in tqdm(encoder_layers):
        attention = bert_layer.attention

        attention.self.query = linear_to_svd(
            attention.self.query,
            hidden_size,
            compression_rate
        ).to(device)

        attention.self.key = linear_to_svd(
            attention.self.key,
            hidden_size,
            compression_rate
        ).to(device)

        attention.self.value = linear_to_svd(
            attention.self.value,
            hidden_size,
            compression_rate
        ).to(device)

        attention.output.dense = linear_to_svd(
            attention.output.dense,
            hidden_size,
            compression_rate
        ).to(device)

        bert_layer.intermediate.dense = linear_to_svd(
            bert_layer.intermediate.dense,
            hidden_size,
            compression_rate
        ).to(device)

        bert_layer.output.dense = linear_to_svd(
            bert_layer.output.dense,
            hidden_size,
            compression_rate
        ).to(device)
    return model
