
from lib.tt.t3nsor.layers import TTEmbedding, TTLinear, TTConv1
from pytorch_pretrained_bert.modeling import *
from lib.tt.t3nsor import decompositions


def make_tt_linear(layer, in_features=768, out_features=768, shape=([4, 4, 6, 8], [4, 4, 6, 8]), rank=32):
    return TTConv1(
        in_features=in_features,
        out_features=out_features,
        init=decompositions.to_tt_matrix(layer.weight, shape=shape, max_tt_rank=rank)
    )


class TTBertEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super(TTBertEmbeddings, self).__init__(config)
        self.word_embeddings = TTEmbedding(shape=[[12, 12, 15, 15], [4, 4, 6, 8]], tt_rank=32)


class TTBertSelfAttention(BertSelfAttention):
    def __init__(self, config):
        super(TTBertSelfAttention, self).__init__(config)

        self.query = make_tt_linear(self.query)
        self.key = make_tt_linear(self.key)
        self.value = make_tt_linear(self.value)


class TTBertSelfOutput(BertSelfOutput):
    def __init__(self, config):
        super(TTBertSelfOutput, self).__init__(config)
        self.dense = make_tt_linear(self.dense)


class TTBertAttention(BertAttention):
    def __init__(self, config):
        super(TTBertAttention, self).__init__(config)
        self.self = TTBertSelfAttention(config)
        self.output = TTBertSelfOutput(config)


class TTBertIntermediate(BertIntermediate):
    def __init__(self, config):
        super(TTBertIntermediate, self).__init__(config)
        self.dense = make_tt_linear(self.dense, in_features=768, out_features=3072, shape=[[4, 4, 6, 8], [6, 8, 8, 8]])


class TTBertOutput(BertOutput):
    def __init__(self, config):
        super(TTBertOutput, self).__init__(config)
        self.dense = make_tt_linear(self.dense, in_features=3072, out_features=768, shape=[[6, 8, 8, 8], [4, 4, 6, 8]])


class TTBertLayer(BertLayer):
    def __init__(self, config):
        super(TTBertLayer, self).__init__(config)
        self.attention = TTBertAttention(config)
        self.intermediate = TTBertIntermediate(config)
        self.output = TTBertOutput(config)


class TTBertEncoder(BertEncoder):
    def __init__(self, config):
        super(TTBertEncoder, self).__init__(config)

        def make_layer():
            return TTBertLayer(config)
        self.layer = nn.ModuleList([make_layer() for _ in range(config.num_hidden_layers)])


class TTBertPooler(BertPooler):
    def __init__(self, config):
        super(TTBertPooler, self).__init__(config)
        self.dense = make_tt_linear(self.dense)


class TTEBertModel(BertModel):
    """ Bert with TT embeddings only
    """
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = TTBertEmbeddings(config)


class TTBertModel(BertModel):
    """ Bert with all TT layers
    """
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = TTBertEmbeddings(config)
        self.encoder = TTBertEncoder(config)
        self.pooler = TTBertPooler(config)
        self.apply(self.init_bert_weights)
