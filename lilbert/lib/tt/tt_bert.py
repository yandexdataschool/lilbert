from pytorch_pretrained_bert.modeling import *
from lib.tt.tt_bert_components import TTEBertModel, TTBertModel


class TTEBertForPreTraining(BertForPreTraining):
    def __init__(self, config):
        super(TTEBertForPreTraining, self).__init__(config)
        self.bert = TTEBertModel(config)


class TTEBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config, num_labels):
        super(TTEBertForSequenceClassification, self).__init__(config, num_labels)
        self.bert = TTEBertModel(config)


class TTBertForPreTraining(BertForPreTraining):
    def __init__(self, config):
        super(TTBertForPreTraining, self).__init__(config)
        self.bert = TTBertModel(config)


class TTBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config, num_labels):
        super(TTBertForSequenceClassification, self).__init__(config, num_labels)
        self.bert = TTBertModel(config)


