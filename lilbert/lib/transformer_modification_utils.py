import functools
from tqdm import tqdm_notebook as tqdm


def rsetattr(obj, attr, val):
    """
    Nested setattr function
    """
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    """
    Nested getattr function
    """
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def replace_transformer_layers(model,
                               NewLayer,
                               blocks=range(12),
                               block_parts=['attention.self.query',
                                            'attention.self.key',
                                            'attention.self.value',
                                            'attention.output.dense',
                                            'intermediate.dense',
                                            'output.dense'],
                               *args, **kwargs
                               ):
    """
    Takes model and replace layers at given blocks at given block parts with a
    new layer returned by NewLayer(*args, **kwargs) class of function
    Input: model -- model with bert.encoder.layer[*] module
           NewLayer -- class or function that returns a layer
             to replace given current layers, takes a layer to be modified
             as a first parameter and *args, **kwargs as the rest parameters
           blocks -- list or range of blocks to be modified
           block_parts -- list of block layers to be modified
           *args, **kwargs -- arguments of NewLayer

    """

    for transformer_layer_ind in tqdm(blocks):
        block = model.bert.encoder.layer[transformer_layer_ind]
        for layer in block_parts:
            rsetattr(block,
                     layer,
                     NewLayer(
                         rgetattr(block, layer),
                         *args, **kwargs)
                     )
