import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvModule(nn.Module):
    def __init__(self, input_size, vocab_size, in_channels=100, n_filters=100, kernel_size=5):
        super(ConvModule, self).__init__()
        
        self.n_filters = n_filters
        padding = 0
        if (input_size - kernel_size + 1) % 2 == 1:
            padding = 1
            
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=n_filters, kernel_size=kernel_size, 
                                    stride=1, padding=int(kernel_size / 2))
        
        self.activation = nn.ReLU()
        self.pooling = nn.MaxPool1d(kernel_size=2, padding=padding)
        self.global_pooling = nn.MaxPool1d(kernel_size=input_size)

    def forward(self, x):
        conv = self.activation(self.conv(x))
        pooling = self.pooling(conv)
        global_pooling = self.global_pooling(conv)
        return pooling, global_pooling.view(-1, self.n_filters)
    

class BlendCNN(nn.Module):
    def __init__(self, maxlen, vocab_size, output_dim, emb_dim=100, n_layers=8, 
                 n_filters=100, kernel_size=5, use_embedding=True):
        
        super(BlendCNN, self).__init__()
        
        self.num_labels = output_dim
        self.use_embedding = use_embedding
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        self.maxlen = maxlen 
        
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        
        self.input_sizes = [maxlen]
        self.conv_modules = [ConvModule(maxlen, vocab_size, emb_dim, n_filters, kernel_size).to(device)]
        for _ in range(n_layers):
            self.input_sizes.append(int((self.input_sizes[-1] + 1) / 2))
            self.conv_modules.append(ConvModule(self.input_sizes[-1], vocab_size, n_filters, n_filters, kernel_size).to(device))
        self.conv_modules = torch.nn.ModuleList(self.conv_modules)
        
        self.linear1 = nn.Linear(n_layers * n_filters, 768)
        self.linear2 = nn.Linear(768, 768)
        self.linear3 = nn.Linear(768, output_dim)
        
        
    def forward(self, x, *args, **kwargs):
        
        if self.use_embedding:
            x = self.embeddings(x)
            
        x = x.view(-1, self.emb_dim, self.maxlen)
        
        all_global_pools = []
        
        for i in range(self.n_layers):
            x, global_pool = self.conv_modules[i](x)
            all_global_pools.append(global_pool)
            
        global_cat = torch.cat(all_global_pools, dim=-1)
        
        linear1 = nn.ReLU()(self.linear1(global_cat))
        linear2 = nn.ReLU()(self.linear2(linear1))
        logits = self.linear3(linear2)

        return logits

