from lib import data_processors

processors = {
    'cola': data_processors.ColaProcessor,
    'mnli': data_processors.MnliProcessor,
    'mrpc': data_processors.MrpcProcessor,
    'sst2': data_processors.SST2Processor,
    'qqp': data_processors.QQPProcessor,
    'sstb': data_processors.StsbProcessor,
    'rte': data_processors.RteProcessor,
    'qnli': data_processors.QnliProcessor,
}

num_labels = {
    'cola': 2,
    'mnli': 3,
    'mrpc': 2,
    'sst2': 2,
    'qqp': 2,
    'sstb': 1,
    'rte': 2,
    'qnli': 2
}

label_lists = {
    task_name: [str(label)
                for label in range(num_labels[task_name])]
    for task_name in num_labels.keys()
}

data_dirs = {
    'cola': '../datasets/COLA',
    'mnli': '../datasets/MNLI',
    'mrpc': '../datasets/MRPC',
    'sst2': '../datasets/SST-2',
    'qqp': '../datasets/QQP',
    'sstb': '../datasets/STS-B',
    'rte': '../datasets/RTE',
    'qnli': '../datasets/QNLI'
}
