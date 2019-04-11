from lib import data_processors

processors = {
    'cola': data_processors.ColaProcessor,
    'mnli': data_processors.MnliProcessor,
    'mrpc': data_processors.MrpcProcessor,
    'sst2': data_processors.SST2Processor,
    'qqp': data_processors.QQPProcessor,
    'swag': data_processors.SWAGProcessor,
}

num_labels = {
    'cola': 2,
    'mnli': 3,
    'mrpc': 2,
    'sst2': 2,
    'qqp': 2
}

label_lists = {
    task_name: [str(label)
                for label in range(num_labels[task_name])]
    for task_name in num_labels.keys()
}
