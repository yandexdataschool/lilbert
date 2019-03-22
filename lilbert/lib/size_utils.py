import os
import torch


def convert_bytes(file_size):
    for size_metrics in ['bytes', 'KB', 'MB', 'GB']:
        if file_size < 1024.0 or size_metrics == 'GB':
            return '{} {}'.format(file_size, size_metrics)
        file_size /= 1024.0


def get_file_size(file_path):
    if os.path.isfile(file_path):
        return convert_bytes(os.stat(file_path).st_size)


def get_model_size(model, cache_dir):
    file_to_save = os.path.join(cache_dir,
                                'model_state_temp.pth')
    torch.save(model.state_dict(),
               file_to_save)
    return get_file_size(file_to_save)
