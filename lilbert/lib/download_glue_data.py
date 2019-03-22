# Adapted from https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e.
''' Script for downloading all GLUE data.

Example of usage (to be run on your shell):
python download_glue_data.py --data_dir='../datasets' --tasks='all' --path_to_mrpc='../MRPC'

For downloading MRPC:
1) Download the original MRPC data from (https://download.microsoft.com/download/D/4/6/D46FF87A-F6B9-4252-AA8B-3604ED519838/MSRParaphraseCorpus.msi).
2) For Windows users, run the .msi file.
   For Mac and Linux users, consider an external library such as 'cabextract'.

   wget https://download.microsoft.com/download/D/4/6/D46FF87A-F6B9-4252-AA8B-3604ED519838/MSRParaphraseCorpus.msi
   sudo apt-get install cabextract
   mkdir MRPC
   cabextract MSRParaphraseCorpus.msi -d MRPC
   cat MRPC/_2DEC3DBE877E4DB192D17C0256E90F1D | tr -d $'\r' > MRPC/msr_paraphrase_train.txt
   cat MRPC/_D7B391F9EAFF4B1B8BCE8F21B20B1B61 | tr -d $'\r' > MRPC/msr_paraphrase_test.txt
   rm MRPC/_*
   rm MSRParaphraseCorpus.msi
   
3) Run this script with --path_to_mrpc='../MRPC'.
'''

import os
import sys
import shutil
import argparse
import tempfile
import urllib.request
import zipfile


TASK_TO_PATH = {
    'CoLA': 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FCoLA.zip?alt=media&token=46d5e637-3411-4188-bc44-5809b5bfb5f4',
    'SST':'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8',
    'MRPC':'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2Fmrpc_dev_ids.tsv?alt=media&token=ec5c0836-31d5-48f4-b431-7480817f1adc',
    'QQP':'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQQP.zip?alt=media&token=700c6acf-160d-4d89-81d1-de4191d02cb5',
    'STS':'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSTS-B.zip?alt=media&token=bddb94a7-8706-4e0d-a694-1109e12273b5',
    'MNLI':'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FMNLI.zip?alt=media&token=50329ea1-e339-40e2-809c-10c40afff3ce',
    'SNLI':'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSNLI.zip?alt=media&token=4afcfbb2-ff0c-4b2d-a09a-dbf07926f4df',
    'QNLI':'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQNLI.zip?alt=media&token=c24cad61-f2df-4f04-9ab6-aa576fa829d0',
    'RTE':'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FRTE.zip?alt=media&token=5efa7e85-a0bb-4f19-8ea2-9e1840f077fb',
    'WNLI':'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FWNLI.zip?alt=media&token=068ad0a0-ded7-4bd7-99a5-5e00222e0faf'
}


def format_mrpc(data_dir, path_to_data):
    print('Processing MRPC...')
    mrpc_dir = os.path.join(data_dir, 'MRPC')
    if not os.path.isdir(mrpc_dir):
        os.mkdir(mrpc_dir)
    mrpc_train_file = os.path.join(path_to_data,
                                   'msr_paraphrase_train.txt')
    mrpc_test_file = os.path.join(path_to_data,
                                  'msr_paraphrase_test.txt')
    assert os.path.isfile(mrpc_train_file), 'Train data not found at %s' % mrpc_train_file
    assert os.path.isfile(mrpc_test_file), 'Test data not found at %s' % mrpc_test_file
    urllib.request.urlretrieve(
        TASK_TO_PATH['MRPC'], os.path.join(mrpc_dir, 'dev_ids.tsv'))

    dev_ids = []
    with open(os.path.join(mrpc_dir, 'dev_ids.tsv'),
              encoding='utf-8') as ids_fh:
        for row in ids_fh:
            dev_ids.append(row.strip().split('\t'))

    with open(mrpc_train_file, encoding='utf-8') as data_fh,\
         open(os.path.join(mrpc_dir, 'train.tsv'),
               'w', encoding='utf-8') as train_fh,\
         open(os.path.join(mrpc_dir, 'dev.tsv'),
               'w', encoding='utf-8') as dev_fh:
        header = data_fh.readline()
        train_fh.write(header)
        dev_fh.write(header)
        for row in data_fh:
            label, id1, id2, s1, s2 = row.strip().split('\t')
            if [id1, id2] in dev_ids:
                dev_fh.write('%s\t%s\t%s\t%s\t%s\n' %
                             (label, id1, id2, s1, s2))
            else:
                train_fh.write('%s\t%s\t%s\t%s\t%s\n' %
                               (label, id1, id2, s1, s2))

    with open(mrpc_test_file, encoding='utf-8') as data_fh,\
         open(os.path.join(mrpc_dir, 'test.tsv'),
              'w', encoding='utf-8') as test_fh:
        header = data_fh.readline()
        test_fh.write('index\t#1 ID\t#2 ID\t#1 String\t#2 String\n')
        for idx, row in enumerate(data_fh):
            label, id1, id2, s1, s2 = row.strip().split('\t')
            test_fh.write('%d\t%s\t%s\t%s\t%s\n' %
                          (idx, id1, id2, s1, s2))
    print('\tCompleted!')


def download_and_extract(task, data_dir):
    print('Downloading and extracting {}...'.format(task))
    data_file = '{}.zip'.format(task)
    urllib.request.urlretrieve(TASK_TO_PATH[task], data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(data_file)
    print('\tCompleted!')


def get_tasks(task_names):
    task_names = task_names.split(',')
    if 'all' in task_names:
        tasks = TASK_TO_PATH.keys()
    else:
        tasks = []
        for task_name in task_names:
            assert task_name in TASK_TO_PATH, 'Task {} not found!'.format(task_name)
            tasks.append(task_name)
    return tasks


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        help='directory where to save data',
                        type=str,
                        default='glue_data')
    parser.add_argument('--tasks',
                        help='tasks to download data for',
                        type=str,
                        default='all')
    parser.add_argument('--path_to_mrpc',
                        help='path to MRPC directory with'
                             'msr_paraphrase_train.txt and'
                             'msr_paraphrase_text.txt',
                        type=str,
                        default='')
    args = parser.parse_args(arguments)

    if not os.path.isdir(args.data_dir):
        os.mkdir(args.data_dir)
    tasks = get_tasks(args.tasks)

    for task in tasks:
        if task == 'MRPC':
            format_mrpc(args.data_dir, args.path_to_mrpc)
        else:
            download_and_extract(task, args.data_dir)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
