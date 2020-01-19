from utils import *
import os
import argparse
import json

# User configuration


# System configuration
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print('base_dir:', base_dir)
data_dir = os.path.join(base_dir, 'data')
output_base_dir = os.path.join(base_dir, 'output')
now_time_dir = os.path.join(output_base_dir, now_time_str())
parameters_json_filepath = os.path.join(now_time_dir, 'parameters.json')
log_dir = os.path.join(now_time_dir, 'log')
create_dirs([output_base_dir, now_time_dir, log_dir])

# filepath
sentences_of_nouns_csv_filepath = os.path.join(data_dir, 'sentences_of_nouns_72832.csv')
vocab_filepath = os.path.join(data_dir, 'vocab.npz')
model_embedding_vectors_pkl_filepath = os.path.join(data_dir, 'model_embedding_vectors.pkl')
sentence_averaged_json_filepath = os.path.join(now_time_dir, 'sentence_averaged.json')

class Parameters:
    def __init__(self):
        self.base_dir = base_dir
        self.data_dir = data_dir
        self.output_base_dir = output_base_dir
        self.parameters_json_filepath = parameters_json_filepath
        self.log_dir = log_dir
        
        self.sentences_of_nouns_csv_filepath = sentences_of_nouns_csv_filepath
        self.vocab_filepath = vocab_filepath
        self.model_embedding_vectors_pkl_filepath = model_embedding_vectors_pkl_filepath
        self.sentence_averaged_json_filepath = sentence_averaged_json_filepath

    def __str__(self):
        item_strf = ['{} = {}'.format(attribute, value) for attribute, value in self.__dict__.items()]
        strf = 'Parameters(\n  {}\n)'.format('\n  '.join(item_strf))
        return strf
    
    def save(self):
        with open(self.parameters_json_filepath, 'w+') as json_file:
            json.dump(self.__dict__, json_file)
        print('Created file:', self.parameters_json_filepath)

parameters = Parameters()
parameters.save()
print(parameters)