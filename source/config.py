from utils import *
import os
import argparse
import json

#parser = argparse.ArgumentParser()
#parser.add_argument('--bigram', type=lambda x: (str(x).lower() == 'true'), default=True)
#parser.add_argument('--nyt_lag', type=int, default=0, help='New York Times API base quarter lag')
#args = parser.parse_args()

# User configuration
min_df = 0.95
max_num_of_unique_words_at_most = 50000
max_len_of_words_in_sentence_at_most = 5000
max_len_of_chars_in_sentence_at_most = 5000
batch_size = 32
n_positive = 2
n_negative = 5

# System configuration
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print('base_dir:', base_dir)
data_dir = os.path.join(base_dir, 'data')
output_base_dir = os.path.join(base_dir, 'output')
now_time_dir = os.path.join(output_base_dir, now_time_str())
log_dir = os.path.join(now_time_dir, 'log')
create_dirs([output_base_dir, now_time_dir, log_dir])


# filepath
kci_sentences_csv_filepath = os.path.join(data_dir, 'kci_sentences_72832.csv')
parameters_json_filepath = os.path.join(now_time_dir, 'parameters.json')
sentences_of_nouns_csv_filepath = os.path.join(now_time_dir, 'sentences_of_nouns_%d.csv')
vocab_filepath = os.path.join(output_base_dir, 'vocab.npz')
#preprocessed_json_filepath = os.path.join(now_time_dir, 'preprocessed.json')
#calculated_ngram_json_filepath = os.path.join(now_time_dir, 'calculated_ngram.json')
#top_n_ngram_xlsx_filepath = os.path.join(now_time_dir, 'top%d.xlsx' % args.max_display_count)
#top_n_ngram_hits_csv_filepath = os.path.join(now_time_dir, 'top%d_hits.csv' % args.max_display_count)

class Parameters:
    def __init__(self):
        #self.bigram = args.bigram
        
        self.base_dir = base_dir
        self.data_dir = data_dir
        self.output_base_dir = output_base_dir
        self.log_dir = log_dir
        self.min_df = min_df
        self.max_num_of_unique_words_at_most = max_num_of_unique_words_at_most
        self.max_len_of_words_in_sentence_at_most = max_len_of_words_in_sentence_at_most
        self.max_len_of_chars_in_sentence_at_most = max_len_of_chars_in_sentence_at_most
        self.batch_size = batch_size
        self.n_positive = n_positive
        self.n_negative = n_negative
        
        self.kci_sentences_csv_filepath = kci_sentences_csv_filepath
        self.parameters_json_filepath = parameters_json_filepath
        self.sentences_of_nouns_csv_filepath = sentences_of_nouns_csv_filepath
        self.vocab_filepath = vocab_filepath

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