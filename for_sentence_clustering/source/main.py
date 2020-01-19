from config import parameters
from utils import *
import pandas as pd
import json
import math
from SentenceHandler import SentenceHandler

sentences_of_nouns_csv_filepath = parameters.sentences_of_nouns_csv_filepath
sentence_averaged_json_filepath = parameters.sentence_averaged_json_filepath

def main():
    df = pd.read_csv(sentences_of_nouns_csv_filepath)
    
    sentence_handler = SentenceHandler()
    df['averaged_vector'] = df.apply(lambda x: sentence_handler.words2averaged_vector(line2words_blank(x['nouns'])), axis=1)
    
    # save as .json
    with open(sentence_averaged_json_filepath, 'w+') as json_file:
        json.dump(df.to_json(orient='records'), json_file)
        print('Created file:', sentence_averaged_json_filepath)
    
if __name__ == '__main__':
    main()