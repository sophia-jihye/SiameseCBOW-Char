from config import parameters
from utils import *
import pandas as pd
import json
from Preprocessor import Preprocessor
from Quantizer import Quantizer

kci_sentences_csv_filepath = parameters.kci_sentences_csv_filepath
sentences_of_nouns_csv_filepath = parameters.sentences_of_nouns_csv_filepath

min_df = parameters.min_df

def raw2preprocessed(preprocessor, df):
    # stopwords
    print('Extracting stopwords ..')
    corpus_of_docs = list()
    for doc_id in df.groupby(['document_id']).groups.keys():
        corpus_of_docs.append(' '.join(df[df['document_id']==doc_id]['sentence'].values))
    stopwords = preprocessor.stopwords(corpus_of_docs, min_df)
    print('# of stopwords = %d \n%s' % (len(stopwords), stopwords))

    # extract nouns
    print('Extracting nouns ..')
    df['nouns'] = df.apply(lambda x: ' '.join(preprocessor.line2words_nouns(x['sentence'], stopwords)), axis=1)
    
    # save as .csv
    df.to_csv(os.path.join(sentences_of_nouns_csv_filepath % len(df) ), index=False)    
    print('Created file:', sentences_of_nouns_csv_filepath)
    return df

def main():
    df = pd.read_csv(kci_sentences_csv_filepath)
    
    preprocessor = Preprocessor()
    df = raw2preprocessed(preprocessor, df)
    
    quantizer = Quantizer(df['nouns'].values)
    
if __name__ == '__main__':
    main()