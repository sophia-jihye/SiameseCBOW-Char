from config import parameters
from utils import *
import pandas as pd
import json
import math
from Preprocessor import Preprocessor
from Quantizer import Quantizer
from Quantizer import EncodedText
from SiameseCBOW import SiameseCBOW

kci_sentences_csv_filepath = parameters.kci_sentences_csv_filepath
sentences_of_nouns_csv_filepath = parameters.sentences_of_nouns_csv_filepath
model_embedding_vectors_pkl_filepath = parameters.model_embedding_vectors_pkl_filepath

min_df = parameters.min_df
siamese_embedding_dimension = parameters.siamese_embedding_dimension
epochs = parameters.epochs

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
    print('Created file:', sentences_of_nouns_csv_filepath % len(df))
    return df

def main():
    df = pd.read_csv(kci_sentences_csv_filepath)
    
    preprocessor = Preprocessor()
    df = raw2preprocessed(preprocessor, df)
    
    quantizer = Quantizer(df['nouns'].values)
    # quantizer.max_len_of_words_in_sentence
    encoded_text = EncodedText(df[['document_id', 'nouns']], quantizer.max_len_of_words_in_sentence, quantizer.word2idx, quantizer.tokens)
    steps_per_epoch = math.trunc(encoded_text.num_of_train_rows / encoded_text.batch_size)
    
    model = SiameseCBOW(len(encoded_text.word2idx), siamese_embedding_dimension, encoded_text.sentence_length, encoded_text.n_positive, encoded_text.n_negative)
    model.fit_generator(iter(encoded_text), steps_per_epoch, epochs)
    
    # save as .pkl
    model.save_embedding_vectors(model_embedding_vectors_pkl_filepath)
    print('Created file:', model_embedding_vectors_pkl_filepath)
    
if __name__ == '__main__':
    main()