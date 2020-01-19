import pandas as pd
import os 
import numpy as np
import pickle
from config import parameters
from utils import *

class SentenceHandler:
    def __init__(self):
        self.word2idx = self.word2idx(parameters.vocab_filepath)
        self.embedding = self.embedding(parameters.model_embedding_vectors_pkl_filepath)

    def word2idx(self, filepath):
        vocab = np.load(filepath)
        print('Loaded: %s' % filepath)
        return vocab['word2idx'].item()
    
    def embedding(self, filepath):
        with open(filepath, 'rb') as f:
            embedding = pickle.load(f)
        print('Loaded: %s' % filepath)
        return embedding
    
    def word_vector(self, word_str):
        word_idx = self.word2idx[word_str]
        return self.embedding[word_idx, :]
    
    def words2averaged_vector(self, words):
        vectors = []
        for word in words:
            vectors.append(self.word_vector(word))
        return np.mean(vectors, axis=0)