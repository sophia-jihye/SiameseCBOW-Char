from config import parameters
from utils import *
import os
import re
import pandas as pd
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer

letter_pattern = re.compile('[^ㄱ-ㅣ가-힣a-zA-Z]+')
doublespace_pattern = re.compile('\s+')

class Preprocessor:
    def __init__(self):
        self.okt = Okt()
        
    def __str__(self):
        return os.path.abspath(__file__)
    
    def line2words_nouns(self, line, stopwords=None):
        words = [word for (word, pos) in self.okt.pos(line) if pos in ['Alpha', 'Noun']]
        if stopwords is not None:
            words = [word for word in words if word not in stopwords]
        return words
    
    def stopwords(self, corpus, min_df):
        vectorizer = CountVectorizer(min_df=min_df, max_df=1.0, tokenizer=lambda x:self.line2words_nouns(x))
        X = vectorizer.fit_transform(corpus)
        stopwords = list(vectorizer.vocabulary_.keys())
        return stopwords