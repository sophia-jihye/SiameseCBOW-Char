import pickle
from datetime import datetime, timedelta
import os
import re
import csv
import time
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


def create_dirs(dir_list):
    for directory in dir_list:
        if not os.path.exists(directory):
            os.makedirs(directory)

def write_log(log_dir, fname, content):
    log_filepath = os.path.join(log_dir, fname)
    text_file = open(log_filepath, "w", encoding='utf-8')
    text_file.write(str(content))
    text_file.close()
    print("Log occurred: ", log_filepath)   

def now_time_str():
    return datetime.now().strftime("%Y%m%d-%H-%M-%S")

prog = re.compile('\s+')
def line2words_blank(line):
    words = prog.split(line)
    return words