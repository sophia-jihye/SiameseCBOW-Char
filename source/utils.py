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
            
def load_pkl(filepath):
    with open(filepath, 'rb') as f:
        current_pkl = pickle.load(f)
    print('Completed loading:', filepath)
    return current_pkl


def end_pkl(target_to_save, pkl_path, start=None):
    with open(pkl_path, 'wb') as f:
        pickle.dump(target_to_save, f)
    print('Creating .pkl completed: ', pkl_path)

    if start is not None:
        elapsed_time = time.time() - start
        elapsed_time_format = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print('END. Elapsed time: ', elapsed_time_format)


def now_time_str():
    return datetime.now().strftime("%Y%m%d-%H-%M-%S")


def get_str_concat(*args):
    _str = ""
    firstLine = True
    for idx, arg in enumerate(args):
        if idx == 0:
            _str += arg
            firstLine = False
            continue
        _str = _str + "_" + arg
    return _str


def dates_between(start_dt, end_dt):
    dates = []
    cursor = start_dt
    while cursor <= end_dt:
        if cursor.strftime('%Y-%m-%d') not in dates:
            dates.append(cursor.strftime('%Y-%m-%d'))
        cursor += timedelta(days=1)
    return dates


def _period_dict(target_dict, _period, start, end):
    for _month in [str('%02d' % i) for i in range(start, end + 1)]:
        target_dict[_month] = _period
    return target_dict

def monthly_dict():
    target_dict = dict()
    for i in range(1, 12+1):
        target_dict = _period_dict(target_dict, str(i), i, i)
    return target_dict

def quarterly_dict():
    target_dict = dict()
    target_dict = _period_dict(target_dict, 'Q1', 1, 3)
    target_dict = _period_dict(target_dict, 'Q2', 4, 6)
    target_dict = _period_dict(target_dict, 'Q3', 7, 9)
    target_dict = _period_dict(target_dict, 'Q4', 10, 12)
    return target_dict


def semiannual_dict():
    target_dict = dict()
    target_dict = _period_dict(target_dict, '1H', 1, 6)
    target_dict = _period_dict(target_dict, '2H', 7, 12)
    return target_dict


def annual_dict():
    target_dict = dict()
    target_dict = _period_dict(target_dict, '', 1, 12)
    return target_dict


def dict_val_as_list_append(target_dict, index_key, val):
    if index_key not in target_dict:
        target_dict[index_key] = list()
    target_dict[index_key].append(val)
    return target_dict


def bow_unigram_freq_dict(word_list, stopset=None):
    vectorizer = CountVectorizer(stop_words=stopset)
    X = vectorizer.fit_transform(word_list)
    terms = vectorizer.get_feature_names()
    freqs = X.sum(axis=0).A1
    unigram_bow_dict = dict(zip(terms, freqs))
    return unigram_bow_dict


def get_0_if_None(content):
    if content is None:
        return 0
    return content


def start_csv(csv_filepath, csv_delimiter=','):
    f = open(csv_filepath, 'w', encoding='utf-8-sig', newline='')
    wr = csv.writer(f, delimiter=csv_delimiter)
    return f, wr


def end_csv(f, csv_filepath):
    f.close()
    print('Creating .csv file completed: ', csv_filepath)

def get_index_dict(fred_gdp_index_dict_pkl_filepath, fred_index_filepath_dict):
    if os.path.exists(fred_gdp_index_dict_pkl_filepath):
        index_dict = load_pkl(fred_gdp_index_dict_pkl_filepath)
    else:
        index_dict = dict()

        for period_category, index_data_filepath in fred_index_filepath_dict.items():
            index_dict[period_category] = dict()

            f = open(index_data_filepath, 'r', encoding='utf-8')
            rdr = csv.reader(f)
            firstline = True
            for line in rdr:
                if firstline:
                    firstline = False
                    continue
                _period = line[0]
                if _period == '':
                    continue
                _val = line[1]
                index_dict[period_category][_period] = _val
            f.close()

        end_pkl(index_dict, fred_gdp_index_dict_pkl_filepath)
    return index_dict

def rescale_(values, scaler):
    values = values.reshape((len(values), 1))
    scaler = scaler.fit(values)
    normalized = scaler.transform(values)
    return normalized