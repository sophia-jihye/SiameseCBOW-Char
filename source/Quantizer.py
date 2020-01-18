from collections import Counter, namedtuple
from config import parameters
from utils import *
import hangul
import os
import re
import numpy as np
import random as rd

class Quantizer:
    def __init__(self, sentences_of_nouns):
        self.log_dir = parameters.log_dir
        self.max_num_of_unique_words_at_most = parameters.max_num_of_unique_words_at_most
        self.max_len_of_words_in_sentence_at_most = parameters.max_len_of_words_in_sentence_at_most
        self.max_len_of_chars_in_sentence_at_most = parameters.max_len_of_chars_in_sentence_at_most
        self.tokens = self.tokens()
        
        vocab_filepath = parameters.vocab_filepath
        self.idx2word, self.word2idx, self.idx2char, self.char2idx, self.max_len_of_words_in_sentence, self.max_len_of_chars_in_sentence = self.vocab_unpack(vocab_filepath, sentences_of_nouns)
    
    def tokens(self):
        Tokens = namedtuple('Tokens', ['EOS', 'UNK', 'START', 'END', 'ZEROPAD'])
        tokens = Tokens(
                EOS='`',
                UNK='|',    # unk word token
                START='{',  # start-of-word token
                END='}',    # end-of-word token
                ZEROPAD=' ' # zero-pad token
            )
        return tokens
    
    def word2jamo(self, word):
        l = [hangul.split_jamo(char) for char in word]
        sum(l, [])
        jamo_list = list()
        for jamo in sum(l,[]):
            jamo_list.extend(jamo)
        return jamo_list
    
    def create_vocab(self, vocab_filepath, sentences_of_nouns):
        word2idx = {self.tokens.UNK:0}
        char2idx = {self.tokens.UNK:0, self.tokens.START:1, self.tokens.END:2, self.tokens.ZEROPAD:3}
        wordcount = Counter()
        charcount = Counter()
        num_of_words = 0
        max_len_of_words_in_sentence_tmp = 0
        max_len_of_chars_in_sentence_tmp = 0 
        for i, line in enumerate(sentences_of_nouns):
            def update(word, chars_of_word):
                wordcount.update([word])
                charcount.update(chars_of_word)
            
            len_of_chars_in_sentence = 0
            words = line2words_blank(line)
            for word in words:
                chars_of_word = self.word2jamo(word)
                update(word, chars_of_word)
                num_of_words += 1
                len_of_chars_in_sentence += len(chars_of_word)
            max_len_of_words_in_sentence_tmp = max(max_len_of_words_in_sentence_tmp, len(words)) 
            max_len_of_chars_in_sentence_tmp = max(max_len_of_chars_in_sentence_tmp, len_of_chars_in_sentence) 
            
        max_num_of_unique_words = min(self.max_num_of_unique_words_at_most, len(wordcount))
        for ii, ww in enumerate(wordcount.most_common(max_num_of_unique_words)):
            word = ww[0]
            word2idx[word] = ii + 1

        for ii, cc in enumerate(charcount.most_common()):
            char = cc[0]
            char2idx[char] = ii + 4

        max_len_of_words_in_sentence = min(self.max_len_of_words_in_sentence_at_most, max_len_of_words_in_sentence_tmp)
        max_len_of_chars_in_sentence = min(self.max_len_of_chars_in_sentence_at_most, max_len_of_chars_in_sentence_tmp)

        log_content = '\n=====\n# of words (not unique): %d \n# of unique words: %d \n# of unique characters: %d \n=====\nSave Only\n=====\nmax_num_of_unique_words=%d \nmax_len_of_words_in_sentence=%d \nmax_len_of_chars_in_sentence=%d \n=====\n' % (num_of_words, len(wordcount), len(charcount), max_num_of_unique_words, max_len_of_words_in_sentence, max_len_of_chars_in_sentence)
        print(log_content)
        write_log(self.log_dir, 'vocab.log', log_content)

        # save vocab file
        idx2word = dict([(value, key) for (key, value) in word2idx.items()])
        idx2char = dict([(value, key) for (key, value) in char2idx.items()])
        np.savez(vocab_filepath, idx2word=idx2word, word2idx=word2idx, idx2char=idx2char, char2idx=char2idx, max_len_of_words_in_sentence=max_len_of_words_in_sentence, max_len_of_chars_in_sentence=max_len_of_chars_in_sentence)
        print('Created file: ', vocab_filepath)
    
    def vocab_unpack(self, vocab_filepath, sentences_of_nouns):
        if not os.path.exists(vocab_filepath):
            self.create_vocab(vocab_filepath, sentences_of_nouns)
        vocab = np.load(vocab_filepath)
        print('Loaded file: ', vocab_filepath)
        return vocab['idx2word'].item(), vocab['word2idx'].item(), vocab['idx2char'].item(), vocab['char2idx'].item(), vocab['max_len_of_words_in_sentence'].item(), vocab['max_len_of_chars_in_sentence'].item()
    
class EncodedText:
    def __init__(self, df, sentence_length, word2idx, tokens):
        self.sentences_by_document, self.num_of_train_rows = self.sentences_by_document(df)
        self.batch_size = parameters.batch_size
        self.n_positive = parameters.n_positive
        self.n_negative = parameters.n_negative
        self.sentence_length = sentence_length
        self.word2idx = word2idx
        self.tokens = tokens
        rd.seed(42)   # random_seed = 42
        
    def sentences_by_document(self, df):
        doc_ids= df['document_id'].unique()
        sentences_by_document = list()
        for doc_id in doc_ids:
            sentences_by_document.append(df[df['document_id']==doc_id]['nouns'].values)
        num_of_train_rows = len(df) - (2*len(doc_ids))
        return sentences_by_document, num_of_train_rows

    def padding(self, line, sentence_length, unk_idx):
        if len(line) < sentence_length:
            line.extend([unk_idx] * (sentence_length - len(line)))
        else:
            line = line[:sentence_length]
        return line

    def line2bow(self, line):
        line = list(map(lambda x: self.word2idx.get(x, self.word2idx[self.tokens.UNK]), line2words_blank(line)))
        return self.padding(line, sentence_length, self.word2idx[self.tokens.UNK])

    def other_than(self, some_list, inf, sup):
        if inf==0:
            return some_list[sup+1:]
        elif sup==len(some_list)-1:
            return some_list[:inf]
        else:
            return some_list[:inf] + some_list[sup+1:]
    
    def __iter__(self):
        _one_batch = [] 
        pos = []
        neg = []
        batch_y = np.array(([1.0/self.n_positive]*self.n_positive + [0.0]*self.n_negative) * self.batch_size).reshape(self.batch_size, self.n_positive + self.n_negative)
        for one_doc in self.sentences_by_document:
            for t in range(len(one_doc)):
                if t-1 < 0 :   # the first sentence of document
                    continue
                if t + 1 >= len(one_doc):   # the last sentence of document
                    continue

                _one_batch.append(self.line2bow(one_doc[t]))
                pos.append(self.line2bow(one_doc[t-1]))
                pos.append(self.line2bow(one_doc[t+1]))
                for i, n in enumerate(rd.sample(self.other_than(one_doc, t-1, t+1), self.n_negative)):
                    neg.append(self.line2bow(n))
                if len(_one_batch) == self.batch_size:
                    yield ([np.array(_one_batch)]+[np.array(p) for p in pos]+[np.array(n) for n in neg], batch_y)
                    _one_batch = [] 
                    pos = []
                    neg = []
        
        
        
        
        
        