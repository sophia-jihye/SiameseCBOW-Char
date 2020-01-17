from collections import Counter, namedtuple
from config import parameters
from utils import *
import hangul
import os
import re
import numpy as np

class Quantizer:
    def __init__(self, sentences_of_nouns):
        self.log_dir = parameters.log_dir
        self.max_num_of_unique_words_at_most = parameters.max_num_of_unique_words_at_most
        self.max_sentence_length_at_most = parameters.max_sentence_length_at_most
        self.prog = re.compile('\s+')
        self.tokens = self.tokens()
        
        vocab_filepath = parameters.vocab_filepath
        self.idx2word, self.word2idx, self.idx2char, self.char2idx, self.max_sentence_length = self.vocab_unpack(vocab_filepath, sentences_of_nouns)
    
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
    
    def line2words_blank(self, line):
        words = self.prog.split(line)
        return words
    
    def create_vocab(self, vocab_filepath, sentences_of_nouns):
        word2idx = {self.tokens.UNK:0}
        char2idx = {self.tokens.ZEROPAD:0, self.tokens.START:1, self.tokens.END:2, self.tokens.UNK:3}
        wordcount = Counter()
        charcount = Counter()
        num_of_words = 0
        max_sentence_length_tmp = 0 
        for i, line in enumerate(sentences_of_nouns):
            def update(word, chars_of_word):
                wordcount.update([word])
                charcount.update(chars_of_word)
            
            len_of_chars_in_sentence = 0
            words = self.line2words_blank(line)
            for word in words:
                chars_of_word = self.word2jamo(word)
                update(word, chars_of_word)
                num_of_words += 1
                len_of_chars_in_sentence += len(chars_of_word)
            max_sentence_length_tmp = max(max_sentence_length_tmp, len_of_chars_in_sentence) 
            
        max_num_of_unique_words = min(self.max_num_of_unique_words_at_most, len(wordcount))
        for ii, ww in enumerate(wordcount.most_common(max_num_of_unique_words)):
            word = ww[0]
            word2idx[word] = ii + 1

        for ii, cc in enumerate(charcount.most_common()):
            char = cc[0]
            char2idx[char] = ii + 4

        print('After first pass of data, max sentence length is:', max_sentence_length_tmp)
        max_sentence_length = min(self.max_sentence_length_at_most, max_sentence_length_tmp)

        log_content = '# of words (not unique): %d \n# of unique words: %d \n# of unique characters: %d \n\n\n=====\nSave Only\n=====\nmax_num_of_unique_words=%d \nmax_sentence_length=%d' % (num_of_words, len(wordcount), len(charcount), max_num_of_unique_words, max_sentence_length)
        print(log_content)
        write_log(self.log_dir, 'vocab.log', log_content)

        # save vocab file
        idx2word = dict([(value, key) for (key, value) in word2idx.items()])
        idx2char = dict([(value, key) for (key, value) in char2idx.items()])
        np.savez(vocab_filepath, idx2word=idx2word, word2idx=word2idx, idx2char=idx2char, char2idx=char2idx, max_sentence_length=max_sentence_length)
        print('Created file: ', vocab_filepath)
    
    def vocab_unpack(self, vocab_filepath, sentences_of_nouns):
        if not os.path.exists(vocab_filepath):
            self.create_vocab(vocab_filepath, sentences_of_nouns)
        vocab = np.load(vocab_filepath)
        print('Loaded file: ', vocab_filepath)
        return vocab['idx2word'], vocab['word2idx'], vocab['idx2char'], vocab['char2idx'], vocab['max_sentence_length']