# Original Reference: https://github.com/lovit/soynlp/blob/master/soynlp/hangle/_hangle.py

import numpy as np
import re

kor_begin     = 44032
kor_end       = 55203
chosung_base  = 588
jungsung_base = 28
jaum_begin = 12593
jaum_end = 12622
moum_begin = 12623
moum_end = 12643

chosung_list = [ 'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 
        'ㅅ', 'ㅆ', 'ㅇ' , 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

jungsung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 
        'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 
        'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 
        'ㅡ', 'ㅢ', 'ㅣ']

jongsung_list = [
    ' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ',
        'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 
        'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 
        'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

jaum_list = ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄸ', 'ㄹ', 
              'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 
              'ㅃ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

moum_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 
              'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

double_jaum_moum_dict = {'ㄲ': ['ㄱ', 'ㄱ'], 'ㄸ': ['ㄷ', 'ㄷ'], 'ㅃ': ['ㅂ', 'ㅂ'], 'ㅆ': ['ㅅ', 'ㅅ'], 'ㅉ': ['ㅈ', 'ㅈ'], 'ㄳ': ['ㄱ', 'ㅅ'], 'ㄵ': ['ㄴ', 'ㅈ'], 'ㄶ': ['ㄴ', 'ㅎ'], 'ㄺ': ['ㄹ', 'ㄱ'], 'ㄻ': ['ㄹ', 'ㅁ'], 'ㄼ': ['ㄹ', 'ㅂ'], 'ㄽ': ['ㄹ', 'ㅅ'], 'ㄾ': ['ㄹ', 'ㅌ'], 'ㄿ': ['ㄹ', 'ㅍ'], 'ㅀ': ['ㄹ', 'ㅎ'], 'ㅄ': ['ㅂ', 'ㅅ'], 'ㅐ': ['ㅏ', 'ㅣ'], 'ㅒ': ['ㅑ', 'ㅣ'], 'ㅔ': ['ㅓ', 'ㅣ'], 'ㅖ': ['ㅕ', 'ㅣ'], 'ㅘ': ['ㅗ', 'ㅏ'], 'ㅙ': ['ㅗ', 'ㅏ', 'ㅣ'], 'ㅚ': ['ㅗ', 'ㅣ'], 'ㅝ': ['ㅜ', 'ㅓ'], 'ㅞ': ['ㅜ', 'ㅓ', 'ㅣ'], 'ㅟ': ['ㅜ', 'ㅣ'], 'ㅢ': ['ㅡ', 'ㅣ']}

doublespace_pattern = re.compile('\s+')
repeatchars_pattern = re.compile('(\w)\\1{3,}')

def normalize(doc, english=False, number=False, punctuation=False, remove_repeat = 0, remains={}):
    if remove_repeat > 0:
        doc = repeatchars_pattern.sub('\\1' * remove_repeat, doc)

    f = ''    
    for c in doc:
        i = ord(c)
        
        if (c == ' ') or (is_korean(i)) or (english and is_english(i)) or (number and is_number(i)) or (punctuation and is_punctuation(i)):
            f += c            
        elif c in remains:
            f += c        
        else:
            f += ' '
            
    return doublespace_pattern.sub(' ', f).strip()

def doublejamo2jamo(item):
    if item in double_jaum_moum_dict:
        item = double_jaum_moum_dict[item]
    return item

def split_jamo(c):    
    i = ord(c)
    if not is_korean(i):
        return [c]
    elif is_jaum(i):
        c = doublejamo2jamo(c)
        return [c]
    elif is_moum(i):
        c = doublejamo2jamo(c)
        return [c]
    
    i -= kor_begin
    
    cho  = i // chosung_base
    jung = ( i - cho * chosung_base ) // jungsung_base 
    jong = ( i - cho * chosung_base - jung * jungsung_base )
    
    chosung = chosung_list[cho]
    jungsung = jungsung_list[jung]
    jongsung = jongsung_list[jong]
    chosung = doublejamo2jamo(chosung)
    jungsung = doublejamo2jamo(jungsung)
    jongsung = doublejamo2jamo(jongsung)
    result = [chosung, jungsung, jongsung]
    result = [item for item in result if item != ' ']
    return result

def is_korean(i):
    i = to_base(i)
    return (kor_begin <= i <= kor_end) or (jaum_begin <= i <= jaum_end) or (moum_begin <= i <= moum_end)

def is_number(i):
    i = to_base(i)
    return (i >= 48 and i <= 57)

def is_english(i):
    i = to_base(i)
    return (i >= 97 and i <= 122) or (i >= 65 and i <= 90)

def is_punctuation(i):
    i = to_base(i)
    return (i == 33 or i == 34 or i == 39 or i == 44 or i == 46 or i == 63 or i == 96)

def is_jaum(i):
    i = to_base(i)
    return (jaum_begin <= i <= jaum_end)

def is_moum(i):
    i = to_base(i)
    return (moum_begin <= i <= moum_end)

def to_base(c):
    if type(c) == str:
        return ord(c)
    elif type(c) == int:
        return c
    else:
        raise TypeError

def combine_jamo(chosung, jungsung, jongsung):
    return chr(kor_begin + chosung_base * chosung_list.index(chosung) + jungsung_base * jungsung_list.index(jungsung) + jongsung_list.index(jongsung))


class ConvolutionalNN_Encoder:
        
    def __init__(self, vocabs={}):
        self.vocabs = vocabs
        
        self.jungsung_hot_begin = 31
        self.jongsung_hot_begin = 52
        self.symbol_hot_begin = 83

        self.cvocabs_ = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄸ', 'ㄹ', 
                   'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅃ', 
                   'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 
                   'ㅎ', 'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ',
                   'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 
                   'ㅢ', 'ㅣ']
        self.cvocabs = {}
        self.cvocabs = {c:i for i, c in enumerate(self.cvocabs_)}

        # svocabs_ = ['.',  ',',  '?',  '!',  '-', ':', 
        #            '0',  '1',  '2',  '3',  '4',  '5',  '6',  '7',  '8',  '9']
        # svocabs = {}
        # svocabs = {s:len(svocabs) + symbol_hot_begin for s in svocabs_}


    def encode_vocab(self, words, unknown=-1, blank=0, input_length=64):
        if len(words) > input_length:
            words = words[:input_length]
        return [self.vocabs[w] if w in self.vocabs else unknown for w in words] + [blank] * (input_length - len(words))