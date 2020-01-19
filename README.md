# SiameseCBOW-Char
Implementation of Sentence Embedding using Character Level Encoding (English+Korean)

## Reference

* https://github.com/shinochin/SiameseCBOW
* https://github.com/jarfo/kchar
* https://github.com/lovit/soynlp/blob/master/soynlp/hangle/_hangle.py

## Development Environment
* Ubuntu 16.04.6 LTS
* Python 3.7.5
* Keras 2.1.2
* Tensorflow 1.13.1

## Folder structure
The following shows basic folder structure.
```
├── source
│   ├── main.py
│   ├── config.py
│   ├── utils.py
│   ├── ...
├── data
│   ├──  kci_sentences_72832.csv
├── output
│   ├── vocab.npz
│   ├── vocab.log
│   ├── 20200118-14-00-44
│        ├── parameters.json
│        ├── sentences_of_nouns_72832.csv
│        ├── log