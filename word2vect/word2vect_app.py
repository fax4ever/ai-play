#!/usr/bin/env python
import os

import dataset
import wget

VOCAB_SIZE = 4_000
UNK = 'UNK' # the token to be used for out of vocabulary words
FILE_NAME = 'wiki.10K.txt'

def main():
    if os.path.isfile(FILE_NAME):
        train_data_path = FILE_NAME
    else:
        train_data_path = wget.download('https://raw.githubusercontent.com/dbamman/anlp19/master/data/' + FILE_NAME)
    result = dataset.Word2VecDataset(train_data_path, VOCAB_SIZE, UNK, 5)
    for i in result:
        print(i)

if __name__ == "__main__":
    main()