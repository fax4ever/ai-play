#!/usr/bin/env python

import dataset
import wget

VOCAB_SIZE = 4_000
UNK = 'UNK' # the token to be used for out of vocabulary words

def main():
    train_data_path = wget.download('https://raw.githubusercontent.com/dbamman/anlp19/master/data/wiki.10K.txt')
    result = dataset.Word2VecDataset(train_data_path, VOCAB_SIZE, UNK, 5)
    for i in result:
        print(i)

if __name__ == "__main__":
    main()