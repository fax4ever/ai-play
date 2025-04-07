#!/usr/bin/env python
import os, shutil, wget, torch
from dataset import Word2VecDataset
from simple_skipgram import SimpleSkipGram
from skipgram import SkipGram
from trainer import Trainer

VOCAB_SIZE = 4_000
UNK = 'UNK' # the token to be used for out of vocabulary words
FILE_NAME = 'wiki.10K.txt'
NEGATIVE_SAMPLING = True

def main():
    torch.manual_seed(42)
    output_folder = os.path.join('output_folder/{}/'.format('neg_sampling' if NEGATIVE_SAMPLING else 'no_sampling'))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if os.path.isfile(FILE_NAME):
        train_data_path = FILE_NAME
    else:
        train_data_path = wget.download('https://raw.githubusercontent.com/dbamman/anlp19/master/data/' + FILE_NAME)

    dataset = Word2VecDataset(train_data_path, VOCAB_SIZE, UNK, 5)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if NEGATIVE_SAMPLING:
        model = SkipGram(VOCAB_SIZE, 300, dataset.id2word, dataset.word2frequency)
    else:
        model = SimpleSkipGram(VOCAB_SIZE, 300)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model, optimizer, device)
    trainer.train(dataloader, output_folder)

if __name__ == "__main__":
    main()