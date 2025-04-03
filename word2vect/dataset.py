import collections
import re

import numpy as np
import torch


def read_data(txt_path):
    data = []
    with open(txt_path) as file:
        for line in file:
            split = tokenize_line(line)
            if split:
                # split is a list of words which is appended to data
                data.append(split)
    return data


def tokenize_line(line, pattern='\W'):
    return [word for word in re.split(pattern, line.lower()) if word]


class Word2VecDataset(torch.utils.data.IterableDataset):

    def __init__(self, txt_path, max_vocab_size, unk_token, window_size):
        self.data_words = read_data(txt_path)
        self.build_vocabulary(max_vocab_size, unk_token)
        self.window_size = window_size

    def __iter__(self):
        for sentence in self.data_words:
            for position_in_sentence, word in enumerate(sentence):
                if not (word in self.word2id) or self.skip_word(word):
                    continue

                word_id = self.word2id[word]
                position_left = max(0, position_in_sentence - self.window_size)
                position_right = min(len(sentence), position_in_sentence + self.window_size)
                window_positions = [x for x in range(position_left, position_right) if x != position_in_sentence]
                for position in window_positions:
                    if not (sentence[position] in self.word2id):
                        # must be a word in the vocabulary
                        continue

                    # index of target word in vocab
                    related_word_id = self.word2id[sentence[position]]
                    output_dict = {'inputs': word_id, 'targets': related_word_id}
                    # return a single association (word -> related word) in the window
                    # done for each word in the window
                    yield output_dict
        pass

    def build_vocabulary(self, max_vocab_size, unk_token):
        counter_list = []
        # context is a list of tokens within a single sentence
        for context in self.data_words:
            counter_list.extend(context)

        counter = collections.Counter(counter_list)
        # word -> id
        self.word2id = {key: index for index, (key, _) in enumerate(counter.most_common(max_vocab_size - 1))}
        assert unk_token not in self.word2id
        self.word2id[unk_token] = max_vocab_size - 1
        # word -> frequency
        self.word2frequency = {x: counter[x] for x in self.word2id if x is not unk_token}

        self.tot_occurrences = sum(self.word2frequency[x] for x in self.word2frequency)
        # id -> word
        self.id2word = {value: key for key, value in self.word2id.items()}
        # sentence -> ids
        self.data_idx = []
        for sentence in self.data_words:
            paragraph = []
            # for each word in the sentence
            for word in sentence:
                id_ = self.word2id[word] if word in self.word2id else self.word2id[unk_token]
                if id_ == self.word2id[unk_token]:
                    continue
                paragraph.append(id_)
            self.data_idx.append(paragraph)

    def skip_word(self, word):
        z = self.word2frequency[word] / self.tot_occurrences  # f(w): relative frequency
        t = 1e-5  # standard value used in practice
        # p_keep: higher for less frequent instances
        p_keep = np.sqrt(t / z)
        p_keep = min(1.0, p_keep)  # cap at 1
        return np.random.rand() >= p_keep
