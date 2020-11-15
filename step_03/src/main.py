import tensorflow as tf
from tensorflow import keras

import numpy as np


imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))


def decode_review(text):
    # 単語を整数にマッピングする辞書
    word_index = imdb.get_word_index()
    # インデックスの最初の方は予約済み
    word_index = {k:(v+3) for k,v in word_index.items()} 
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


def main():
    print(train_data[0])
    print(decode_review(train_data[0]))


if __name__ == '__main__':
    main()
