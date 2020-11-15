import tensorflow as tf
from tensorflow import keras

import numpy as np


imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))


def main():
    print(tf.__version__)


if __name__ == '__main__':
    main()
