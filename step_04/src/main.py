import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

import os


OUTPUT_DIR = '/output'
NUM_WORDS = 10000

(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)

def multi_hot_sequences(sequences, dimension):
    # 形状が (len(sequences), dimension)ですべて0の行列を作る
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0  # 特定のインデックスに対してresults[i] を１に設定する
    return results


train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

plt.plot(train_data[0])
plt.savefig(os.path.join(OUTPUT_DIR, 'check.png'))


def main():
    baseline_model = keras.Sequential([
        # `.summary` を見るために`input_shape`が必要 
        keras.layers.Dense(16, activation='relu', input_shape=(NUM_WORDS,)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    baseline_model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy', 'binary_crossentropy'])

    baseline_model.summary()


if __name__ == '__main__':
    main()
