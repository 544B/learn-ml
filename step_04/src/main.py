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
plt.clf()


def get_baseline_model_history():
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
    return baseline_model.fit(train_data,
                train_labels,
                epochs=20,
                batch_size=512,
                validation_data=(test_data, test_labels),
                verbose=2)


def get_smaller_model_history():
    smaller_model = keras.Sequential([
        keras.layers.Dense(4, activation='relu', input_shape=(NUM_WORDS,)),
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    smaller_model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy', 'binary_crossentropy'])

    smaller_model.summary()
    return smaller_model.fit(train_data,
                train_labels,
                epochs=20,
                batch_size=512,
                validation_data=(test_data, test_labels),
                verbose=2)


def get_bigger_model_history():
    bigger_model = keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(NUM_WORDS,)),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    bigger_model.compile(optimizer='adam',
                         loss='binary_crossentropy',
                         metrics=['accuracy','binary_crossentropy'])

    bigger_model.summary()

    return bigger_model.fit(train_data, train_labels,
                epochs=20,
                batch_size=512,
                validation_data=(test_data, test_labels),
                verbose=2)


def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16,10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')

        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()
    plt.xlim([0,max(history.epoch)])

    plt.savefig(os.path.join(OUTPUT_DIR, 'history_comparison.png'))
    plt.clf()


def main():
    baseline_history = get_baseline_model_history()
    smaller_history = get_smaller_model_history()
    bigger_history = get_bigger_model_history()

    plot_history([
        ('baseline', baseline_history),
        ('smaller', smaller_history),
        ('bigger', bigger_history)
    ])


if __name__ == '__main__':
    main()
