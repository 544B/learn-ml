import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

import os

OUTPUT_DIR = '/output'


imdb = keras.datasets.imdb
# 単語を整数にマッピングする辞書
word_index = imdb.get_word_index()
# インデックスの最初の方は予約済み
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3


def decode_review(text):
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


def setup_data(data):
    return keras.preprocessing.sequence.pad_sequences(
            data,
            value=word_index["<PAD>"],
            padding='post',
            maxlen=256)


def get_model(vocab_size=10000):
    model = keras.Sequential()

    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    return model


def gen_training_loss_graph(epochs, loss, val_loss):
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_loss.png'))
    plt.clf()


def gen_training_acc_graph(epochs, acc, val_acc):
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_acc.png'))
    plt.clf()


def main():
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

    print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
    print("Testdata entries: {}, labels: {}".format(len(test_data), len(test_labels)))

    print('row: {}'.format(train_data[0]))
    print('decode: {}'.format(decode_review(train_data[0])))

    print('Before: {},{}'.format(len(train_data[0]), len(train_data[1])))

    train_data = setup_data(train_data)
    test_data = setup_data(test_data)
    print('After: {},{}'.format(len(train_data[0]), len(train_data[1])))
    print('after-decode: {}'.format(decode_review(train_data[0])))

    vocab_size = 10000
    model = get_model(vocab_size)
    model.summary()

    x_val = train_data[:vocab_size]
    partial_x_train = train_data[vocab_size:]
    y_val = train_labels[:vocab_size]
    partial_y_train = train_labels[vocab_size:]

    history = model.fit(partial_x_train,
            partial_y_train,
            epochs=40,
            batch_size=512,
            validation_data=(x_val, y_val),
            verbose=1)
    results = model.evaluate(test_data, test_labels, verbose=2)
    print(results)

    history = history.history
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)

    gen_training_loss_graph(epochs, loss, val_loss)
    gen_training_acc_graph(epochs, acc, val_acc)


if __name__ == '__main__':
    main()
