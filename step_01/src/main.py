from tensorflow import keras

# MNIST データセットをロード
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# サンプルを整数から浮動小数点数に変換
x_train, x_test = x_train / 255.0, x_test / 255.0
# 損失関数を作成
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def get_model():
    """
    レイヤーを積み重ねてkeras.Sequentialモデルを構築
    訓練用のオプティマイザと損失関数を設定
    """
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])
    model.compile(
        optimizer='adam',
        loss=loss_fn,
        metrics=['accuracy']
        )
    return model


def check_loss(model):
    """
    クラスが正しい確率の対数をとって符号を反転させたもの
    この値はモデルがこのクラスが正しいと確信しているときに 0 になる
    訓練されていないモデルはランダムに近い確率を出力する
    """
    predictions = model(x_train[:1]).numpy()
    return loss_fn(y_train[:1], predictions).numpy()


def main():
    model = get_model()

    print('# 訓練前の損失確認: {}'.format(check_loss(model)))
    print('# パラメータ調整')
    model.fit(x_train, y_train, epochs=5)
    print('# モデル性能確認')
    model.evaluate(x_test,  y_test, verbose=2)
    print('# 訓練後の損失確認: {}'.format(check_loss(model)))

if __name__ == '__main__':
    main()
