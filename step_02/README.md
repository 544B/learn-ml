STEP.02: Basic image classification
===

## Documents
- [Basic classification: Classify images of clothing](https://www.tensorflow.org/tutorials/keras/classification?hl=ja)


## Notes
### Model
#### Layers
##### Flatten
画像を（28×28ピクセルの）2次元配列から、28×28＝784ピクセルの、1次元配列に変換する。
画像の中に積まれているピクセルの行を取り崩し、横に並べると考える。
学習すべきパラメータはなく、ただデータのフォーマット変換を行うだけ。
##### Dense
密結合あるいは全結合されたニューロン。
最初の Dense 層には、128個のノード（あるはニューロン）がある。
最後の層でもある2番めの層は、10ノードのsoftmax層。
この層は、合計が1になる10個の確率の配列を返す。
それぞれのノードは、今見ている画像が10個のクラスのひとつひとつに属する確率を出力する。

### CompileSettings
#### LossFunction
訓練中にモデルがどれくらい正確かを測定。
この関数の値を最小化することにより、訓練中のモデルを正しい方向に向かわせる。
#### Optimizer
モデルが見ているデータと、損失関数の値から、どのようにモデルを更新するかを決定する。
#### Metrics
訓練とテストのステップを監視するのに使用。
例では画像が正しく分類された比率（accuracy - 正解率）を使用する。


## Run
```
docker-compose up --build
```
