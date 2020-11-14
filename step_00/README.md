STEP.00: Setup Development Environment
===

## Documents
- [tensorflow docker](https://www.tensorflow.org/install/docker?hl=ja)

### タグ
| TAG | Description |
| --- | ----------- |
| latest | TensorFlow CPU バイナリイメージの最新リリース |
| nightly | TensorFlow のナイトリービルドのイメージ（不安定） |
| version | TensorFlow のバイナリイメージのバージョンを指定（例: 2.1.0） |
| devel | TensorFlow master 開発環境のナイトリービルド |
| custom-op | TF カスタムオペレーションを開発するための特別な試験運用版イメージ |

### タグのバリエーション
| Variation | Description |
| --------- | ----------- |
| <TAG>-gpu | GPUサポート|
| <TAG>-jupyter | jupyter |


## Notes
- GPUサポートはホストマシンの設定が必要
- 今後の学習で実行しやすいように docker-compose を準備

## Run
```
docker-compose up --build
```
