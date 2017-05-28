import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
import numpy as np
from smallcnn import save_history

img_width, img_height = 150, 150
train_data_dir = 'dataset/jpg/Naomi'
nb_train_samples = 136

result_dir = 'dataset/results/Naomi'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)


def save_bottleneck_features():
    """VGG16にDog vs Catの訓練画像、バリデーション画像を入力し、
    ボトルネック特徴量（FC層の直前の出力）をファイルに保存する"""

    # VGG16モデルと学習済み重みをロード
    # Fully-connected層（FC）はいらないのでinclude_top=False）
    model = VGG16(include_top=False, weights='imagenet')
    model.summary()

    # ジェネレータの設定
    datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Dog vs Catのトレーニングセットを生成するジェネレータを作成
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode=None,
        shuffle=False)

    # ジェネレータから生成される画像を入力し、VGG16の出力をファイルに保存
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
    np.save(os.path.join(result_dir, 'bottleneck_features_train.npy'),
            bottleneck_features_train)


if __name__ == '__main__':
    # 訓練データとバリデーションデータのボトルネック特徴量の抽出
    save_bottleneck_features()
