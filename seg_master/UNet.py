"""
@author: LiShiHang
@software: PyCharm
@file: UNet.py
@time: 2018/12/27 16:54
@desc:
"""

from tensorflow.keras import Model, layers
from tensorflow.keras.applications import vgg16
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Reshape, MaxPool2D, concatenate, UpSampling2D


def UNet(nClasses, input_height, input_width):
    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, 3))

    vgg_streamlined = vgg16.VGG16(
        include_top=False,
        weights='imagenet', input_tensor=img_input)
    assert isinstance(vgg_streamlined, Model)

    o = UpSampling2D((2, 2))(vgg_streamlined.output)
    o = concatenate([vgg_streamlined.get_layer(
        name="block4_pool").output, o], axis=-1)
    o = Conv2D(512, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    o = UpSampling2D((2, 2))(o)
    o = concatenate([vgg_streamlined.get_layer(
        name="block3_pool").output, o], axis=-1)
    o = Conv2D(256, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    o = UpSampling2D((2, 2))(o)
    o = concatenate([vgg_streamlined.get_layer(
        name="block2_pool").output, o], axis=-1)
    o = Conv2D(128, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    o = UpSampling2D((2, 2))(o)
    o = concatenate([vgg_streamlined.get_layer(
        name="block1_pool").output, o], axis=-1)
    o = Conv2D(64, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    # UNet 네트워크 처리 입력이 2 배 미러링되므로 최종 입력 및 출력이 2 배 감소합니다.
    # 여기에서 직접 업 샘플링하고 원래 크기를 설정합니다.
    o = UpSampling2D((2, 2))(o)
    o = Conv2D(64, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    o = Conv2D(1, (1, 1), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation("sigmoid")(o)



    # o = Conv2D(nClasses, (1, 1), padding="same")(o)
    # o = BatchNormalization()(o)
    # o = Activation("relu")(o)

    # o = Reshape((-1, nClasses))(o)
    # o = Activation("softmax")(o)

    model = Model(inputs=img_input, outputs=o)
    return model


if __name__ == '__main__':
    m = UNet(15, 320, 320)
    # print(m.get_weights()[2]) # 看看权重改变没，加载vgg权重测试用
    from keras.utils import plot_model
    plot_model(m, show_shapes=True, to_file='model_unet.png')
    print(len(m.layers))
    m.summary()
