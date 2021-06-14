# 基于Tensorflow及Keras深度学习框架实现CIFAR-10图像的识别分类
# 环境：Anaconda4.5.11+Python3.7.0+Pycharm2020.1+tensorflow1.9.0(包含keras)+cuda9.0+cudnn7.0
# 选用神经网络：VGG-16,带有BN层

from tensorflow import keras
from tensorflow.keras.datasets import cifar10 #CIFAR-10数据集可直接通过Keras下载
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import Input, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.layers import Dense, Softmax
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

batch_size = 32 # 每批训练的样本数
num_classes = 10 # 类别个数，在CIFAR-10中是10
epochs = 200 # 训练轮数
data_augmentation = False # 训练是否使用数据增强，若使用则可增大网络的泛化能力
drop_rate=0.5

# 将下载的数据集分为训练集和测试集，并取出数据和标签
(data_train, label_train), (data_test, label_test) = cifar10.load_data()
print('data_train shape:', data_train.shape) #打印训练集大小，CIFAR-10为5W张32*32*3的图像
print('data_test shape:', data_test.shape) #打印测试集大小，CIFAR-10为1W张32*32*3的图像
print('train samples:',data_train.shape[0]) #训练样本数
print('test samples:',data_test.shape[0]) #测试样本数
data_train = data_train.astype('float32') #astype转换数据类型
data_test = data_test.astype('float32')
data_train /= 255 #数据归一化
data_test /= 255

# 转换整形标签为onehot编码
label_train = keras.utils.to_categorical(label_train, num_classes)
label_test = keras.utils.to_categorical(label_test, num_classes)

#经典的卷积层结构CBR=CONV+BN+RELU
def Conv_BN_Relu(filters, kernel_size, strides, input_layer):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',kernel_initializer="he_normal")(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


input_layer = Input((32, 32, 3))

x = Conv_BN_Relu(filters=64, kernel_size=(3, 3), strides=1, input_layer=input_layer)
x = Conv_BN_Relu(filters=64, kernel_size=(3, 3), strides=1, input_layer=x)
x = MaxPooling2D(pool_size=(2,2),strides=2,padding='same')(x)

x = Conv_BN_Relu(filters=128, kernel_size=(3, 3), strides=1, input_layer=x)
x = Conv_BN_Relu(filters=128, kernel_size=(3, 3), strides=1, input_layer=x)
x = MaxPooling2D(pool_size=(2,2),strides=2,padding='same')(x)

x = Conv_BN_Relu(filters=256, kernel_size=(3, 3), strides=1, input_layer=x)
x = Conv_BN_Relu(filters=256, kernel_size=(3, 3), strides=1, input_layer=x)
x = Conv_BN_Relu(filters=256, kernel_size=(3, 3), strides=1, input_layer=x)
x = MaxPooling2D(pool_size=(2,2),strides=2,padding='same')(x)

x = Conv_BN_Relu(filters=512, kernel_size=(3, 3), strides=1, input_layer=x)
x = Conv_BN_Relu(filters=512, kernel_size=(3, 3), strides=1, input_layer=x)
x = Conv_BN_Relu(filters=512, kernel_size=(3, 3), strides=1, input_layer=x)
x = MaxPooling2D(pool_size=(2,2),strides=2,padding='same')(x)

x = Conv_BN_Relu(filters=512, kernel_size=(3, 3), strides=1, input_layer=x)
x = Conv_BN_Relu(filters=512, kernel_size=(3, 3), strides=1, input_layer=x)
x = Conv_BN_Relu(filters=512, kernel_size=(3, 3), strides=1, input_layer=x)
x = MaxPooling2D(pool_size=(2,2),strides=2,padding='same')(x)

x = Flatten()(x)
x = Dense(2048)(x)
x = BatchNormalization()(x)
x = Dropout(drop_rate)(x)
x = Activation("relu")(x)

# FC layer 2
x = Dense(1024)(x)
x = BatchNormalization()(x)
x = Dropout(drop_rate)(x)
x = Activation("relu")(x)

# FC layer 3
x = Dense(num_classes)(x)
x = BatchNormalization()(x)
y = Activation("softmax")(x)

model = Model([input_layer], [y])
model.summary()

#回调函数，用于对模型的训练过程进行改进
plateau=ReduceLROnPlateau(monitor="val_acc", #10epoch准确率没有上升，降低学习率，并设置最小学习率
                                verbose=1,
                                mode='max',
                                factor=0.5,
                                min_lr=0.5e-6,
                                patience=10)
early_stopping=EarlyStopping(monitor='val_acc', #30epoch准确率没有上升，停止训练
                                   verbose=1,
                                   mode='max',
                                   patience=30)

# 初始化Adam优化器,学习率设为0.0001
opt = keras.optimizers.Adam(lr=0.0001)

# 模型编译，用交叉熵损失函数，Adam优化器，优化目标是准确率
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

print("train____________")
# 通过data_augmentation的值来选择是否使用数据增强
if not data_augmentation:
        print('Not using data augmentation.') #不用数据增强
        history=model.fit(data_train, #训练样本数据
                  label_train, #训练样本标签
                  batch_size=batch_size, #每批训练的样本数
                  epochs=epochs, #训练轮数
                  verbose=2, #用于打印输出日志
                  validation_data=(data_test, label_test), #指定用于验证的数据集
                  callbacks=[plateau,early_stopping]) #回调函数
else:
        print('Using real-time data augmentation.') #使用数据增强
        # keras.preprocessing.image中的图片生成器
        datagen = ImageDataGenerator(
            # 是否对输入的图片每个通道减去每个通道对应均值
            featurewise_center=False,
            # 是否每张图片减去样本均值, 使得每个样本均值为0
            samplewise_center=False,
            # 是否将输入除以数据集的标准差以完成标准化
            featurewise_std_normalization=False,
            # 是否将输入的每个样本除以其自身的标准差
            samplewise_std_normalization=False,
            # 是否对输入数据施加ZCA白化
            zca_whitening=False,
            # ZCA白化的参数
            zca_epsilon=1e-06,
            # 随机旋转图像的角度
            rotation_range=0,
            # 随机水平偏移的幅度
            width_shift_range=0.1,
            # 随机竖直偏移的幅度
            height_shift_range=0.1,
            # 随机水平翻转图片
            horizontal_flip=True,
            # 随机竖直翻转图片
            vertical_flip=False,
        )
        # 数据增强
        datagen.fit(data_train)
        # 利用增强后的数据对模型进行训练.
        history=model.fit_generator(datagen.flow(data_train, label_train,batch_size=batch_size),
                                    steps_per_epoch=data_train.shape[0] // batch_size,  #一个epoch结束的时机
                                    epochs=epochs,
                                    verbose=2,
                                    validation_data=(data_test, label_test),
                                    callbacks=[plateau,early_stopping])

print("test_____________")
loss,acc=model.evaluate(data_test,label_test) #利用测试集对模型进行验证并输出损失和正确率
print("loss=",loss)
print("accuracy=",acc)


#利用matplotlib绘制训练归一化损失和预测准确度
acc = history.history['acc']
loss = history.history['loss']
epochs = range(1, len(acc) + 1)
plt.title('CIFAR-10: VGG16 without data augmentation')
plt.xlabel('epochs')
plt.ylabel('Traing Accuracy / Loss')
plt.plot(epochs, acc, 'red', label='Training Accuracy')
plt.plot(epochs, loss, 'blue', label='Traing Loss')
plt.legend()
plt.savefig('E:\Results\VGG16-1.png')
plt.show()