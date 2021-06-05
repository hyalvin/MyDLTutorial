# 基于Tensorflow及Keras深度学习框架实现CIFAR-10图像的识别分类
# 环境：Anaconda4.5.11+Python3.7.0+Pycharm2020.1+tensorflow1.9.0(包含keras)+cuda9.0+cudnn7.0
# 选用神经网络：CNN

from tensorflow import keras
from tensorflow.keras.datasets import cifar10 # CIFAR-10数据集可直接通过Keras下载
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.python.keras.layers import Activation
import matplotlib.pyplot as plt

batch_size = 32 # 每批训练的样本数
num_classes = 10 # 类别个数，在CIFAR-10中是10
epochs = 100 # 训练轮数
data_augmentation = True # 训练是否使用数据增强，若使用则可增大网络的泛化能力

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

# 构造序列网络模型，即按声明顺序搭建
model = Sequential()
# 模型一共15个卷积层，每个卷积层后接RELU激活函数；3个2*2的最大池化层，每个池化层后接Dropout，Rate设定为0.5，可以有效防止过拟合；
# 尾部还有两个全连接层，节点数分别为500和10(10个类别)，第一个全连接层后接RELU激活函数和Dropout，第二个用softmax激活(多分类问题用softmax)

model.add(Conv2D(32, (3, 3), padding='same', input_shape=data_train.shape[1:]))
# Conv2D(32, (3, 3)：该层有32个3*3的卷积核，'same'：周围空白填充，特征图大小不变，data_train.shape[1:]：网络自适应的识别输入张量大小
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3), padding='same', input_shape=data_train.shape[1:]))
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3), padding='same', input_shape=data_train.shape[1:]))
model.add(Activation('relu'))

model.add(Conv2D(48, (3, 3), padding='same', input_shape=data_train.shape[1:]))
model.add(Activation('relu'))

model.add(Conv2D(48, (3, 3), padding='same', input_shape=data_train.shape[1:]))
model.add(Activation('relu'))

# 最大池化和Dropout
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(80, (3, 3), padding='same', input_shape=data_train.shape[1:]))
model.add(Activation('relu'))

model.add(Conv2D(80, (3, 3), padding='same', input_shape=data_train.shape[1:]))
model.add(Activation('relu'))

model.add(Conv2D(80, (3, 3), padding='same', input_shape=data_train.shape[1:]))
model.add(Activation('relu'))

model.add(Conv2D(80, (3, 3), padding='same', input_shape=data_train.shape[1:]))
model.add(Activation('relu'))

model.add(Conv2D(80, (3, 3), padding='same', input_shape=data_train.shape[1:]))
model.add(Activation('relu'))

# 最大池化和Dropout
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3), padding='same', input_shape=data_train.shape[1:]))
model.add(Activation('relu'))

model.add(Conv2D(128, (3, 3), padding='same', input_shape=data_train.shape[1:]))
model.add(Activation('relu'))

model.add(Conv2D(128, (3, 3), padding='same', input_shape=data_train.shape[1:]))
model.add(Activation('relu'))

model.add(Conv2D(128, (3, 3), padding='same', input_shape=data_train.shape[1:]))
model.add(Activation('relu'))

model.add(Conv2D(128, (3, 3), padding='same', input_shape=data_train.shape[1:]))
model.add(Activation('relu'))

model.add(GlobalMaxPooling2D())
model.add(Dropout(0.5))

model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax')) #softmax激活

model.summary () #打印网络结构及参数

#回调函数，用于对模型的训练过程进行改进
plateau=ReduceLROnPlateau(monitor="val_acc", #5epoch准确率没有上升，降低学习率，并设置最小学习率
                                verbose=1,
                                mode='max',
                                factor=0.5,
                                min_lr=0.5e-6,
                                patience=5)
early_stopping=EarlyStopping(monitor='val_acc', #20epoch准确率没有上升，停止训练
                                   verbose=1,
                                   mode='max',
                                   patience=20)

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
plt.title('CIFAR-10: CNN with data augmentation')
plt.xlabel('epochs')
plt.ylabel('Traing Accuracy / Loss')
plt.plot(epochs, acc, 'red', label='Training Accuracy')
plt.plot(epochs, loss, 'blue', label='Traing Loss')
plt.legend()
plt.savefig('E:\Results\CNN-2.png')
plt.show()