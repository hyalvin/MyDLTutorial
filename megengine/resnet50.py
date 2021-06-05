# 框架：MegEngine1.3.1
# 网络：VGG16
# 数据集：CIFAR-10

# 导入库

import megengine.module as M


# 设置超参数
batch_size = 32
num_classes = 4
epochs = 100
drop_rate = 0.5


class ConvBN(M.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride,
                 padding
                 ):
        super().__init__()
        self.conv=M.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,
                           stride=stride,padding=padding,bias=False)
        self.bn=M.BatchNorm2d(out_channels)

    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        return x

class ResidualBlock(M.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size,
                 stride,
                 padding:int=0):
        super().__init__()
        #self.stride=stride
        self.conv1=ConvBN(in_channels=in_channels,out_channels=out_channels,kernel_size=(1,1),
                           stride=stride,padding=padding)
        self.relu1 = M.ReLU()
        self.conv2=ConvBN(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size,
                           stride=(1,1),padding=1)
        self.relu2 = M.ReLU()
        self.conv3=ConvBN(in_channels=out_channels,out_channels=4*out_channels,kernel_size=(1,1),
                           stride=(1,1),padding=padding)
        if stride==(1,1):
            self.identity=M.Identity()
        else:
            self.identity=ConvBN(in_channels=in_channels,out_channels=4*out_channels,kernel_size=(1,1),
                           stride=stride,padding=padding)
        self.relu3 = M.ReLU()

    def forward(self,x):
        identity=self.identity(x)
        x=self.conv1(x)
        x=self.relu1(x)
        x=self.conv2(x)
        x=self.relu2(x)
        x=self.conv3(x)
        x=x+identity
        x=self.relu3(x)
        return x


class ResNet50(M.Module):
    def __init__(self):
        super().__init__()
        self.conv1=ConvBN(in_channels=3,out_channels=64,kernel_size=(7,7),
                           stride=(2,2),padding=(3,3))
        self.relu=M.ReLU()
        self.maxpool=M.MaxPool2d(kernel_size=(3,3),stride=(2,2),padding=(1,1))
        def make_block(in_channels,out_channels,block_nums):
            block = []
            for i in range(block_nums):
                block.append(ResidualBlock(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3),
                                           stride=(2,2) if i==0 else (1,1),padding=0))
                in_channels=4*out_channels
            return M.Sequential(*block)
        self.block1=make_block(in_channels=64,out_channels=64,block_nums=3)
        self.dropout1 = M.Dropout(drop_prob=drop_rate)
        self.block2=make_block(in_channels=256, out_channels=128, block_nums=4)
        self.dropout2 = M.Dropout(drop_prob=drop_rate)
        self.block3=make_block(in_channels=512, out_channels=256, block_nums=6)
        self.dropout3 = M.Dropout(drop_prob=drop_rate)
        self.block4=make_block(in_channels=1024, out_channels=512, block_nums=3)
        self.dropout4 = M.Dropout(drop_prob=drop_rate)
        self.avgpool = M.AvgPool2d(kernel_size=(10,10))

        def make_fc(in_features, out_features):
            fc = []
            fc.append(M.Linear(in_features=in_features, out_features=out_features))
            fc.append(M.BatchNorm1d(out_features))
            fc.append(M.Dropout(drop_prob=drop_rate))
            fc.append(M.ReLU())
            return M.Sequential(*fc)

        self.fc1 = make_fc(in_features=2048, out_features=512)
        self.fc2 = make_fc(in_features=512, out_features=128)
        self.fc3 = M.Linear(in_features=128, out_features=num_classes)
        self.bn = M.BatchNorm1d(num_classes)
        self.softmax = M.Softmax()

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.block1(x)
        x = self.dropout1(x)
        x = self.block2(x)
        x = self.dropout2(x)
        x = self.block3(x)
        x = self.dropout3(x)
        x = self.block4(x)
        x = self.dropout4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.bn(x)
        x = self.softmax(x)
        return x

# 实例化网络并打印结构
resnet50 = ResNet50()
print(resnet50)