# 框架：MegEngine1.3.1
# 网络：VGG16
# 数据集：CIFAR-10

# 导入库
import megengine as mge
import megengine.module as M
import megengine.functional as F
from megengine import optimizer
from megengine.optimizer import MultiStepLR
from megengine.autodiff import GradManager
from megengine.jit import trace
from megengine.data import DataLoader
from megengine.data.sampler import RandomSampler, SequentialSampler
from megengine.data.transform import Normalize, Compose, ToMode,  RandomHorizontalFlip, RandomVerticalFlip
import numpy as np
import time as t
from MyDataset import MyDataset

# 设置超参数
batch_size = 32
num_classes = 4
epochs = 300
drop_rate = 0.5

train_dataset=MyDataset(root_dir=r'E:/lemon_datasets/train_images',
                        names_file=r'E:/lemon_datasets/train_images.csv',
                        random_rotation=True,
                        random_crop=True)
test_dataset=MyDataset(root_dir=r'E:/lemon_datasets/test_images',
                       names_file=r'E:/lemon_datasets/test_images.csv',
                       random_rotation=False,
                       random_crop=False)


# 搭建VGG16
class Conv_BN_Relu(M.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size,
                 stride,
                 padding):
        super().__init__()
        self.conv = M.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding)
        self.bn = M.BatchNorm2d(out_channels)
        self.relu = M.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class VGG16(M.Module):
    def __init__(self):
        super().__init__()
        self.conv_bn_relu1 = Conv_BN_Relu(in_channels=3, out_channels=16, kernel_size=(5, 5),
                                        stride=(1, 1), padding=(2, 2))
        self.maxpool1 = M.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.dropout1 = M.Dropout(drop_prob=drop_rate)
        self.conv_bn_relu2 = Conv_BN_Relu(in_channels=16, out_channels=64, kernel_size=(3, 3),
                                        stride=(1, 1), padding=(1, 1))
        self.maxpool2 = M.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.dropout2 = M.Dropout(drop_prob=drop_rate)
        def make_block(in_channels, out_channels, block_nums):
            block = []
            for i in range(block_nums):
                block.append(Conv_BN_Relu(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=(3, 3),
                                          stride=(1, 1),
                                          padding=(1, 1)))
                in_channels = out_channels
            block.append(M.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)))
            block.append(M.Dropout(drop_prob=drop_rate))
            return M.Sequential(*block)
        self.block1 = make_block(in_channels=64, out_channels=64, block_nums=2)
        self.block2 = make_block(in_channels=64, out_channels=128, block_nums=2)
        self.block3 = make_block(in_channels=128, out_channels=256, block_nums=3)
        self.block4 = make_block(in_channels=256, out_channels=512, block_nums=3)
        self.block5 = make_block(in_channels=512, out_channels=512, block_nums=3)

        def make_fc(in_features, out_features):
            fc = []
            fc.append(M.Linear(in_features=in_features, out_features=out_features))
            fc.append(M.BatchNorm1d(out_features))
            fc.append(M.Dropout(drop_prob=drop_rate))
            fc.append(M.ReLU())
            return M.Sequential(*fc)
        self.fc1 = make_fc(in_features=512*5*5, out_features=1024)
        self.fc2 = make_fc(in_features=1024, out_features=64)
        self.fc3 = M.Linear(in_features=64, out_features=num_classes)
        self.bn = M.BatchNorm1d(num_classes)
        self.softmax = M.Softmax()

    def forward(self,x):
        x = self.conv_bn_relu1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = self.conv_bn_relu2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.bn(x)
        x = self.softmax(x)
        return x


# 实例化网络并打印结构
VGG16 = VGG16()
# print(VGG16)
# state_dict=mge.load('VGG16.mge')
# VGG16.load_state_dict(state_dict)

# 创建Dataloader用于训练
print('\n'+"----------start training----------"+'\n')
sampler_train=RandomSampler(dataset=train_dataset, batch_size=batch_size, drop_last=False)
train_mean,train_std = train_dataset.get_mean_std()
transform_train = Compose([
    RandomHorizontalFlip(),
    Normalize(mean=train_mean, std=train_std),
    ToMode("CHW")
])
train_dataloader = DataLoader(dataset=train_dataset, sampler=sampler_train, transform=transform_train)


@trace
def train_func(data, label, *, net, gm):
    net.train()
    with gm:
        pred = net(data)
        loss = F.loss.cross_entropy(pred=pred, label=label)
        gm.backward(loss)
    return pred,loss


optimizer = optimizer.Adam(params=VGG16.parameters(), lr=0.1, weight_decay=1e-4)
schedule = MultiStepLR(optimizer=optimizer, milestones=[50, 100, 200], gamma=0.5)
gm = GradManager().attach(VGG16.parameters())


# 如果想用动态计算图模式，please set trace.enabled = False
# trace.enabled = False

for epoch in range(epochs):
    start=t.time()
    total_loss=0
    correct=0
    total=0
    for batch_data,batch_label in train_dataloader:
        batch_label = np.array(batch_label).astype(np.int32)
        optimizer.clear_grad()
        pred,loss = train_func(mge.tensor(batch_data),mge.tensor(batch_label), net=VGG16, gm=gm)
        optimizer.step()
        total_loss += loss.numpy().item()
        correct += (pred.numpy().argmax(axis=1) == batch_label).sum().item()
        total += batch_label.shape[0]
    schedule.step()
    print("epoch{}: lr={:.8f}, loss={:.6f}, training accuracy={:.2f}%, time={:.2f}s".format(epoch,schedule.get_lr()[0],total_loss/len(train_dataloader),correct*100.0/total,t.time()-start))
print('\n'+"----------end training----------"+'\n')


# 模型保存
mge.save(VGG16.state_dict(),'VGG16_v2.mge')


# 模型测试
print('\n'+"----------start testing----------"+'\n')
sampler_test=SequentialSampler(dataset=test_dataset,batch_size=batch_size)
test_mean,test_std=test_dataset.get_mean_std()
transform_test=Compose([
    Normalize(mean=test_mean, std=test_std),
    ToMode("CHW")
])
test_dataloader = DataLoader(dataset=test_dataset,sampler=sampler_test,transform=transform_test)


@trace
def eval_func(data, label, *, net):
    net.eval()
    pred = net(data)
    loss = F.loss.cross_entropy(pred=pred, label=label)
    return pred,loss


correct = 0
total = 0
for data, label in test_dataloader:
    label = np.array(label).astype(np.int32)
    pred, _ = eval_func(mge.tensor(data), mge.tensor(label), net=VGG16)
    correct += (pred.numpy().argmax(axis=1) == label).sum().item()
    total += label.shape[0]

print("correct={}, total={}, testing accuracy={:.2f}%".format(correct,total,correct*100.0/total))
print('\n'+"----------end testing----------")