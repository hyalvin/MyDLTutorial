import os
from megengine.data.dataset import Dataset
import cv2
import numpy as np
import random
import time as t

# step1: 定义MyDataset类， 继承Dataset, 重写抽象方法：__len()__, __getitem()__
class MyDataset(Dataset):

    def __init__(self,
                 root_dir,
                 names_file,
                 random_rotation:bool=False,
                 random_crop:bool=False):
        super().__init__()
        self.root_dir = root_dir
        self.names_file = names_file
        self.data=[]
        self.label=[]
        self.random_rotation=random_rotation
        self.random_crop=random_crop

        if not os.path.isfile(self.names_file):
            print(self.names_file + 'does not exist!')

        #RandomCrop
        def cv2_crop(im, box):
            return im.copy()[box[1]:box[3], box[0]:box[2], :]

        def image_padding(image, padding_size: int = 5):
            return cv2.copyMakeBorder(image,
                                      top=padding_size,
                                      bottom=padding_size,
                                      left=padding_size,
                                      right=padding_size,
                                      borderType=cv2.BORDER_REPLICATE,
                                      )

        def image_crop(image, output_size):
            a = random.randint(0, 10)
            image_pad = image_padding(image, padding_size=a)
            b = random.randint(0, a)
            box = (b, b, b + output_size, b + output_size)
            return cv2_crop(image_pad, box)

        #RandomRotation
        def get_image_rotation(image):

            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation = random.randint(-90, 90)

            # 得到旋转矩阵，第一个参数为旋转中心，第二个参数为旋转角度，第三个参数为旋转之前原图像缩放比例
            M = cv2.getRotationMatrix2D(center, rotation, 1)
            # 进行仿射变换，第一个参数图像，第二个参数是旋转矩阵，第三个参数是变换之后的图像大小
            image_rotation = cv2.warpAffine(image, M, (width, height))
            return image_rotation


        with open(self.names_file) as file:
            R_mean,R_std=0,0
            G_mean,G_std=0,0
            B_mean,B_std=0,0
            while True:
                f=file.readline()
                if len(f)==0:
                    break
                image = cv2.imread(os.path.join(self.root_dir,f.split(',')[0]))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if image is not None:
                    self.data.append(image)
                    self.label.append(int(f.split(',')[1]))
                    if self.random_rotation:
                        image_rotate=get_image_rotation(image)
                        self.data.append(image_rotate)
                        self.label.append(int(f.split(',')[1]))
                    if self.random_crop and (int(f.split(',')[1])==0 or int(f.split(',')[1])==3):
                        image_croped=image_crop(image,output_size=640)
                        self.data.append(image_croped)
                        self.label.append(int(f.split(',')[1]))
            num=len(self.label)*640*640
            for image in self.data:
                R_mean+=np.sum(image[:,:,0])
                G_mean+=np.sum(image[:,:,1])
                B_mean+=np.sum(image[:,:,2])
            R_mean/=num
            G_mean/=num
            B_mean/=num
            for image in self.data:
                R_std+=np.sum((image[:,:,0]-R_mean)**2)
                G_std+=np.sum((image[:,:,1]-G_mean)**2)
                B_std+=np.sum((image[:,:,2]-B_mean)**2)
            R_std=np.sqrt(R_std/num)
            G_std=np.sqrt(G_std/num)
            B_std=np.sqrt(B_std/num)
            self.mean=[R_mean,G_mean,B_mean]
            self.std=[R_std,G_std,B_std]
            print(self.mean,self.std)
            print(len(self.label))
            num0,num1,num2,num3=0,0,0,0
            for item in self.label:
                if item==0:
                    num0+=1
                elif item==1:
                    num1+=1
                elif item==2:
                    num2+=1
                else:
                    num3+=1
            print(num0,num1,num2,num3)



    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx],self.label[idx]

    def get_mean_std(self):
        return self.mean,self.std






