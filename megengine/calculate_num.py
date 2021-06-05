import numpy as np

num0=0
num1=0
num2=0
num3=0

with open('E:/lemon_datasets/test_images.csv') as file:
    while True:
        f=file.readline()
        if len(f)==0:
            break
        if int(f.split(',')[1])==0:
            num0+=1
        elif int(f.split(',')[1])==1:
            num1+=1
        elif int(f.split(',')[1])==2:
            num2+=1
        else:
            num3+=1

print(num0,num1,num2,num3)