import cv2
import random


def get_image_rotation(image):
    #通用写法，即使传入的是三通道图片依然不会出错
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation = random.randint(-90,90)

    #得到旋转矩阵，第一个参数为旋转中心，第二个参数为旋转角度，第三个参数为旋转之前原图像缩放比例
    M = cv2.getRotationMatrix2D(center, rotation, 1)
    #进行仿射变换，第一个参数图像，第二个参数是旋转矩阵，第三个参数是变换之后的图像大小
    image_rotation = cv2.warpAffine(image, M, (width, height))
    return image_rotation


if __name__=='__main__':
    img=cv2.imread('E:/lemon_datasets/test_images/test_0000.jpg')
    cv2.imshow(winname='original_image',mat=img)
    # cv2.waitKey()
    img_rotation=get_image_rotation(img)
    print(img_rotation.shape)
    cv2.imshow(winname='rotation_image',mat=img_rotation)
    cv2.waitKey()
