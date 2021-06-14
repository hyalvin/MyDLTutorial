import cv2
import random
from PIL import Image


def cv2_crop(im, box):
    return im.copy()[box[1]:box[3], box[0]:box[2], :]


def image_padding(image,padding_size:int=5):
    return cv2.copyMakeBorder(image,
                              top=padding_size,
                              bottom=padding_size,
                              left=padding_size,
                              right=padding_size,
                              borderType=cv2.BORDER_REPLICATE,
                              )


def image_crop(image,output_size):
    a = random.randint(5,15)
    image_pad = image_padding(image, padding_size=a)
    b = random.randint(0,a)
    box=(b,b,b+output_size,b+output_size)
    return cv2_crop(image_pad,box)


if __name__=='__main__':
    img=cv2.imread('E:/lemon_datasets/test_images/test_0002.jpg')
    cv2.imshow(winname='original_image',mat=img)
    # cv2.waitKey()
    # img_crop=image_crop(image=img,output_size=640)
    # print(img_crop.shape)
    # cv2.imshow(winname='croped_image',mat=img_crop)
    box=(100,100,548,548)
    img_resize=cv2.resize(img,(224,224))
    cv2.imshow(winname='croped_image',mat=img_resize)
    print(img_resize.shape)
    cv2.waitKey()

