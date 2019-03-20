
import sys
import cv2
import numpy as np


def cross_correlation_2d(img, kernel):
    #find out if its 2d (B&W) or 3d (RGB) image
    dimension = img.ndim
    #get shape of kernel to find size of m&n
    kernel_shape = kernel.shape
    m= kernel_shape[0]
    n= kernel_shape[1]
    # #find out how many rows of padding zeros we need
    width= (m-1)/2
    height = (n-1)/2
    # pad array with zeros
    cc_image = np.zeros(img.shape)

    if dimension == 3:
        for channel in range(img.shape[2]):
            current_channel = img[:,:,channel]
            padded_channel = np.pad(current_channel,((width,width),(height,height)),'constant',constant_values=0)
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    row = padded_channel[i:i+m,j:j+n]
                    total = np.sum(row * kernel) 
                    cc_image[i,j,channel] = total
        return cc_image
    else:
        zero_padded_array = np.pad(img,((width,width),(height,height)),'constant',constant_values=0)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                row = zero_padded_array[i:i+m,j:j+n]
                total = np.sum(row * kernel) 
                cc_image[i,j] = total
        return cc_image



def convolve_2d(img, kernel):
    temp = np.flipud(kernel)
    temp = np.fliplr(temp)
    return cross_correlation_2d(img,temp)



def gaussian_blur_kernel_2d(sigma, height, width):
    #accounts for non-square matrtices
    center_height=(int)(height/2)
    center_width=(int)(width/2)
    #make empty kernel with correct size
    kernel=np.zeros((height,width))
    for x in range(height):
       for y in range(width):
          diff=np.sqrt((x-center_height)**2+(y-center_width)**2)
          kernel[x,y]=np.exp(-(diff**2)/float(2*sigma**2))
    return kernel/np.sum(kernel)

def low_pass(img, sigma, size):
    kernel = gaussian_blur_kernel_2d(sigma,size, size)
    return convolve_2d(img, kernel)


def high_pass(img, sigma, size):
    #this calculates low pass
    kernel = gaussian_blur_kernel_2d(sigma,size, size)
    #original - low pass = high pass
    return img - convolve_2d(img,kernel)


def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)


