import imutils
import cv2
from skimage.segmentation import clear_border
import pytesseract
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
from itertools import accumulate
import os
import os.path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from math import *

"""
Read a dataset a of images
-------------------
Return a list of images

"""
def read_images(path_name,no_cars=426,ext='png'):
    images = []
    for i in range(no_cars):
        car_label = path_name.format(i,ext)
        if os.path.isfile(car_label):
            images.append(io.imread(car_label))
        elif os.path.isfile(path_name.format(i,"jpg")):
            images.append(path_name.format(i,"jpg"))
        elif os.path.isfile(path_name.format(i,"JPG")):
            images.append(path_name.format(i,"JPG"))
    return images

    
    
    
"""
Create histogram distribution to see the intensity values and investigate if we cannot stretch of equalize the intensities to make the image lighter
"""
def create_image_histogram(gray):
    hist,bins = np.histogram(gray.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    return cdf_normalized
    
def get_blackhat(gray,kernel_width=13,kernel_height=5,kernel=None): # Reveal Dark Regions (Letter,Digits,Symbols) on Light Backgrounds
    if kernel == None:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_width,kernel_height))
    blackhat = cv2.morphologyEx(gray.copy(),cv2.MORPH_BLACKHAT,kernel)
    return blackhat,kernel




def get_closing(gray,kernel_width=3,kernel_height=3,kernel=None): # Fill Small Holes and Identify Larger Structures: Reveal Light Characters
    if kernel == None:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_width,kernel_height))
    closing = cv2.morphologyEx(gray.copy(),cv2.MORPH_CLOSE,kernel)
    return closing,kernel
    

def sobel_gradient(blackhat):
    grad_x = cv2.Sobel(blackhat,ddepth=cv2.CV_32F,dx=1,dy=0,ksize=-1)
    grad_x = np.absolute(grad_x)
    (min_val,max_val) = (np.min(grad_x),np.max(grad_x))
    grad_x = 255 * ((grad_x - min_val) / (max_val - min_val)) #rescale 
    grad_x = grad_x.astype('uint8')
    return grad_x
    
def morphological_preprocessing(gray,square_kernel_width=3,square_kernel_height=3,blackhat_kernel_width=13,blackhat_kernel_height=5,gaussian_blur=5,verbose=False):
    light,light_kernel = get_closing(gray,square_kernel_width,square_kernel_height)
    blackhat,blackhat_kernel = get_blackhat(gray,kernel_width=blackhat_kernel_width,kernel_height=blackhat_kernel_height)
    threshold = cv2.threshold(light,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    grad_x = sobel_gradient(blackhat)
    smooth_grad_x = cv2.GaussianBlur(grad_x,(gaussian_blur,gaussian_blur),0)
    smooth_grad_x = cv2.morphologyEx(smooth_grad_x,cv2.MORPH_CLOSE,blackhat_kernel)
    smooth_thresh = cv2.threshold(smooth_grad_x, 0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    morphological_thresh = cv2.erode(smooth_thresh,None,iterations=2)
    morphological_thresh = cv2.dilate(morphological_thresh,None,iterations=2)
    light_threshold = cv2.bitwise_and(morphological_thresh,morphological_thresh,mask=light)
    light_threshold = cv2.dilate(light_threshold,None,iterations=2)
    light_threshold = cv2.erode(light_threshold,None,iterations=1)
    return light_threshold.copy()



#===================================================================== Light Manipulation ==================================

def hist_eq_(img): # Perform Histogram Equalization
    unique_elements, counts_elements = np.unique(img, return_counts=True)
    cdf_ = list(accumulate(counts_elements))
    hv_eq = lambda cdf_v,cdf_min,M,N,L:np.round(((cdf_v-cdf_min)/(M*N - cdf_min))*(L-1))
    M,N = img.shape
    intensity = 256
    hv = [hv_eq(cdfv,min(cdf_),M,N,intensity) for cdfv in cdf_]
    hv_eq = lambda cdf_v,cdf_min,M,N,L:np.round(((cdf_v-cdf_min)/(M*N - cdf_min))*(L-1)).astype(int)
    hv = [hv_eq(cdfv,min(cdf_),M,N,intensity) for cdfv in cdf_]
    i_ = np.zeros((M,N))
    for x,y in zip(unique_elements,hv):
        i_ = np.where(img==x,y,i_)
    return i_

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)
#===================================================================== Plotting Images===================================================================== 

def image_random_sample(images,k=16):
    image_data = images[:k]
    fig = plt.figure(figsize=(26, 20))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(int(sqrt(k)), int(sqrt(k))),  # creates 2x2 grid of axes
                 axes_pad=0.1,)  # pad between axes in inch.)

    for ax, im in zip(grid, image_data):
        ax.imshow(im)

    plt.title("Random Sample of Images")
    plt.show()

def plot_histogram(gray,cdf_normalized):
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(gray.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.title("Histogram Plot of the Grayscale Image")
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()
    
    
"""
Input: 
img1: First Image
img2: Second Image
title1: Title for First Image
title2: Title for Second Image
-----------------------------
"""

def plot_images(img1,img2,title1="",title2=""):
    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_subplot(121) # 121 - 1 Row 2 Columns and Target 1st Column of Row
    ax1.imshow(img1,cmap='gray')
    ax1.set(xticks=[],yticks=[],title=title1)
    ax2 = fig.add_subplot(122) # 121 - 1 Row 2 Columns and Target 1st Column of Row
    ax2.imshow(img2,cmap='gray')
    ax2.set(xticks=[],yticks=[],title=title2) 
    fig.show()
    