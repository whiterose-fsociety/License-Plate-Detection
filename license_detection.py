import numpy as np 
import operator
from skimage import io
import matplotlib.pyplot as plt
from scipy import signal,ndimage
import cv2
from scipy.ndimage import rotate

def neighbourhood(f,Db,square_pad=1,point=(0,0),type_='zero'): # Given a point on the image, return the neighbour
    if type_ == 'zero':
        fpad = np.pad(f,square_pad)
    elif type_ == "replicate":
        fpad = np.pad(f,square_pad,'symmetric')
    DbsizeA,DbsizeB = Db.shape
    shift = (square_pad,square_pad)
    ps = tuple(map(operator.add,point,shift))
    return fpad[ps[0]-1:DbsizeA+point[0],ps[1]-1:DbsizeB+point[1]] 

def neighboursum(f,Db):
    sum_ = 0
    for i in range(len(Db)):
        for j in range(len(Db[i])):
            sum_ += f[i,j]*Db[i,j]
    return sum_


def convolve(f,Db,square_pad=1,type_='zero'):
    f_convolve = f.copy()
    for i in range(len(f_convolve)):
        for j in range(len(f_convolve[i])):
            p = (i,j)
            neighbours = neighbourhood(f,Db,square_pad=square_pad,point=p,type_=type_)
            f_convolve[i,j] = neighboursum(neighbours,Db)
    return f_convolve

def map_(f, K): #Scale Image
    smallest = np.amin(f)
    fm = f - smallest
    fs = K * (fm / np.amax(fm))
    return fs


def threshold_gradient(gmag,gdir,A=90,TA=45,type_='easy'):
    Gdimx,Gdimy = gmag.shape
    gxy = np.zeros((Gdimx,Gdimy))
    T = 0.3*max(gmag.flatten())
    if type_ == 'easy':
        gxy = (Gmag > T).astype(int)
    else:
        for i in range(len(gxy)):
            for j in range(len(gxy[i])):
                if (gmag[i,j] > T) and (gdir[i,j] > A-TA and gdir[i,j] < A+TA):
                    gxy[i,j] = 1
    return gxy


def fetch_ones(array):
    ones = []
    for i in range(len(array)):
        if array[i] == 1:
            ones.append(i)
    return ones


def fill_array(array,start,end):
    fill_ = array.copy()
    for i in range(start,end):
        fill_[i] = 1
    return fill_

def fill_gap(row,L=25): # Row = specific row with zeros and ones
    coords = fetch_ones(row)
    fill = row.copy()
    for i in range(len(coords)-1):
        diff = coords[i+1] - coords[i]
        if (diff <= L) and (diff > 1):
            fill = fill_array(fill,coords[i],coords[i+1])
    return fill


def fill_gradient(gmag,L=25):
    grad = gmag.copy()
    for i in range(len(grad)):
        grad[i] = fill_gap(gmag[i,:],L=L)
    return grad



sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
image = io.imread('car.tif')
"""  Digital Image Processing Using Matlab Page 566 of  845
# sob = np.hypot(scipy_filter_x,scipy_filter_y)
# wneg45 = np.array([[-2,-1,0],[-1,0,1],[0,1,2]])
# float_ = map_(image,1)"""


Gx = filter_x = -signal.convolve2d(image,sobel_x,boundary='symm',mode='same')
Gy = filter_y = -signal.convolve2d(image,sobel_y,boundary='symm',mode='same')
filter_x2 = signal.convolve2d(image,sobel_x,boundary='symm',mode='same')
filter_y2 = signal.convolve2d(image,sobel_y,boundary='symm',mode='same')

Gdir = np.arctan2(-filter_y,filter_x) * 180/np.pi
Gmag = np.hypot(filter_x,filter_y)
Gmag2 = np.hypot(filter_x2,filter_y2)
T = 0.3*max(Gmag.flatten())
A = 90
TA = 45
gxy = threshold_gradient(Gmag,Gdir,A=A,TA=TA,type_='hard')
gradient_fill = fill_gradient(gxy)        
gxy_rot45 = rotate(gxy,angle=45)
gxy_rot90 = rotate(gxy,angle=90)
gxy_rot90_ = rotate(gxy,angle=-90)
gxy_rot135 = rotate(gxy,angle=135)
gxy_rot180 = rotate(gxy,angle=180)

fgxy_45 = fill_gradient(gxy_rot45)
fgxy_90 = fill_gradient(gxy_rot90)
fgxy_90_ = fill_gradient(gxy_rot90_)
fgxy_135 = fill_gradient(gxy_rot135)
fgxy_180 = fill_gradient(gxy_rot180)

x = rotate(fgxy_90,angle=-90)
g= gradient_fill
logical_or = np.bitwise_or(g,x)
fig1, f1_axes = plt.subplots(ncols=7, nrows=1, constrained_layout=True)
f1_axes[0].imshow(rotate(fgxy_45,angle=-45),cmap='gray')
f1_axes[1].imshow(rotate(fgxy_135,angle=-135),cmap='gray')
f1_axes[2].imshow(rotate(fgxy_90,angle=-90),cmap='gray')
f1_axes[3].imshow(rotate(fgxy_90_,angle=90),cmap='gray')
f1_axes[4].imshow(rotate(fgxy_180,angle=-180),cmap='gray')
f1_axes[5].imshow(gradient_fill,cmap='gray')
f1_axes[6].imshow(logical_or,cmap='gray')
plt.show()
