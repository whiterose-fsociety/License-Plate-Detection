#============================ Personal Helper Modules
from . import environment as en
#============================ Libraries
import skimage.io as io
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import imutils
import pytesseract
import numpy as np
import re

"""
Task: Check the ratio of the minimum bounding rectangle that is drawn by the contour
Input: 
Area
Width
Height
Verbose: Print all the debug information
-----------------------------
Return: True if the aspect ratio matches a rectangle
"""
def check_ratio(area,width,height,min_aspect_ratio=4,max_aspect_ratio=8,verbose=False):
    ratio = float(width)/float(height)
    boolean = False
    if ratio < 1:
        ratio = 1 / ratio 
    if verbose:
        print("5) Ratio After Mininum Bounding Rectangle: {}".format(ratio)) # Display the ratio
    if ratio >= min_aspect_ratio and ratio <= max_aspect_ratio:
        if verbose:
            print("6) The Area Is Given By Minimum Bounding Rectangle: {}".format(area))
        boolean = True
    return boolean
        
    
"""
Input:
rect: The minimum bounding rectangle found from the contour

--------------------
"""
def check_orientation(rect,min_aspect_ratio=4,max_aspect_ratio=8,verbose=False):
    (x,y),(width,height),angle = rect
    boolean = False
    if verbose:
        print("1) Coordinates (X,Y): ({},{})".format(x,y))
        print("2) Dimensions (Width,Height): ({},{})".format(width,height))
        print("3) Angle : {}".format(angle))
    if width > height: #CV2 returns a negative angle
        angle =- angle
    else:
        angle = 90 + angle 
    if height == 0  or width == 0:
        boolean = False
    area = width * height
    if verbose:
        print("4) Area: {}".format(area))
    return check_ratio(area,width,height,min_aspect_ratio=min_aspect_ratio,max_aspect_ratio=max_aspect_ratio,verbose=verbose)
    

"""
Input
Cnt: Contour detailing continuous points
Precision: Maximum distance from contour to approximated contour (It is a percentage of the arc length)
Verbose: Print all the debug information
---------------------------
Return: Approximate the number of sides of a contour and return the result
"""
def get_number_of_edges(contour,precision=0.02,closed=True,verbose=False):
    perimeter = cv2.arcLength(contour,closed) # Closed
    edges_count = cv2.approxPolyDP(contour,precision * perimeter,closed) # Closed
    num_edges = len(edges_count)
    if verbose:
        print("A) The Approximate Amount of Sides Is: {}".format(num_edges))
    return num_edges


"""
Input: Edge Detected Image
Edge Detected Image: Image that has undergone edge detection such as canny 
Keep: How many contours to keep 
-----------------------------
Return: A list of contours(continuous points with the same intensity)
"""
def find_license_plate_contours(edge_detected_image,keep=10):
    cnts = cv2.findContours(edge_detected_image.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts,key=cv2.contourArea,reverse=True)[:keep]
    return cnts



"""
Input: Contours of an Image (take the first n:)
Image: Original RGB Image
Contours: Contours arranged in descending order
Precision: Maximum distance from contour to approximated contour (It is a percentage of the arc length)
Min_Asept_Ratio: 
Max_Asept_Ratio: 
Closed: Is the contour a closed object ?
Verbose: Print all the debug information
-----------------------------

Return: Image cropped to define a license plate 
"""
def find_license_plate(image,cnts,precision=0.02,min_aspect_ratio=4,max_aspect_ratio=8,closed=True,verbose=False):
    plate = image.copy()
    found = False
    least_possible_plates_coordinates = []
    for cnt in cnts: # Loop through the largest contours
        num_edges = get_number_of_edges(cnt,precision,closed,verbose) # Get the number of edges surrounding the contours
        (x,y,w,h) = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/float(h) 
        if verbose:
            print("B) Width,Height For Current Contour Before Minimum Bounding Rectangle: ({},{})".format(w,h))
            print("C) Aspect Ratio For Current Contour Before Minimum Bounding Rectangle: {}".format(aspect_ratio))
        if num_edges == 4:
            if verbose:
                print("D) The Potential Coordinates For The License Plate: ({}:{},{}:{})".format(y,y+h,x,x+w))
            possible_plates = image[y:y+h,x:x+w]
            least_possible_plates_coordinates.append(possible_plates)            
        if (aspect_ratio >= min_aspect_ratio and aspect_ratio <= max_aspect_ratio) or num_edges == 4: # Check that the contour is a rectangle
            if verbose:
                print("0) Cropped Image Coordinates: ({}:{},{}:{})".format(y,y+h,x,x+w))
            min_rect = cv2.minAreaRect(cnt)
            if(check_orientation(rect=min_rect,min_aspect_ratio=min_aspect_ratio,max_aspect_ratio=max_aspect_ratio,verbose=verbose)): # The contour has the properties of a rectange
                plate = image[y:y+h,x:x+w]
                found = True
                break
        if verbose:
            print("7) The Current Contour Is Likely A License Plate: {}".format(found))
            print("============================")
    if verbose:
        print("******************************************************************")            
        print("8) The License Plate Was Found: {}".format(found))
        print("******************************************************************")            
    return plate,found,least_possible_plates_coordinates
            

    
"""
Run OCR to check the license plate was actually found
"""
def get_license_text(plate_image):
    psm = 7
    alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    options = "-c tessedit_char_whitelist={}".format(alphanumeric)
    options += " --psm {}".format(psm)
    text = pytesseract.image_to_string(plate_image,config=options)
    text = str(re.split('\n',text)[0])
    return text
    
def license_plate_detection(type_="type1",verbose=False):
    pass

