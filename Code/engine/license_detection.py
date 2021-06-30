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
min_aspect_ratio = 2;max_aspect_ratio = 8

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
    
#============================ Generalized License Plate Detection With Hyperparameter Tuning
"""
Perform stepwise license plate detection
"""
def detect_license_plate(img,gray,contours,min_aspect_ratio=min_aspect_ratio,max_aspect_ratio=max_aspect_ratio,verbose=False):
    morphological_image = en.morphological_preprocessing(gray)
    morphological_contours = find_license_plate_contours(morphological_image)
    plate,found,__ = find_license_plate(img,morphological_contours,min_aspect_ratio=min_aspect_ratio,max_aspect_ratio=max_aspect_ratio,verbose=False)
    if (found == False) or len(get_license_text(plate)) <= 2:
        if verbose:
            print("Morphological Operation: No License Plate Detected")
            print("Attempt License Plate Detection Using Canny Edge Detection .....")
            print("==============================================================")
        blur = cv2.bilateralFilter(gray,11,90,90)
        edges = cv2.Canny(blur,30,200)
        contours = find_license_plate_contours(edges)
        _,found,__ = find_license_plate(img,contours,min_aspect_ratio=min_aspect_ratio,max_aspect_ratio=max_aspect_ratio,verbose=False)#(image,cnts,precision=0.02,min_aspect_ratio=4,max_aspect_ratio=8,closed=True,verbose=False)
        if found == True and len(get_license_text(_)) >=2:
            if verbose:
                print("License Plate Detection Using Canny Edge Detection Operation: License Plate Detected")
            plate = _

        for pt in __:
            if verbose:
                print("Checking Possible Plates:")
            if len(get_license_text(pt)) >2:
                plate = pt # License Text = '\x0c'
                found = True
                plt.imshow(plate,cmap='gray')
                return plate,found

        if (found == False) or len(get_license_text(plate)) <= 2:
            if verbose:
                print("License Plate Detection Using Canny Edge Detection Operation: No License Plate Detected")
                print("Attempt License Plate Detection Using Canny Edge Detection By Blurring The Canny Edge Image .....")
                print("==============================================================")
            edges_blur = cv2.GaussianBlur(edges,(5,5),0)
            contours = find_license_plate_contours(edges_blur)
            _,found,__ = find_license_plate(img,contours,min_aspect_ratio=min_aspect_ratio,max_aspect_ratio=max_aspect_ratio,verbose=False)
            if found == True and len(get_license_text(_)) > 2:
                plate = _    
            else:
                if verbose:
                    print("No License Plate")
                plate = img.copy()
                found = False 
                return plate,found
    return plate,found



"""
Given a list of gamma values, loop through each one and illuminate or darken the image and detect a license
Return an image that best approximates a license plate if it exists
"""
def gamma_hyperparameter_tuning(img,gray,contours,min_aspect_ratio=min_aspect_ratio,max_aspect_ratio=max_aspect_ratio,verbose=True):
    if verbose:
        print("Performing Gamma Hyper parameter tuning")
    gammas = np.linspace(0.5,1,6)
    plate = img.copy()
    found = False
    for gamma in gammas:
        adjusted_gray = en.adjust_gamma(gray,gamma)
        morphological_image = en.morphological_preprocessing(adjusted_gray)
        morphological_contours = find_license_plate_contours(morphological_image)
        plate,found = detect_license_plate(img,adjusted_gray,morphological_contours,min_aspect_ratio=min_aspect_ratio,max_aspect_ratio=max_aspect_ratio,verbose=verbose)
        if found:
            break
    return plate,found,gamma



"""
Adjust the intensities of the image based on the hyperparameters of the histogram equalization, gamma values, and normal blurred gray image.
If the license plate exists, then perform license plate detection on all these hyperparameters and choose the best one that approximates the license plate
if it does not exist then return the image itself

"""
def hyperparameter_tuning(img,gray,contours,min_aspect_ratio=min_aspect_ratio,max_aspect_ratio=max_aspect_ratio,verbose=True):
    hist = en.hist_eq_(gray).astype('uint8')
    hist_tuning = detect_license_plate(img,hist,contours,min_aspect_ratio=min_aspect_ratio,max_aspect_ratio=max_aspect_ratio,verbose=verbose)
    gray_tuning = detect_license_plate(img,gray,contours,min_aspect_ratio=min_aspect_ratio,max_aspect_ratio=max_aspect_ratio,verbose=verbose)
    gamma_tuning = gamma_hyperparameter_tuning(img,gray,contours,min_aspect_ratio=min_aspect_ratio,max_aspect_ratio=max_aspect_ratio,verbose=verbose)
    
    hist_plates = (hist_tuning[0],hist_tuning[1],len(get_license_text(hist_tuning[0])))
    gray_plates = (gray_tuning[0],gray_tuning[1],len(get_license_text(gray_tuning[0])))
    gamma_plates = (gamma_tuning[0],gamma_tuning[1],len(get_license_text(gamma_tuning[0])))
    plates = [hist_plates,gray_plates,gamma_plates]
    max_ = 0
    current_plate = None
    for best_plate in plates:
        if best_plate[1] == True and best_plate[2] > max_:
            current_plate = best_plate
            max_ = best_plate[2]
    if current_plate == None:
        return img.copy(),False
    else:
        return current_plate,True