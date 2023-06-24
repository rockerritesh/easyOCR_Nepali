import matplotlib.pyplot as plt
import cv2
import easyocr
import numpy as np
from pylab import rcParams
from IPython.display import Image

def map(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def highPassFilter(img,kSize):
    if not kSize%2:
        kSize +=1
    kernel = np.ones((kSize,kSize),np.float32)/(kSize*kSize)
    filtered = cv2.filter2D(img,-1,kernel)
    filtered = img.astype('float32') - filtered.astype('float32')
    filtered = filtered + 127*np.ones(img.shape, np.uint8)
    filtered = filtered.astype('uint8')
    return filtered

def blackPointSelect(img, blackPoint):
    img = img.astype('int32')
    img = map(img, blackPoint, 255, 0, 255)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_TOZERO)
    img = img.astype('uint8')
    return img

def whitePointSelect(img,whitePoint):
    _,img = cv2.threshold(img, whitePoint, 255, cv2.THRESH_TRUNC)
    img = img.astype('int32')
    img = map(img, 0, whitePoint, 0, 255)
    img = img.astype('uint8')
    return img

def blackAndWhite(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    (l,a,b) = cv2.split(lab)
    img = cv2.add( cv2.subtract(l,b), cv2.subtract(l,a) )
    return img
def scan_effect(img):
    
    blackPoint = 66
    whitePoint = 130
    image = highPassFilter(img,kSize = 51)
    image_white = whitePointSelect(image, whitePoint)
    img_black = blackPointSelect(image_white, blackPoint)
    image=blackPointSelect(img,blackPoint)
    white = whitePointSelect(image,whitePoint)
    img_black = blackAndWhite(white)
    return img_black

reader = easyocr.Reader(['hi'])


from pdf2image import convert_from_path
import shutil
import os

# Path to the PDF file
path_folder = "book"

# Path to the directory
directory_path = "temp"

# Create the directory if it doesn't exist
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
    print(f"Directory {directory_path} created successfully.")
else:
    print(f"Directory {directory_path} already exists.")
    
    
    
for path in os.listdir(path_folder):
    pdf_path = path_folder+'/' + path
    print(pdf_path)
    
    # Convert PDF to images
    images = convert_from_path(pdf_path)

    text = " "

    for i, image in enumerate(images):

        image_path = f"temp/image_{i+1}.jpg"  # Output image path and format
        image.save(image_path, "JPEG")

        image = cv2.imread(image_path)

        image = scan_effect(image)

        output = reader.readtext(image)
        
        os.remove(image_path)
        
        for i in range(len(output)):
            text += output[i][1]

        #print(text)
        # Save the text to a separate text file
        text_file_path = f"temp/text_{i+1}.txt"
        with open(text_file_path, "w", encoding="utf-8") as text_file:
            text_file.write(text)

        print(f"Text from page {i+1} saved.")
        
        #break;
        
    # Save the combined text to a single text file
    combined_text_file_path = f"combined_text{path}.txt"
    with open(combined_text_file_path, "w", encoding="utf-8") as combined_text_file:
        combined_text_file.write(text)

    print("Combined text saved.")
    
    #break;
    
 
# Path to the folder to be removed
folder_path = "temp"

# Verify if the folder exists
if os.path.exists(folder_path):
    # Remove the folder and its contents
    shutil.rmtree(folder_path)
    print(f"Folder {folder_path} and its contents have been successfully removed.")
else:
    print(f"Folder {folder_path} does not exist.")
