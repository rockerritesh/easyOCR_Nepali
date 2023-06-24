## OCR PDF Text Extraction

This code snippet demonstrates how to extract text from a PDF file using Optical Character Recognition (OCR). The extracted text is saved to separate text files for each page and combined into a single text file.

### Requirements

The following libraries are required to run the code:
- matplotlib
- cv2 (OpenCV)
- easyocr
- numpy
- pylab
- IPython
- pdf2image
- shutil
- os

You can install these dependencies using the following command:
```python
pip install matplotlib opencv-python easyocr numpy pdf2image
```

### Code Explanation

1. Import the necessary libraries:
```python
import matplotlib.pyplot as plt
import cv2
import easyocr
import numpy as np
from pylab import rcParams
from IPython.display import Image
```

2. Define helper functions for image processing:
- `map()`: Maps a value from one range to another range.
- `highPassFilter()`: Applies a high-pass filter to the image.
- `blackPointSelect()`: Selects black points in the image based on a threshold value.
- `whitePointSelect()`: Selects white points in the image based on a threshold value.
- `blackAndWhite()`: Converts the image to black and white using LAB color space.
- `scan_effect()`: Applies a scanning effect to the image using the above helper functions.

3. Initialize the OCR reader:
```python
reader = easyocr.Reader(['hi'])
```

4. Specify the path to the PDF folder and create a temporary directory to store the image files:
```python
path_folder = "book"
directory_path = "temp"

if not os.path.exists(directory_path):
    os.makedirs(directory_path)
    print(f"Directory {directory_path} created successfully.")
else:
    print(f"Directory {directory_path} already exists.")
```

5. Loop through each PDF file in the specified folder:
- Convert the PDF to images using `convert_from_path()` from `pdf2image`.
- Iterate over the images and perform OCR and image processing operations.
- Save the extracted text for each page to a separate text file.
- Remove the temporary image file.
- Combine all the extracted text into a single text file.
```python
for path in os.listdir(path_folder):
    pdf_path = path_folder + '/' + path
    print(pdf_path)

    images = convert_from_path(pdf_path)
    text = ""

    for i, image in enumerate(images):
        image_path = f"temp/image_{i+1}.jpg"
        image.save(image_path, "JPEG")

        image = cv2.imread(image_path)
        image = scan_effect(image)
        output = reader.readtext(image)
        os.remove(image_path)

        for i in range(len(output)):
            text += output[i][1]

        text_file_path = f"temp/text_{i+1}.txt"
        with open(text_file_path, "w", encoding="utf-8") as text_file:
            text_file.write(text)

        print(f"Text from page {i+1} saved.")

    combined_text_file_path = f"combined_text{path}.txt"
    with open(combined_text_file_path, "w", encoding="utf-8") as combined_text_file:
        combined_text_file.write(text)

    print("Combined text saved.")
```

6. Remove the temporary directory:
```python
folder_path = "temp"

if os.path.exists(folder_path):
    shutil.rmtree(folder_path)
    print(f"Folder {folder_path} and its contents have been successfully removed.")
else:
    print(f"Folder {folder_path} does not exist.")
```

This code allows you to extract text from multiple PDF files and save them as

 separate text files, as well as combine all the extracted text into a single text file. Remember to specify the correct paths for the PDF files and adjust the OCR language if necessary.