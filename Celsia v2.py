#ALL IN ONE
'''
Project CELSIA or Computervision Enhanced Lightweight System Interface for Automation
is an project hosted by Osmond Fan in 2022. The target is to use this program to
let everyone create their own universal chatbot that doesn't break any copyright
policies of any APPs by not breaking into their base codes for information, while
capable of determining a large range of Computer UIs.
'''
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from joblib import dump, load
from sklearn.cluster import KMeans
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import tensorflow as tf
import nltk
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.corpus import stopwords
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
import numpy as np
import heapq
from concurrent.futures import ThreadPoolExecutor
from annoy import AnnoyIndex
import pyautogui
import subprocess,time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.optim import AdamW
import pytesseract
from pytesseract import Output
from PIL import ImageGrab, ImageChops
#Activate Tesseract
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# Define desired image size
img_size = (100, 100)

kept = ''
imgkept = ''

def load_images_from_folder(folder):
    """
    Load and resize images from the specified folder.

    :param folder: The path to the folder containing the images to load.
    :return: A tuple containing a list of loaded and resized images and a list of their corresponding file paths.
    """
    images = []
    image_paths = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isdir(file_path):
            subfolder_images, subfolder_image_paths = load_images_from_folder(file_path)
            images.extend(subfolder_images)
            image_paths.extend(subfolder_image_paths)
        elif filename.endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(file_path, 0)
            img = cv2.resize(img, img_size)
            images.append(img)
            image_paths.append(file_path)
    return images, image_paths

def train_model(folder, model_file):
    """
    Train a model for the specified folder and save it to the specified file.

    :param folder: The path to the folder containing the training data.
    :param model_file: The path to the file where the trained model will be saved.
    """
    # Load and resize training data
    images, image_paths = load_images_from_folder(folder)
    images = np.array(images, dtype=object)

    # Check if there are enough images
    if len(images) > 0:
        # Normalize pixel values
        images = images.astype('float32') / 255.0

        # Create CNN model
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        # Compile CNN model using SGD optimizer from tf.keras.optimizers.legacy
        opt = tf.keras.optimizers.legacy.SGD()
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        # Convert images array to float32
        images = images.astype(np.float32)

        # Train CNN model
        try:
            history = model.fit(images.reshape(len(images), img_size[0], img_size[1], 1), np.ones(len(images)), epochs=2, batch_size=150)
            # Save trained model to file
            print(model_file, 'here')
            model.save(model_file)
        except Exception as e:
            print(e)

from collections import namedtuple

TextBox = namedtuple('TextBox', ['x', 'y', 'width', 'height'])

def merge_text_boxes(text_boxes, max_x_diff=25, max_y_diff=150):
    """
    Merge text boxes that are close to each other.

    :param text_boxes: A list of text boxes represented as (x, y, width, height) tuples.
    :param max_x_diff: The maximum difference in x-coordinates for two text boxes to be considered for merging.
    :param max_y_diff: The maximum difference in y-coordinates for two text boxes to be considered for merging.
    :return: A list of merged text boxes.
    """
    # Convert text boxes to named tuples
    text_boxes = [TextBox(*box) for box in text_boxes]

    # Sort text boxes by their y-coordinates
    text_boxes = sorted(text_boxes, key=lambda box: box.y)

    # Initialize list of merged boxes
    merged_boxes = []

    # Merge text boxes
    for box in text_boxes:
        merged = False
        for i, merged_box in enumerate(merged_boxes):
            if abs(box.x - merged_box.x) <= max_x_diff and abs(box.y - merged_box.y) <= max_y_diff:
                merged_boxes[i] = TextBox(
                    min(box.x, merged_box.x),
                    min(box.y, merged_box.y),
                    max(box.x + box.width, merged_box.x + merged_box.width) - min(box.x, merged_box.x),
                    max(box.y + box.height, merged_box.y + merged_box.height) - min(box.y, merged_box.y)
                )
                merged = True
                break
        if not merged:
            merged_boxes.append(box)

    # Continue merging until no more merges are possible
    while True:
        new_merged_boxes = []
        for box in merged_boxes:
            merged = False
            for i, merged_box in enumerate(new_merged_boxes):
                if abs(box.x - merged_box.x) <= max_x_diff and abs(box.y - merged_box.y) <= max_y_diff:
                    new_merged_boxes[i] = TextBox(
                        min(box.x, merged_box.x),
                        min(box.y, merged_box.y),
                        max(box.x + box.width, merged_box.x + merged_box.width) - min(box.x, merged_box.x),
                        max(box.y + box.height, merged_box.y + merged_box.height) - min(box.y, merged_box.y)
                    )
                    merged = True
                    break
            if not merged:
                new_merged_boxes.append(box)
        if len(new_merged_boxes) == len(merged_boxes):
            break
        else:
            merged_boxes = new_merged_boxes

    # Remove any gigantic boxes that completely cover other boxes
    final_merged_boxes = []
    for box in new_merged_boxes:
        is_gigantic = False
        for other_box in new_merged_boxes:
            if other_box != box and \
                    other_box.x >= box.x and other_box.x + other_box.width <= box.x + box.width and \
                    other_box.y >= box.y and other_box.y + other_box.height <= box.y + box.height:
                is_gigantic = True
                break
        if not is_gigantic:
            final_merged_boxes.append(box)

    return final_merged_boxes

import numpy as np
import cv2

def detect_color_clusters(image, color_range=((0, 160), (200, 250), (0, 120)),min_area=200):
    # Create masks for each color channel
    masks = [cv2.inRange(image[:,:,i], lower_bound, upper_bound) for i, (lower_bound, upper_bound) in enumerate(color_range)]
    
    # Combine the masks
    mask = cv2.bitwise_and(*masks)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area (number of pixels)
    min_contour_area = min_area  # Minimum number of pixels for a contour to be considered a "cluster"
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    # Create a copy of the image to draw on
    image_copy = image.copy()
    
    # Initialize a list to store the bounding boxes
    bounding_boxes = []
    
    # Draw rectangles around the color clusters
    for cnt in contours:
        # Calculate the bounding rectangle for the contour
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Check if there are larger bounding boxes that contain this box
        is_inside = False
        for other_box in bounding_boxes:
            other_x, other_y, other_w, other_h = other_box
            if x > other_x and y > other_y and x + w < other_x + other_w and y + h < other_y + other_h:
                is_inside = True
                break
        
        # If not inside any larger box, add the bounding box to the list
        if not is_inside:
            bounding_boxes.append((x, y, w, h))
            
            # Draw the rectangle on the image copy
            cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0,255,0), 2)
    
    # Display the image with rectangles drawn on it
    #cv2.imshow('Image with Rectangles', image_copy)
    #cv2.waitKey(0)

    return bounding_boxes



def ocr_only(screenshot):
    # Convert the image to grayscale
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    # Use Tesseract to do OCR on the image
    text = pytesseract.image_to_string(gray, config='--psm 6 -l eng+chi_sim+chi_tra')

    return text



def detect_text_only(screenshot, model_file=False, max_x_diff=25, max_y_diff=150, bg_color=None):
    # Convert screenshot to grayscale
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    # Initialize lists for text bounding boxes and OCR text
    text_boxes = []
    ocr_text = []

    # Double the size of the image
    screenshot = cv2.resize(screenshot, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)

    # Use PyTesseract to detect text boxes and OCR text
    d = pytesseract.image_to_data(gray, output_type=Output.DICT, config='--psm 6 -l eng+chi_sim')
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        text = d['text'][i]
        text_boxes.append((x,y,w,h))
        ocr_text.append(text)

    # Merge text boxes
    merged_text_boxes = merge_text_boxes(text_boxes, max_x_diff, max_y_diff)
    merged_ocr_text = []
    for big_box in merged_text_boxes:
        bx1, by1, bw, bh = big_box
        bx2, by2 = bx1 + bw, by1 + bh
        small_boxes_inside = [small_box for small_box in text_boxes if small_box[0] >= bx1 and small_box[1] >= by1 and small_box[0]+small_box[2] <= bx2 and small_box[1]+small_box[3] <= by2]
        merged_text = ' '.join([ocr_text[text_boxes.index(small_box)] for small_box in small_boxes_inside])
        merged_ocr_text.append(merged_text)

    return merged_text_boxes, merged_ocr_text


def detect_buttons_text(screenshot, model_file, max_x_diff=25, max_y_diff=150):
    # Load pre-trained model from file
    clf = load(model_file)
    
    # Convert screenshot to grayscale
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    # Find contours in screenshot
    _, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize lists for button and text bounding boxes
    button_boxes = []
    text_boxes = []

    # Classify each contour as either a button or text
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, img_size)
        if clf.predict(roi_resized.reshape(1,-1)) == 1:
            button_boxes.append((x,y,w,h))
        else:
            #text_boxes.append((x,y,w,h))
            pass

    # Use PyTesseract to detect text boxes
    d = pytesseract.image_to_data(gray, output_type=Output.DICT, config='--psm 6 -l eng+chi_sim')
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        text_boxes.append((x,y,w,h))

    # Merge text boxes
    merged_text_boxes = merge_text_boxes(text_boxes, max_x_diff, max_y_diff)

    # Remove large text boxes that contain smaller text boxes
    filtered_text_boxes = []
    for big_box in merged_text_boxes:
        bx1, by1, bw, bh = big_box
        bx2, by2 = bx1 + bw, by1 + bh
        small_boxes_inside = [small_box for small_box in merged_text_boxes if small_box != big_box and small_box[0] >= bx1 and small_box[1] >= by1 and small_box[0]+small_box[2] <= bx2 and small_box[1]+small_box[3] <= by2]
        if not small_boxes_inside:
            filtered_text_boxes.append(big_box)
        else:
            x_diffs = [abs(small_box[0] - big_box[0]) for small_box in small_boxes_inside]
            y_diffs = [abs(small_box[1] - big_box[1]) for small_box in small_boxes_inside]
            if all(x_diff <= max_x_diff for x_diff in x_diffs) and all(y_diff <= max_y_diff for y_diff in y_diffs):
                filtered_text_boxes.append(big_box)

    return button_boxes, filtered_text_boxes

def detect_buttons_text2(screenshot, model_file, max_x_diff=25, max_y_diff=150):
    # Load pre-trained model from file
    clf = load(model_file)
    
    # Convert screenshot to grayscale
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    # Find contours in screenshot
    _, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize lists for button and text bounding boxes
    button_boxes = []
    text_boxes = []

    # Classify each contour as either a button or text
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, img_size)
        if clf.predict(roi_resized.reshape(1,-1)) == 1:
            button_boxes.append((x,y,w,h))
        else:
            pass

    # Use PyTesseract to detect text boxes
    d = pytesseract.image_to_data(gray, output_type=Output.DICT, config='--psm 6 -l eng+chi_sim')
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        text_boxes.append((x,y,w,h))

    # Merge text boxes
    merged_text_boxes = merge_text_boxes(text_boxes, max_x_diff, max_y_diff)

    # Remove large text boxes that contain smaller text boxes
    filtered_text_boxes = []
    for big_box in merged_text_boxes:
        bx1, by1, bw, bh = big_box
        bx2, by2 = bx1 + bw, by1 + bh
        small_boxes_inside = [small_box for small_box in merged_text_boxes if small_box != big_box and small_box[0] >= bx1 and small_box[1] >= by1 and small_box[0]+small_box[2] <= bx2 and small_box[1]+small_box[3] <= by2]
        if not small_boxes_inside:
            filtered_text_boxes.append(big_box)
        else:
            x_diffs = [abs(small_box[0] - big_box[0]) for small_box in small_boxes_inside]
            y_diffs = [abs(small_box[1] - big_box[1]) for small_box in small_boxes_inside]
            if all(x_diff <= max_x_diff for x_diff in x_diffs) and all(y_diff <= max_y_diff for y_diff in y_diffs):
                filtered_text_boxes.append(big_box)

    # Scan all the merged text boxes with OCR and store the results
    ocr_results = []
    for box in filtered_text_boxes:
        x, y, w, h = box
        roi = gray[y:y+h, x:x+w]
        ocr_result = pytesseract.image_to_string(roi)
        ocr_results.append(ocr_result)

    return button_boxes, filtered_text_boxes, ocr_results

import cv2

def findocr(image, color_range=((0, 160), (200, 250), (0, 120)), max_x_diff=25, max_y_diff=150):
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Detect color clusters
    bounding_boxes = detect_color_clusters(image, color_range)

    # Initialize a list to store the OCR results
    ocr_results = []

    # Draw the bounding boxes on the image
    for x, y, w, h in bounding_boxes:
        # Crop the image to the bounding box
        cropped_image = image[y:y+h, x:x+w]

        # Detect text in the cropped image
        ocr_text = ocr_only(cropped_image)

        # Replace "\n" with an actual new line
        ocr_text = ocr_text.replace("\\n", "\n")

        # Check if the text is not just whitespace or non-alphanumeric characters
        if ocr_text.strip() and any(char.isalnum() for char in ocr_text):
            # Add the OCR text to the results list
            ocr_results.append(ocr_text)
            
            # Draw a bounding box around the region where OCR was performed
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    print(ocr_results)
    # Show the image with bounding boxes
    #cv2.imshow("OCR Bounding Boxes", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return ocr_results

def replace_content(match):
    return f"Celsia: {match.group(1)}"

def dbfindocr(image, color_range=((0, 160), (200, 250), (0, 120)), max_x_diff=25, max_y_diff=150):
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Detect color clusters
    bounding_boxes = detect_color_clusters(image, color_range)

    # Initialize a list to store the OCR results
    ocr_results = []

    # Draw the bounding boxes on the image
    for x, y, w, h in bounding_boxes:
        # Crop the image to the bounding box
        cropped_image = image[y:y+h, x:x+w]

        # Detect text in the cropped image
        ocr_text = ocr_only(cropped_image)

        # Replace "\n" with an actual new line
        ocr_text = ocr_text.replace("\\n", "\n")

        # Check if the text is not just whitespace or non-alphanumeric characters
        if ocr_text.strip() and any(char.isalnum() for char in ocr_text):
            # Add the OCR text to the results list
            ocr_results.append(ocr_text)
            
            # Draw a bounding box around the region where OCR was performed
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    print(ocr_results)
    # Show the image with bounding boxes
    cv2.imshow("OCR Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return ocr_results

TextBox = namedtuple('TextBox', ['x', 'y', 'width', 'height'])

def capture_screenshot():
    """
    Capture a screenshot and return it as a NumPy array.

    :return: A NumPy array representing the captured screenshot.
    """
    # Capture screenshot using pyautogui
    screenshot = pyautogui.screenshot()

    # Convert screenshot to NumPy array
    screenshot = np.array(screenshot)

    # Convert screenshot from RGB to BGR
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

    return screenshot

def find_contours(screenshot, model_file):
    """
    Find contours in the specified screenshot and classify them as either text or buttons using a pre-trained model.

    :param screenshot: A NumPy array representing the screenshot to find contours in.
    :param model_file: The path to the file containing the pre-trained model.
    :return: A tuple containing a list of button bounding boxes and a list of text bounding boxes.
    """
    # Load pre-trained model from file
    clf = load(model_file)

    # Convert screenshot to grayscale
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    # Find contours in screenshot
    _, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize lists for button and text bounding boxes
    button_boxes = []
    text_boxes = []

    # Classify each contour as either a button or text
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (100, 100))
        if clf.predict(roi_resized.reshape(1,-1)) == 1:
            button_boxes.append((x,y,w,h))
            # Draw green rectangle around button on screenshot
            cv2.rectangle(screenshot, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            text_boxes.append((x,y,w,h))
            # Draw red rectangle around text on screenshot
            cv2.rectangle(screenshot, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Merge text boxes using merge_text_boxes function
    text_boxes = merge_text_boxes(text_boxes)

    return button_boxes, text_boxes

def add_legend(image):
    # Define colors for legend
    button_color = (0, 255, 0)
    text_color = (255, 0, 0)

    # Define font for legend
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Create a white background for the legend
    legend_width = 300
    legend_height = image.shape[0]
    legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255

    # Add text to the legend
    cv2.putText(legend,'Legend',(10,40), font, 1,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(legend,'Buttons',(10,80), font, 1,button_color,2,cv2.LINE_AA)
    cv2.putText(legend,'Text',(10,120), font, 1,text_color,2,cv2.LINE_AA)

    # Concatenate the legend and the image
    image_with_legend = np.concatenate((image, legend), axis=1)

    return image_with_legend


def classify_images(folder, model_folder, n_clusters=5, new_only=False):
    """
    Classify images in the specified folder using the specified model and a k-means algorithm.

    :param folder: The path to the folder containing the images to classify.
    :param model_folder: The path to the folder containing the trained model.
    :param n_clusters: The number of clusters to form using the k-means algorithm.
    :param new_only: Whether to classify only images in a subfolder named "new".
    :return: A 2D list of image file paths, where each inner list corresponds to a cluster and contains the file paths of the images assigned to that cluster.
    """
    # Load trained model from file
    model_file = os.path.join(folder, os.path.basename(folder) + '.h5')
    model = load_model(model_file)

    # Load and resize images from specified folder
    if new_only:
        folder = os.path.join(folder, 'new')
    images, image_paths = load_images_from_folder(folder)
    images = np.array(images, dtype=object)

    # Normalize pixel values
    images = images.astype('float32') / 255.0

    # Obtain classification scores for each image
    scores = model.predict(images.reshape(len(images), img_size[0], img_size[1], 1), batch_size=200)

import os
import subprocess
import time
from PIL import Image
import cv2

def capture_window_with_index(index=1,application="Finder"):
    print(application)
    applescript = f'tell application "System Events" to get id of window {index} of application "'+application+'"'
    window_id = subprocess.run(["osascript", "-e", applescript], stdout=subprocess.PIPE, text=True).stdout.strip()
    output_path = f"screenshot_{index}.png"
    subprocess.run(["screencapture", "-l", window_id, output_path])

    # Wait for a short time to ensure the file is fully created
    time.sleep(1)

    # Open the captured image using OpenCV and return it
    captured_image = cv2.imread(output_path)
    #print(captured_image)
    return captured_image



def show_image(img):
    # Display the image using the show method
    img.show()

def OCR(model_file="contours.joblib",screenshot=None):
    # Debugging code: print the value of the screenshot variable

    # Find contours in screenshot and classify them as either text or buttons using pre-trained model
    text_boxes, text_info = detect_text_only(screenshot, model_file)

    # Draw red rectangles around text on screenshot
    for box in text_boxes:
        x, y, w, h = box
        cv2.rectangle(screenshot, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display image of boxed contours
    
    #cv2.imshow('Screenshot', screenshot)
    return text_boxes, text_info

def OCR2(model_file="contours.joblib",screenshot=None, color_range=((0, 160), (200, 250), (0, 120))):
    # Debugging code: print the value of the screenshot variable

    # Find contours in screenshot and classify them as either text or buttons using pre-trained model
    text_info = findocr(screenshot)

    return text_info

print('20%')

def remove_empty_folders_recursively(directory):
    """
    Remove and delete empty folders in the specified directory and all of its subdirectories.

    :param directory: The path to the directory to remove empty folders from.
    """
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path):
            # Recursively remove empty subfolders
            remove_empty_folders_recursively(folder_path)
            # Remove folder if it is empty
            if not os.listdir(folder_path):
                os.rmdir(folder_path)

def train_model_recursively(folder, model_folder, max_depth=None, depth=0):
    """
    Train a model for the specified folder and its subdirectories and save it to the specified file.

    :param folder: The path to the folder containing the training data.
    :param model_folder: The path to the folder where the trained models will be saved.
    :param max_depth: The maximum depth of recursion. If None, recursion will continue until all subdirectories have been processed.
    :param depth: The current depth of recursion.
    """
    # Train model for current folder
    model_file = os.path.join(model_folder, os.path.basename(folder) + '.h5')
    train_model(folder, model_file)

    # Recursively train models for subdirectories
    if max_depth is None or depth < max_depth:
        for subfolder in os.listdir(folder):
            subfolder_path = os.path.join(folder, subfolder)
            if os.path.isdir(subfolder_path):
                model_folder = subfolder_path
                print(model_folder,subfolder_path)
                #print(subfolder_path,folder,subfolder,model_folder)
                train_model_recursively(subfolder_path, model_folder, max_depth, depth + 1)


def classify_images_recursively(folder, model_folder, n_clusters=5, max_depth=None, depth=0):
    """
    Classify images in the specified folder and its subdirectories using the specified model and a k-means algorithm.

    :param folder: The path to the folder containing the images to classify.
    :param model_folder: The path to the folder containing the trained models.
    :param n_clusters: The number of clusters to form using the k-means algorithm.
    :param max_depth: The maximum depth of recursion. If None, recursion will continue until all subdirectories have been processed.
    :param depth: The current depth of recursion.
    :return: A dictionary where the keys are folder paths and the values are 2D lists of image file paths, where each inner list corresponds to a cluster and contains the file paths of the images assigned to that cluster.
    """
    # Classify images in current folder
    clusters = classify_images(folder, model_folder, n_clusters)
    result = {folder: clusters}

    # Recursively classify images in subdirectories
    if max_depth is None or depth < max_depth:
        for subfolder in os.listdir(folder):
            subfolder_path = os.path.join(folder, subfolder)
            if os.path.isdir(subfolder_path):
                result.update(classify_images_recursively(subfolder_path, model_folder, n_clusters, max_depth, depth + 1))

    # Return result
    return result

#CELSI = Computational Emotion Learning and Sentiment Interface



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

import concurrent.futures

def load_model(path):
    return BertForSequenceClassification.from_pretrained(path)
'''
model_paths = ['/Users/osmond/Desktop/CELSI/Celsi077KB', '/Users/osmond/Desktop/CELSI/terrified', '/Users/osmond/Desktop/CELSI/anger', '/Users/osmond/Desktop/CELSI/happy', '/Users/osmond/Desktop/CELSI/embarassed', '/Users/osmond/Desktop/CELSI/romantic']

with concurrent.futures.ThreadPoolExecutor() as executor:
    results = executor.map(load_model, model_paths)

main_model, fear_model, anger_model, happy_model, shy_model, romance_model = results
'''
class EmotionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text, label = self.data[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

print('40%')

def train(model, data_loader, optimizer, device):
    model.train()
    
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        loss = outputs.loss
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

def evaluate(model, data_loader, device):
    model.eval()
    
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            predictions = torch.argmax(outputs.logits, dim=1)
            correct_predictions += torch.sum(predictions == labels)
            total_predictions += labels.shape[0]
    
    return correct_predictions / total_predictions

def fine_tune_emotion_classification(train_data, val_data, n_classes, n_epochs=25):
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=n_classes)
    
    # Initialize data loaders
    train_dataset = EmotionDataset(train_data, tokenizer, max_length=128)
    train_data_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    val_dataset = EmotionDataset(val_data, tokenizer, max_length=128)
    val_data_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # Fine-tune model
    for epoch in range(n_epochs):
        train(model, train_data_loader, optimizer, device)
        accuracy = evaluate(model, val_data_loader, device)
        print(f'Epoch {epoch + 1}/{n_epochs} | Validation Accuracy: {accuracy:.4f}')
    
    return model

# Define the index meanings
index_dict = {0: "happy", 1: "fear", 2: "anger", 3: "embarassed", 4: "flirtish", 5: "lovestruck", 6: "confused", 7: "emotionless",8:"caring",9:"disgusted",10:"jealous",11:"guilty"}

# Define the csv file name
csv_file = "stories.csv"

def classify_emotion(query_message, tokenizer, model):
    encoding = tokenizer.encode_plus(
        query_message,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    prediction = torch.argmax(outputs.logits, dim=1)
    return prediction.item()

'''
# Example usage
query_message = "I am so happy today!"
emotion = classify_emotion(query_message, tokenizer, main_model)
print(emotion)
'''

nlp = spacy.load('en_core_web_sm')
print('60%')
def get_keywords(text, cache):
    if text in cache:
        return cache[text]
    
    doc = nlp(text)
    
    keywords = []
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN', 'VERB']:
            keywords.append(token.text.lower())
    
    stop_words = set(stopwords.words('english'))
    keywords = [word for word in keywords if word not in stop_words]
    
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words([token.text for token in doc])
    bigrams = finder.nbest(bigram_measures.pmi, 10)
    keywords.extend([' '.join(bigram) for bigram in bigrams])
    
    cache[text] = keywords
    return keywords

def calculate_weight(message, sender_messages, cache):
    message_time = datetime.strptime(message[1], '%Y-%m-%d %H:%M:%S')
    recent_messages = [m for m in sender_messages if abs((datetime.strptime(m[1], '%Y-%m-%d %H:%M:%S') - message_time).total_seconds()) <= 5 * 3600]
    recent_keywords = [get_keywords(m[2], cache) for m in recent_messages]
    keyword_counts = [sum([k.count(keyword) for k in recent_keywords]) for keyword in get_keywords(message[2], cache)]
    weight = sum(keyword_counts)
    return weight

class ChatDatabase:
    def __init__(self, filename):
        self.filename = filename
        self.messages = []
        with open(filename,mode='a') as file:
            file.write('')
        self.load_messages()
        self.index = None
        

    
    def load_messages(self):
        with open(self.filename, 'r') as f:
            for line in f:
                sender, time, text = line.strip().split('\t', 2)
                message = (sender, time, text)
                self.messages.append(message)
    
    def add_message(self, sender, time, text):
        message = (sender, time, text)
        self.messages.append(message)
        with open(self.filename, 'a') as f:
            f.write(f'{sender}\t{time}\t{text}\n')
    
    def build_index(self, cache, n_trees=10):
        # Get TF-IDF vectors for all messages
        sender_messages = [message for message in self.messages if message[0] == sender]
        messages_keywords = [' '.join(get_keywords(message[2], cache)) for message in sender_messages]
        
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(messages_keywords)
        
        # Build Annoy index
        self.index = AnnoyIndex(tfidf_matrix.shape[1], 'angular')
        for i in range(tfidf_matrix.shape[0]):
            self.index.add_item(i, tfidf_matrix[i].toarray()[0])
        self.index.build(n_trees)

    
    def get_relevant_messages(self, sender, query, N, cache, n_threads=30):
        # Get all messages from sender and extract keywords
        sender_messages = [message for message in self.messages if message[0] == sender]
        
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            messages_keywords = list(executor.map(lambda message: ' '.join(get_keywords(message[2], cache)), sender_messages))
        
        # Get TF-IDF vector for query
        query_keywords = ' '.join(get_keywords(query, cache))
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(messages_keywords + [query_keywords])
        query_vector = tfidf_matrix[-1].toarray()[0]
        
        # Find most relevant messages using Annoy index
        relevant_indices = self.index.get_nns_by_vector(query_vector, N)
        
        relevant_messages = [sender_messages[i] for i in relevant_indices]
        return relevant_messages

db = ChatDatabase('messages.txt')
print('80%')

'''
query = 'where should we go?'
sender = 'Alice'
N = 10
cache = {}

db.build_index(cache, n_trees=5)

relevant_messages = db.get_relevant_messages(sender, query, N, cache)

for message in relevant_messages:
    print(message)
'''

from transformers import BertForSequenceClassification, BertTokenizer
import re

# Define the path to the directory where the model and tokenizer were saved
saved_model_path = "/Users/osmond/Desktop/CELSI/usefulness"

# Load the saved model
model = load_model("/Users/osmond/Desktop/CELSI/usefulness")

def GUT(query):
    query = str(query)
    query_message = query.split('.')
    #Split by every clause of sentence
    final = []
    for query in query_message:
        if classify_emotion(query, tokenizer, model) == 1:
            final.append(query)
            
    #0 = Not useful, 1 = useful
    
    return " ".join(final)

import platform
import subprocess

def activate_window(title):
    if platform.system() == "Darwin":
        script = f'''
        tell application "System Events"
            set frontmost of (first process whose front window's title contains "{title}") to true
        end tell
        '''
        subprocess.run(["osascript", "-e", script])
    else:
        import pyautogui
        windows = pyautogui.getWindowsWithTitle(title)
        if windows:
            windows[0].activate()

def exit_fullscreen(app_name):
    if platform.system() == "Darwin":
        script = f'''
        tell application "System Events"
            set frontmostProcess to first process whose frontmost is true
            set frontmostApp to name of frontmostProcess
            tell process "{app_name}"
                set value of attribute "AXFullScreen" of window 1 to false
            end tell
        end tell
        '''
        subprocess.run(["osascript", "-e", script])
    else:
        import pyautogui
        pyautogui.hotkey('win', 'down')
from screeninfo import get_monitors

def get_screen_dimensions():
    monitor = get_monitors()[0]  # Assuming there is only one monitor
    return monitor.width, monitor.height

def wait_until_loaded(x=0, y=0, width=0, height=0, threshold=10, max_wait=60):
    if x == y == width == height == 0:
        width, height = get_screen_dimensions()
        monitor_region = (0, 0, width, height)
    else:
        monitor_region = (x,y,width,height)
    reference_screen = pyautogui.screenshot(region=monitor_region).convert('L')  # Convert to grayscale
    print(width,height)
    while max_wait > 0:
        current_screen = pyautogui.screenshot(region=monitor_region).convert('L')  # Convert to grayscale
        
        # Convert PIL Images to numpy arrays
        reference_array = np.array(reference_screen)
        current_array = np.array(current_screen)

        # Calculate the absolute difference
        diff_image = np.abs(reference_array - current_array)

        # Sum all the differences. If they are more than zero, then there is a change
        if np.sum(diff_image)/(width*height) > threshold:
            print(np.sum(diff_image)/(width*height))
            break
        else:
            print(np.sum(diff_image)/(width*height))
        
        time.sleep(1)  # Wait for a second before checking again
        max_wait -= 1
        reference_screen = current_screen

    
import platform
import subprocess

def make_fullscreen(app_name):
    if platform.system() == "Darwin":
        script = f'''
        tell application "System Events"
            set frontmostProcess to first process whose frontmost is true
            set frontmostApp to name of frontmostProcess
            tell process "{app_name}"
                set value of attribute "AXFullScreen" of window 1 to true
            end tell
        end tell
        '''
        #print(script)
        subprocess.run(["osascript", "-e", script])
    else:
        import pyautogui
        pyautogui.hotkey('win', 'up')

import pyperclip,re

def move_mouse(x, y):
    pyautogui.moveTo(x, y)

def click_mouse(x, y):
    pyautogui.click(x, y)

def move_mouse_relative(x_offset, y_offset):
    pyautogui.moveRel(x_offset, y_offset)

def drag_mouse(x, y):
    pyautogui.dragTo(x, y)

def drag_mouse_relative(x_offset, y_offset):
    pyautogui.dragRel(x_offset, y_offset)

def press_key(key):
    pyautogui.press(key)

def press_key_combination(*keys):
    pyautogui.hotkey(*keys)

def type_text(text, interval=0):
    if re.search("[\u4e00-\u9FFF]", text):
        pyperclip.copy(text)
        pyautogui.hotkey("command", "v")
    else:
        pyautogui.typewrite(text, interval=interval)

def press_and_hold_key_combination(*keys):
    for key in keys:
        pyautogui.keyDown(key)

def release_key_combination(*keys):
    for key in keys:
        pyautogui.keyUp(key)

def scroll_mouse(amount):
    pyautogui.scroll(amount)

def keep(info):
    kept = info

def crop_image(image, x, y, w, h):
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image
    
'''background'''
# Define a function that runs an AppleScript command and returns the output
def run_applescript(command):
    # Use osascript to run the command
    process = subprocess.run(["osascript", "-e", command], capture_output=True)
    # Decode and strip the output
    output = process.stdout.decode().strip()
    # Return the output
    return output

#To go to Edge browser to ask bing
# Get a list of all window names
window_names = run_applescript('tell application "System Events" to get name of every window of every process')


'''N Script Area'''
def N(commands):
    global kept
    for command in commands:
        if command.startswith("move_mouse"):
            x, y = map(int, command.split()[1:])
            move_mouse(x, y)
        elif command.startswith("click_mouse"):
            x, y = map(int, command.split()[1:])
            click_mouse(x, y)
        elif command.startswith("move_mouse_relative"):
            x_offset, y_offset = map(int, command.split()[1:])
            move_mouse_relative(x_offset, y_offset)
        elif command.startswith("drag_mouse"):
            x, y = map(int, command.split()[1:])
            drag_mouse(x, y)
        elif command.startswith("drag_mouse_relative"):
            x_offset, y_offset = map(int, command.split()[1:])
            drag_mouse_relative(x_offset, y_offset)
        elif command.startswith("press_key"):
            key = command.split()[1]
            press_key(key)
        elif command.startswith("press_key_combination"):
            keys = command.split()[1:]
            press_key_combination(*keys)
        elif command.startswith("type_text"):
            text = ' '.join(command.split()[1:])
            if text == "-k":
                type_text(" ".join(kept))
                #print(kept[0])
            else:
                type_text(text)
        elif command.startswith("press_and_hold_key_combination"):
            keys = command.split()[1:]
            for key in keys:
                pyautogui.keyDown(key)
            for key in keys:
                pyautogui.keyUp(key)
        elif command.startswith("wait"):
            args = command.split()[1:]
            print(args)
            if len(args) == 1:
                # Wait for the specified number of seconds
                seconds = float(args[0])
                time.sleep(seconds)
            else:
                # Call the wait_until_loaded function
                threshold = int(args[0]) if len(args) > 0 else 10
                max_wait = float(args[1]) if len(args) > 1 else 10
                wait_until_loaded(threshold=threshold, max_wait=max_wait)
        elif command.startswith("keep"):
            kept = command.split()[1:]
            #print(kept)
        elif command.startswith("Contour"):
            return OCR()
        elif command.startswith("window_text"):
            args = command.split()[1:]
            if len(args) == 6:
                index, application, x_origin, y_origin, x_width, y_height = args
                #print(index,application)
                index = int(index)
                x_origin = int(x_origin)
                y_origin = int(y_origin)
                x_width = int(x_width)
                y_height = int(y_height)
                if index == 0:
                    index = 1
                ans1 = capture_window_with_index(index, application)
                return crop_image(ans1,x_origin,y_origin,x_width,y_height)
            else:
                index, application = args
                if index == 0:
                    index = 1
                return capture_window_with_index(index, application)

        elif command.startswith("show_image"):
            show_img(imgkept)
        elif command.startswith("get_useful_text"):
            text = command.split()[1:]
            #print(text)
            ans = GUT(text)
            return ans
        elif command.startswith("switch_window"):
            name = command.split()[1:]
            activate_window(name[0])
            print(name[0])
        elif command.startswith("fullscreen"):
            app_name = command.split()[1].strip('"')
            #print(app_name)
            make_fullscreen(app_name)
        elif command.startswith("un_fullscreen"):
            app_name = command.split()[1].strip('"')
            #print(app_name)
            exit_fullscreen(app_name)
        


            
        
contour = "Contour"
show_image = "show_image"


open_bing = [
    "press_and_hold_key_combination command space",
    "release_key_combination command space",
    "type_text edge",
    "press_key enter",
    "press_key enter",
    "type_text www.bing.com ",
    "press_key enter",
    "wait 2",
    "type_text Hi",
    "press_key enter",
    "wait 2",
    "scroll_mouse 2000"
]

open_chatgpt = [
    "press_and_hold_key_combination command space",
    "release_key_combination command space",
    "type_text safari",
    "press_key enter",
    "press_key enter",
    "press_and_hold_key_combination command n",
    "release_key_combination command n",
    "type_text chat.openai.com ",
    "press_key enter",
    
]

print('100%')
    
