#ALL IN ONE
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
import os
import subprocess,time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.optim import AdamW

# Define desired image size
img_size = (100, 100)

kept = ''

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

    # Use k-means algorithm to cluster images based on their classification scores
    if len(scores) >= n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, n_init=20)
        kmeans.fit(scores)

        # Create 2D list of image file paths, where each inner list corresponds to a cluster
        clusters = [[] for _ in range(n_clusters)]
        for i, label in enumerate(kmeans.labels_):
            clusters[label].append(image_paths[i])
    else:
        clusters = [image_paths]

    # Return 2D list of image file paths
    return clusters


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
                type_text(kept)
            else:
                type_text(text)
        elif command.startswith("press_and_hold_key_combination"):
            keys = command.split()[1:]
            for key in keys:
                pyautogui.keyDown(key)
            for key in keys:
                pyautogui.keyUp(key)
        elif command.startswith("wait"):
            keys = command.split()[1:]
            time.sleep(int(keys[0]))
        elif command.startswith("keep"):
            kept = command.split()[1:]
        


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
    "keep hello",
    "type_text -k"
    
]

print('100%')
    
