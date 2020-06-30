#!/usr/bin/env python3

# main.py
# Face recognition project
# Eric Biscocho, Anthony DiFalco
# CSE 40535, Computer Vision Fall 2018

import os, argparse
import cv2 as cv2
import numpy as np

from matplotlib import pyplot as plt
from skimage import feature
from PIL import Image

##############################  GLOBAL VARS/FUNCS  ##############################
# Subject image path (Yale Face Database); can change to training/validation sets.
imgPath = "./yalefaces"

# Viola-Jones' Haar Cascade for feature extraction; can change to different frontal face cascades.
cascadePath = "./haarcascades/"
faceCascade = cv2.CascadeClassifier(cascadePath + "haarcascade_frontalface_alt.xml")

# LBP-based features by linear SVM for classification
classifier = cv2.face.LBPHFaceRecognizer_create()

def return_histogram(image):
    '''
    Get the histogram from the LBP face recognizer for each image.
    '''
    localBinaryPattern = feature.local_binary_pattern(image, 24, 8, method="uniform")
    hist, _ = np.histogram(localBinaryPattern.ravel(), bins=np.arange(1,255), range=(1,255))
    hist = hist.astype("float")
    hist = hist/(hist.sum()+1e-7)
    return hist

def save_histogram(np_hist, label):
    '''
    Save each image's histogram to folder.
    '''
    bin_comp = [i for i in range(1,255)]
    plt.hist(np_hist, bins=bin_comp)
    plt.title(label)
    count = 1
    while (1):
        if os.path.exists("histograms/{}-{}.png".format(label, count))==True:
            count += 1
        else:
            plt.savefig("histograms/{}-{}.png".format(label, count))
            break

def get_images(path, num):
    '''
    Training will be based on all configurations except for the user inputted one;
    the user selected configuration will be used in testing.
    '''
    if num == 0:
        img_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith(img_config) and f != ".DS_Store"]
    elif num > 0:
        if num < 10:
            temp = '0' + str(num)
            img_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith(img_config) and f.startswith("subject" + temp) and f != ".DS_Store"]
        elif num > 9 and num < 16:
            img_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith(img_config) and f.startswith("subject" + str(args.n)) and f != ".DS_Store"]

    images = [] # stores the face images
    labels = [] # stores the subject associated with an image

    for img in img_paths:
        imageBW = Image.open(img).convert('L') # Convert the image to BW.
        image = np.array(imageBW, "uint8") # Convert the image into numpy array.
        subject = int(os.path.split(img)[1].split('.')[0].replace("subject", '')) # Get the subject associated with an image.
        faces = faceCascade.detectMultiScale(image) # Detect the subject's face.

        for (x, y, w, h) in faces:
            # If a face is detected, then append that image and its label.
            images.append(image[y:y+h, x:x+w])
            labels.append(subject)

            # Get the histogram of a given image and save it to a folder.
            hist, _ = np.histogram(image[y:y+h, x:x+w], bins=np.arange(1,255), range=(1,255))
            save_histogram(hist, subject)

            cv2.imshow("Adding faces to training set...", image[y:y+h,x:x+w])
            cv2.waitKey(50)

    return images, labels

def imgConfig(argument):
    '''
    Parse command line argument input.
    '''
    choice = {
        'c':    ".centerlight",
        'g':    ".glasses",
        "ha":   ".happy",
        'l':    ".leftlight",
        "ng":   ".noglasses",
        'n':    ".normal",
        'r':    ".rightlight",
        's':    ".sad",
        'sl':   ".sleepy",
        'su':   ".surprised",
        'w':    ".wink"
    }
    return choice[argument]

##############################  MAIN EXECUTION  ##############################
if __name__ == "__main__":
    print('Parsing args...')
    parser = argparse.ArgumentParser()
    parser.add_argument("-configure", nargs='?', default='r', type=str, help="image configuration to test on")
    parser.add_argument("-n", nargs='?', default=11, type=int, help="number of subjects")
    args = parser.parse_args()
    img_config = imgConfig(args.configure)
    num = args.n
    print('Getting images ...')
    images, subjects = get_images(imgPath, num)
    cv2.destroyAllWindows()
    print('Training ...')
    classifier.train(images, np.array(subjects)) # Train the classifier to recognize the faces of subjects.

    # Test the classifier using the image configuration chosen by the user.
    if num == 0: # all subjects
        img_paths = [os.path.join(imgPath, f) for f in os.listdir(imgPath) if f.endswith(img_config) and f != ".DS_Store"]
    elif num > 0: # one subject
        if num < 10:
            temp = '0' + str(num)
            img_paths = [os.path.join(imgPath, f) for f in os.listdir(imgPath) if f.endswith(img_config) and f.startswith("subject" + temp) and f != ".DS_Store"]
        elif num > 9 and num < 16:
            img_paths = [os.path.join(imgPath, f) for f in os.listdir(imgPath) if f.endswith(img_config) and f.startswith("subject" + str(args.n)) and f != ".DS_Store"]

    for path in img_paths:
        imgPredictBW = Image.open(path).convert('L')
        imgPredict = np.array(imgPredictBW, "uint8")
        faces = faceCascade.detectMultiScale(imgPredict)
        for (x, y, w, h) in faces:
            predSubject, conf = classifier.predict(imgPredict[y:y+h, x:x+w])
            realSubject = int(os.path.split(path)[1].split('.')[0].replace("subject", ''))
            if realSubject == predSubject:
                print("Subject {} '{}' is recognized with a confidence level of {}%.".format(realSubject, img_config.strip('.'), round(conf,2)))
            else:
                print("Subject {} '{}' is mistaken for Subject {}.".format(realSubject, img_config.strip('.'), predSubject))
            cv2.imshow("Recognizing Subject " + str(realSubject) + "...", imgPredict[y:y+h, x:x+w])
            cv2.waitKey(1000)
