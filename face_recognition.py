import sys
import os
import dlib
import glob
from dataLoader import *

# Does all the preliminary face recognition tasks. Put into a function so it isn't run everytime
# face_recognition.py is imported
def initializeFaceRecognition():
    # Hard coded paths because I couldn't get command line args to work:
    predictor_path = "shape_predictor_5_face_landmarks.dat"
    face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
    faces_folder_path = "./images"

    # Load all the models we need: a detector to find the faces, a shape predictor
    # to find face landmarks so we can precisely localize the face, and finally the
    # face recognition model.
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)

    win = dlib.image_window()

    return predictor_path, face_rec_model_path, faces_folder_path, detector, sp, facerec, win

# Creates a vector for one image for testing purposes:
def processOneImage(filename):
    predictor_path, face_rec_model_path, faces_folder_path, detector, sp, facerec, win = initializeFaceRecognition()
    img = dlib.load_rgb_image(filename)
    faces_folder_path = filename
    #win.clear_overlay()
    #win.set_image(img)
    dets = detector(img, 1)
    #print(dets)
    #print("Number of faces detected: {}".format(len(dets)))
    # Now process each face we found.
    
    for k, d in enumerate(dets):
        #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
         #   k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = sp(img, d)
        # Draw the face landmarks on the screen so we can see what face is currently being processed.
        #win.clear_overlay()
        #win.add_overlay(d)
        #win.add_overlay(shape)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        #print(face_descriptor)
        #print("Computing descriptor on aligned image ..")

        # Let's generate the aligned image using get_face_chip
        face_chip = dlib.get_face_chip(img, shape)

        # Now we simply pass this chip (aligned image) to the api
        face_descriptor_from_prealigned_image = facerec.compute_face_descriptor(face_chip)
        return face_descriptor_from_prealigned_image
# Function to read csv file and find names of images we've already processed:
def findLoadedImages():
    vectorFile = open("vectors.csv", 'r')
    loadedImages = set()
    for line in vectorFile:
        line = line.split(',')
        # add img name to loadedImages
        loadedImages.add(line[3])
    vectorFile.close()
    return loadedImages

# Runs through a images directory, processing every image inside of it.
# Most of the code copied from http://dlib.net/face_recognition.py.html:
def process_images():
    predictor_path, face_rec_model_path, faces_folder_path, detector, sp, facerec, win = initializeFaceRecognition()
    dicts = createImageDictionaries()

    # Now process all the images
    i = 0
    loadedImages = findLoadedImages()
    vectorFile = open("vectors.csv", 'a')
    for dict in dicts:
        # skip loading images we've already loaded:
        if dict['imgname'] in loadedImages:
            print("skipping", dict['imgname'])
            i+=1
            continue

        # File directory:
        f = "./images\\" + dict["imgname"]
        print("Processing file: {}".format(f))
        img = dlib.load_rgb_image(f)

        win.clear_overlay()
        win.set_image(img)
        dets = detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))

        # Now process each face we found.
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = sp(img, d)
            # Draw the face landmarks on the screen so we can see what face is currently being processed.
            win.clear_overlay()
            win.add_overlay(d)
            win.add_overlay(shape)
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            print(face_descriptor)

            print("Computing descriptor on aligned image ..")

            # Let's generate the aligned image using get_face_chip
            face_chip = dlib.get_face_chip(img, shape)

            # Now we simply pass this chip (aligned image) to the api
            face_descriptor_from_prealigned_image = facerec.compute_face_descriptor(face_chip)
            print(face_descriptor_from_prealigned_image)

            # Add 128d vector to the dictionary:
            dict["vector"] = face_descriptor_from_prealigned_image

            #dlib.hit_enter_to_continue()

            # Write the vector to a new csv file. (I forgot to include writing the pixels)
            vectorFile.write(str(dict['age']) + ',' + str(dict['ethnicity']) + ',' + str(dict['gender']) + ',' + str(dict["imgname"]) + ',')
            for val in dict["vector"][:-1]:
                vectorFile.write(str(val))
                vectorFile.write(',')
            for val in dict["vector"][-1:]:
                vectorFile.write(str(val))
                vectorFile.write('\n')
        print("IMAGE", i, "DONE")
        i+=1
    vectorFile.close()

    # Pickle the dictionary containing all the vectors:
    outfile = open("newDicts", 'wb')
    pickle.dump(dicts, outfile)
    outfile.close()
    return dicts

