import csv
import numpy as np
import random
from PIL import Image
import pickle


def createRepDicts(reps, testType):
    dict_string = "{}Dict".format(testType)
    try:
        infile = open(dict_string, 'rb')
        repsD = pickle.load(infile)
        infile.close()
        return repsD
    except IOError:
        outfile = open(dict_string, 'wb')
        pickle.dump(reps, outfile)
        outfile.close()
    return
# Turn csv file into list of dictionaries containing values for pixels, ethnicity, age, etc:
# This is the new version that has the pix and HOG data in the dictionary
def createHOGDicts():
    # Attempt to read pickle file called "hogDict" before reading CSV...
    # Can read pickle file MUCH faster than CSV. FASTER IS BETTER.
    try:
        infile = open("hogDict", 'rb')
        images = pickle.load(infile)
        infile.close()
        return images
    except IOError:
        print("No newDicts pickle file found. Creating it...")
        images = []

        with open('vectors.csv', 'r') as csvfile:
            data = csv.reader(csvfile, delimiter=',', quotechar='|')
            fields = next(data)
            for row in data:
                HOG = []
                age = row[0]
                ethnicity = row[1]
                gender = row[2]
                imgname = row[3]

                # Turn list of strings into list of ints:
                for i in range(4,132):
                    HOG.append(float(row[i]))

                imageDict = {'age': int(age), 'ethnicity': int(ethnicity), 'gender': int(gender), 'imgname': imgname,
                             'HOG': HOG, 'pix': [],

                             'class': 0}
                images.append(imageDict)

            # Write imageDict to pickled file to make initial data loading faster:
            outfile = open("hogDict", 'wb')
            pickle.dump(images, outfile)
            outfile.close()
            return images

# Turn csv file into list of dictionaries containing values for pixels, ethnicity, age, etc:
def createImageDictionaries():
    # Attempt to read pickle file called "imageDicts" before reading CSV...
    # Can read pickle file MUCH faster than CSV. FASTER IS BETTER.
    try:
        infile = open("imageDicts", 'rb')
        images = pickle.load(infile)
        infile.close()
        return images
    except IOError:
        print("No imageDicts pickle file found. Creating it...")
        images = []
        with open('age_gender.csv', 'r') as csvfile:
            data = csv.reader(csvfile, delimiter=',', quotechar='|')
            fields = next(data)
            for row in data:
                age = row[0]
                ethnicity = row[1]
                gender = row[2]
                imgname = row[3]
                pixels = row[4].split()

                # Turn list of strings into list of ints:
                for i in range(len(pixels)):
                    pixels[i] = int(pixels[i])

                imageDict = {'age': int(age), 'ethnicity': int(ethnicity), 'gender': int(gender), 'imgname': imgname, 'pix': pixels,


                             'class': 0}
                images.append(imageDict)

            # Write imageDict to pickled file to make initial data loading faster:
            outfile = open("imageDicts", 'wb')
            pickle.dump(images, outfile)
            outfile.close()
            return images


# Creates babies and oldies without HOG data:
def getBabiesOldies():
    images = createImageDictionaries()
    output = []
    for image in images:
        if image['age'] < 2 or image['age'] > 90:
            if image['age'] < 3:
                image['age'] = 0
            if image['age'] > 90:
                image['age'] = 1
            output.append(image) 
    return output

# Creates babies and oldies images with the HOG data:
def getBabiesOldiesHOG():
    images = createHOGDicts()
    output = []
    for image in images:
        if image['age'] < 2 or image['age'] >= 80:
            if image['age'] < 3:
                image['age'] = 0
            if image['age'] >= 80:
                image['age'] = 1
            output.append(image)
    return output

def getBabiesMiddiesOldiesHOG():
    images = createHOGDicts()
    output = []

    for image in images:
        if image['age'] < 2 or image['age'] >= 80:
            if image['age'] < 3:
                image['age'] = 0

            if image['age'] >= 80:
                image['age'] = 2
            output.append(image)
        if image['age'] >= 25 and image['age'] <= 30:
            image['age'] = 1
            output.append(image)
    return output


#given size of sample, returns random sample from data
def getRandomSampleHOG(size):
    images = createHOGDicts()
    selections = random.sample(range(len(images)), size)
    output = []
    for i in range(0, len(selections)):
        output.append(images[selections[i]])
    
    return output

def loadData():
    rows = []
    images = []
    pixelData = []
    ethnicity = []
    with open('./dataset/age_gender.csv', 'r') as csvfile:
        data = csv.reader(csvfile, delimiter=',', quotechar='|')
        
        fields = next(data)
        

        for row in data:
            rows.append(row) 
            images.append(row[4])
            pixelData.append(row[4])
            ethnicity.append(row[1])
        
        for j in range(len(images)):
        # Turn space separated string of numbers into list of number strings:
            images[j] = images[j].split() # "123 234 212" -> ['123, '234', '212']
            pixelData[j] = pixelData[j].split()
        # Turn every string number into an integer:
            for i in range(len(images[j])):
                images[j][i] = int(images[j][i])
                pixelData[j][i] = int(pixelData[j][i])

        # Turn list of lists of pixels into a list of dictionaries containing entries for
        # a list of pixels ('pix) and a class value ('class'):
            images[j] = {'pix': images[j], 'class': 0}
        
        return rows, images, ethnicity, pixelData
        #48*48 images


# function to merge the HOG dicts with the pixel dicts (probably not needed anymore):
def addPixelsToHOG():
    # Create
    HOGDICT = createHOGDicts()
    PIXDICT = createImageDictionaries()
    try:
        infile = open("hogDict", 'rb')
        HOGDICT = pickle.load(infile)
        infile.close()
        print("SUCCESSFULLY READ PICKLE")
    except IOError:
        print("MAKING PICKLE")
        for hdict in HOGDICT:
            for pdict in PIXDICT:
                if hdict['imgname'] == pdict['imgname']:
                    hdict['pix'] = pdict['pix']

        outfile = open("hogDict", 'wb')
        pickle.dump(HOGDICT, outfile)
        outfile.close()

    newVectors = open("newVectors.csv", 'w')

    newVectors.write('age' + ','+ 'ethnicity' + ',' + 'gender' + ',' + 'image Name' + "\n")
    for dict in HOGDICT:
        newVectors.write(str(dict['age']) + ',' + str(dict['ethnicity']) + ',' + str(dict['gender']) + ',' + str(dict['imgname']) + ',')
        for val in dict['HOG']:
            newVectors.write(str(val) + ',')
        for i in range(len(dict['pix'])):
            dict['pix'][i] = str(dict['pix'][i])

        print("DICT PIX", dict['pix'])
        pixString = " ".join(dict['pix'])

        print("PIX STRING", pixString)
        newVectors.write(pixString + "\n")
    newVectors.close()


# function to pickle a list of reps so they don't have to be created again:
def saveReps(reps, fileName):
    outfile = open(fileName + ".pkl", 'wb')
    pickle.dump(reps, outfile)
    outfile.close()


# load a pickled list of reps from a given file name
def loadReps(fileName):
    try:
        infile = open(fileName, 'rb')
        reps = pickle.load(infile)
        infile.close()
        return reps
    except IOError:
        print("Error reading", fileName)
        return []
