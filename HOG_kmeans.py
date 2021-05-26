import csv
from face_recognition import processOneImage
import random
import random
from PIL import Image
from dataLoader import createRepDicts, loadData, createImageDictionaries, getBabiesMiddiesOldiesHOG, getRandomSampleHOG
#from face_recognition import *
from helper import *
from statistics import mode
import pickle

''' k means algorithm steps:
1. pick random representatives
2. find closest representative for each data point, give it a class
3. find average position of each data point in each class, make that the new representative
'''


# function to track convergence of algorithm
def Jclust(reps, images):
    sum = 0
    for rep in reps:
        for image in images:
            if rep['class'] == image['class']:
                for j in range(len(image['HOG'])):
                    sum = sum + abs(image['HOG'][j] - rep['HOG'][j]) ** 2
    return (sum / len(images))

#experimental function to map blind labels to corresponding label in original dataset
def mapLabels(images, clusterNum, testType, reps):
    print("Mapping labels to class")
    inferred_labels = {}
    for i in range(clusterNum):
        labels = []
        # 1. Make array of all the images where class = i
        def isPartOfClass(image):
            if (image['class'] == i):
                return True
        index = list(filter(isPartOfClass, images))
        # 2. Find the actual label for each image and append it to new array
        labels = [image[testType] for image in index]
        # ignore clusters that have no images in them:
        if len(labels) == 0:
            #print("length of labels is 0")
            continue

        # 3. Find most common label for that cluster using actual labels
        most_common_label = mode(labels)

        # 4. Append that label to a dictionary. Key = most common label, value: class
        if most_common_label in inferred_labels:
            inferred_labels[most_common_label].append(i)
        else:
            inferred_labels[most_common_label] = [i]

    # Change class values to actual label from inferred_labels:
    for image in images:

        # iterate over all the actual labels:
        for actual_label in inferred_labels:

            # find what actual_label corresponds to that cluster:
            if image['class'] in inferred_labels[actual_label]:

                # assign that label to the class:
                image['class'] = actual_label
                break
    print(inferred_labels)

    for rep in reps:
        for actual_label in inferred_labels:
            if rep['class'] in inferred_labels[actual_label]:
                rep['class'] = actual_label
                break



#given the reps of a completed kmeans and a set of hogdicts, predict what their class would be
#this function will not work with current setup. 
#maybe define a dictionary, set HOG to input val, then run


def predict(reps, inputHOG):
    min = meanSquareDistance(reps[0]['HOG'], inputHOG)
    cluster = reps[0]['class']
    for i in range(1, len(reps)):
        it = meanSquareDistance(reps[i]['HOG'], inputHOG)
        if it < min:
            min = it
            cluster = reps[i]['class']
    return cluster


#Kmeans now takes in clusterNum as input, so we can run it on gender, ethnicity, or age easily
# Attempted implementation of the kmeans algorithm the professor showed in class:
def kmeans(imageInput, trainingSetSize, inputReps, clusterNum):
    print('running k means!')
    clusterNumber = clusterNum  # number of expected clusters, will probably be in the order of hundreds
    # iterations count should be dependent on convergence of Jclust function
    images = []
    trainingSetIndicies = []
    # pick k random images from images, using all of them takes forever:
    # images = random.choices(images, k = 1000)

    ##Data selection, tracking, then normalization

    set = list(range(0, len(imageInput)))
    # set = list(range(0, len(imageInput)-1))
    random.shuffle(set)
    for i in range(trainingSetSize):
        images.append(imageInput[set[i]])
        trainingSetIndicies.append(set[i])

    # PART 1: CREATE [clusterNumber] RANDOM REPRESENTATIVES:
    reps = []
    JClusterResults = []
    if len(inputReps) == 0:
        for i in range(clusterNumber):
            randomInt = random.randint(0, len(images) - 1)
            # Grab a random image from images, copy its HOGels, and create a new dictionary from it:
            rep = {'pix': images[randomInt]['pix'].copy(), 'HOG': images[randomInt]['HOG'].copy(), 'class': i, 'indice': trainingSetIndicies[i] }
            reps.append(rep)
            #displayImage(rep['pix'])
    else:
        reps = inputReps
        #for rep in reps:
            #displayImage(rep['pix'])

    # Do the next two parts [iterationCount] times:
    loop = 0
    converged = False
    while (converged == False):
        loop += 1
        # PART 2: ASSIGN CLASS TO EACH DATA POINT (IMAGE) BASED ON CLOSEST REPRESENTATIVE:
        # Find minimum distance from an image to a representative:
        for i in range(len(images)):
            # Initialize minimum distance to the distance to the first representative:
            #displayImage(images[i]['pix'])
            minDist = meanSquareDistance(reps[0]['HOG'], images[i]['HOG'])
            images[i]['class'] = reps[0]['class']

            # Loop over every representative but the first one (already checked it):
            for rep in reps[1:]:
                dist = meanSquareDistance(rep['HOG'], images[i]['HOG'])
                # if we found a closer representative: update the image to belong to its class:
                if dist < minDist:
                    minDist = dist
                    images[i]['class'] = rep['class']

        # PART 3: FIND AVERAGE POSITION OF EACH IMAGE IN EACH CLASS, MAKE THAT THE NEW REPRESENTATIVE:

        for i in range(len(reps)):
            # calculate average:
            reps[i]['HOG'] = findClassAvgHOG(reps[i], images)
        JClusterResults.append(Jclust(reps, images))
        # if a couple of iterations have gone by and the Jclust function isn't decreasing by more than .5% we have converged
        if len(JClusterResults) > 4:
            if (JClusterResults[len(JClusterResults) - 1] / JClusterResults[len(JClusterResults) - 2] > .999):
                converged = True

                # Return the reps so we can see how they were changed:
    return reps, JClusterResults, images, trainingSetIndicies


def accuracy_testAge(images, trainingSet):
    
    count = 0

    #predicting age
    for i in range(len(trainingSet)):
        if int(images[i]['class']) == int(images[i]['age']) or (abs(int(images[i]['class']) - int(images[i]['age'])) < 7):
            count += 1
            
    return count/len(trainingSet)

def accuracy_testEthnicity(images, trainingSet):
    
    count = 0
    #de-normalizing!
    for image in images:
        image['pix'] = denormalizeImage(image['pix'])

    #predicting age
    for i in range(len(trainingSet)):
        if int(images[i]['class']) == int(images[i]['ethnicity']):
            count += 1
            
    return count/len(trainingSet)

def accuracy_testGender(images, trainingSet):
    
    count = 0

    #predicting age
    for i in range(len(trainingSet)):
        if int(images[i]['class']) == int(images[i]['gender']):
            count += 1
            
    return count/len(trainingSet)


def accuracy_test(images, trainingSet, testType):
    if testType == 'age':
        return accuracy_testAge(images, trainingSet)
    elif testType == 'ethnicity':
        return accuracy_testEthnicity(images, trainingSet)
    elif testType == 'gender':
        return accuracy_testGender(images, trainingSet)
    else:
        print('test types are: age, gender, ethnicity. You specified none of them')
        
    return

def predetermineReps(imageData, inputReps):
    for i in range(2):
        set_ = [0, len(imageData) - 1]
        rep = {'pix': imageData[set_[i]]['pix'].copy(), 'HOG': imageData[set_[i]]['HOG'].copy(), 'class': i}
        inputReps.append(rep)
    return inputReps

# Function to run kmeans many times and return the best result:
#changed kMeans to have test type as input to make it more general
def iterateKmeans(imageInput, trainingSetSize, testType, clusterNum, maxIterations = 50):
    accuracies = []
    bestAccuracy = 0
    bestReps = []
    bestDicts = []
    worstDicts = []
    
    imageInput = getDistributedData(imageInput, clusterNum, testType)

    for i in range(maxIterations):
        reps, JClustResults, imageResults, trainingIndices = kmeans(imageInput, trainingSetSize, [], clusterNum)
        mapLabels(imageResults, clusterNum, testType, reps)
        accuracy = accuracy_test(imageResults, trainingIndices, testType)
        print("Accuracy of kmeans test", i + 1, ":", accuracy)
        accuracies.append(accuracy)
        if accuracy > bestAccuracy:
            bestReps = reps
            bestAccuracy = accuracy
            bestDicts = imageResults
            bestTrainingIndices = trainingIndices
        # Grab inverted label dictionaries
        if accuracy <= 0.05:
            worstDicts = imageResults
            print("Inverted dictionary accuracy:", accuracy)
    accuracies.sort(reverse = True)
    print("Top three accuracies:", accuracies[0], accuracies[1], accuracies[2])
    return bestReps, bestDicts, bestTrainingIndices
    # Uncomment to predict the class of one face using distance to best reps:

    # closestRep = testOneFace(bestReps, processOneImage("old_dude.jpg"))
    # print(closestRep["class"])
    # closestRep = testOneFace(bestReps, processOneImage("baby.jpg"))
    # print(closestRep["class"])

#ok in this function need to go thru reps and maps its class value to actual value
def kMeansDicts():
    print('here cool')
    ageReps, ageDicts, ageIndices = iterateKmeans(randomSample, 300, 'age', 50, 50)
    genderReps, genderDicts, genderIndices = iterateKmeans(randomSample, 100, 'gender', 30, 25)
    ethnicityReps, ethnicityDicts, ethnicityIndices = iterateKmeans(randomSample, 300, 'ethnicity', 50, 50)
    createRepDicts(ageReps, "age")
    createRepDicts(genderReps, "gender")
    createRepDicts(ethnicityReps, "ethnicity")
    #hog = processOneImage("./static/uploads/WIN_20200511_10_46_40_Pro.jpg")
    #print(len(hog))
    #print(hog)
    #ok got hog
    #now...compare hog to bestReps and then output class

    

# makes it so code isn't run when file is imported:
if __name__ == "__main__":
    rows = []
    HOGels = []
    imageData = []
    labels = []
    ethnicity = []

    # rows, imageData, ethnicity, HOGels = loadData()
    #babiesOldiesData = getBabiesOldiesHOG()
    #babiesMiddiesOldies = getBabiesMiddiesOldiesHOG()

    randomSample = getRandomSampleHOG(10000)
    inputReps = []
    inputRepClasses = []

    # Must predetermine reps in order to have baby rep and old guy rep in order to get .98, unless we get lucky:
    #inputReps = predetermineReps(babiesOldiesData, inputReps)
    #iterakeKmeans(inputData, traningDataSize, testType, clusterNum, iterations)
    #gender -> 30 clusters -> 94%
    #ethnicity -> 50 -> 80%
    #age -> 50 -> 85%
    #bestReps, bestDicts, bestIndices, bestMap = iterateKmeans(randomSample, 100, 'age', 50, 3)
    kMeansDicts()
