import csv
import numpy as np
import random
import random
from PIL import Image
from dataLoader import loadData, createImageDictionaries, getBabiesOldies
import matplotlib.pyplot as plt

# File containing all the helper methods unrelated to loading data

# Helper function to normalize a list of pixels:
def normalizeImage(pixels):
    for j in range(len(pixels)):
        pixels[j] = pixels[j] / 255.0
    return pixels

# Helper function to denormalize a list of pixels:
def denormalizeImage(pixels):
    for j in range(len(pixels)):
        pixels[j] = pixels[j] * 255.0
    return pixels

# Helper function that displays an image to the screen given a 1d list of pixels:
# takes second parameter for whether or not to denormalize the image
def displayImage(pixels, denormalize = False):
    if denormalize:
        pixelCopy = pixels.copy()
        pixelCopy= denormalizeImage(pixelCopy)
        pixels = pixelCopy
    pixArray = np.array(pixels)
    pixArray2D = np.reshape(pixArray, (-1, 48))
    img = Image.fromarray(pixArray2D)
    img.show()


# Helper function that computes the mean square distance of an image and a representative:
def meanSquareDistance(rep, dataPoint):
    sum = 0
    # Standard deviation formula:
    for i in range(len(rep)):
        squareDistance = (rep[i] - dataPoint[i])**2
        sum += squareDistance
    MSD = sum/ len(rep)
    return MSD


# Helper function that averages all the pixels in all the images of a given class
# and then returns that average image:
def findClassAvg(rep, images):
    # displayImage(rep['pix'])

    # Create empty list of pixels:
    avgPix = [0] * len(images[0]['pix'])
    classImageCount = 0
    count = 0
    for image in images:

        # if the image belongs to the class we are looking for:
        if image['class'] == rep['class']:
            # Skip images that are missing pixels:
            if image['pix'] == None:
                count += 1
                continue

            # tally up the number of images that we are averaging:
            classImageCount += 1

            # iterate over all pixels in the image and add it to the respective pixel of avgPix:
            for i in range(len(image['pix'])):
                avgPix[i] += float(image['pix'][i])

    # Divide each sum of pixels by the number of images to create an average:
    for i in range(len(avgPix)):
        if classImageCount == 0:
            print('something is fucked!')
        avgPix[i] /= classImageCount
    return avgPix

def findClassAvgHOG(rep, images):
    # displayImage(rep['HOG'])

    # Create empty list of HOGels:
    avgHOG = [0] * len(images[0]['HOG'])
    classImageCount = 0
    count = 0
    for image in images:

        # if the image belongs to the class we are looking for:
        if image['class'] == rep['class']:
            # Skip images that are missing HOGels:
            if image['HOG'] == None:
                count += 1
                continue

            # tally up the number of images that we are averaging:
            classImageCount += 1

            # iterate over all HOGels in the image and add it to the respective HOGel of avgHOG:
            for i in range(len(image['HOG'])):
                avgHOG[i] += float(image['HOG'][i])

    # Divide each sum of HOGels by the number of images to create an average:
    for i in range(len(avgHOG)):
        if classImageCount == 0:
            pass
        else:
            avgHOG[i] /= classImageCount
    return avgHOG
# Experimental function to average all the faces for a given ethnicity (takes forever probably doesn't work)
def findAvgFace(images, ethnicities, selection):
    # Filter all images that have the selected ethnicity:
    filteredImages = []
    for j in range(len(images)):
        if int(ethnicities[j]) != selection:
            continue
        images[j] = images[j].split()
        for i in range(len(images[j])):
            images[j][i] = int(images[j][i])
            filteredImages.append(images[j])

    # Average the images:
    avgPix = [0] * len(filteredImages[0])
    for image in filteredImages:
        for i in range(len(image)):
            avgPix[i] += image[i]
    for i in range(len(avgPix)):
        avgPix[i] //= len(filteredImages)

    displayImage(avgPix)

# Saves an image to the images directory:
def saveImage(imageDict):
    pixArray = np.array(imageDict['pix'])
    pixArray2D = np.reshape(pixArray, (-1, 48))
    img = Image.fromarray(pixArray2D)
    img = img.convert('RGB')
    img.save("images/" + imageDict['imgname'])

# Creates a jpg file for every image in the dataset and saves it to an images folder:
def createImageDirectory():
    for dict in createImageDictionaries():
        saveImage(dict)

# my best guesses at what the ethnicities are. I'm not very sure about 3 and 4.
ethnicityMap = {0: 'white', 1: "black", '2': "asian", '3': 'indian', '4': 'hispanic'}
genderMap = {0: 'male', 1: 'female'}
classMaps = {'ethnicity': ethnicityMap, 'gender': genderMap}
# Checks which rep an image is closer to:
def testOneFace(reps, testImageVector, testType):
    minDist = meanSquareDistance(reps[0]["HOG"], testImageVector)
    minRep = reps[0]
    for rep in reps[1:]:
        dist = meanSquareDistance(rep["HOG"], testImageVector)
        if dist < minDist:
            minDist = dist
            minRep = rep
    print("The predicted", testType, "of the person in this image is", classMaps[testType][minRep['class']])
    return minRep

def getDistributedData(images, clusterNum, testType, size = 500):
    newImages = []
    clusterCounts = [0] * clusterNum

    for image in images:
        #loop thru images and if their testType value = i, then add to array
        #size of array limited to insure somewhat equal data distribution
        for i in range(clusterNum):
            if image[testType] == i and clusterCounts[i] < size:
                newImages.append(image)
                clusterCounts[i] +=1
                # no need to check for other i values once we found a match, so break:
                break
    return newImages


sums_over_iterations = []
genderGraphCount = 0
# returns a list of how many images are in each cluster:
def sumClusters(images, clusterNum):
    sums = [0] * clusterNum
    for image in images:
        sums[image['class']] += 1
    sums_over_iterations.append(sums)
    return sums

def createGraphOfClusterSums(testType):
    global genderGraphCount
    global sums_over_iterations
    print(sums_over_iterations)
    clusters = []
    for i in range(len(sums_over_iterations[0])):
        clusters.append([])
    legendHandles = []
    legendLabels = []
    for sums in sums_over_iterations:
        for i in range(len(sums)):
            cluster = clusters[i]
            # print(cluster)
            cluster.append(sums[i])
    for i in range(len(clusters)):
        legendHandle, = plt.plot(clusters[i], label = "cluster " + str(i))
        legendHandles.append(legendHandle)
        legendLabels.append("cluster " + str(i))
    plt.legend(legendHandles, legendLabels)
    plt.ylabel('# images in cluster')
    plt.xlabel("Kmeans iteration")
    plt.title("# images in each " + testType + " cluster over iterations")
    plt.savefig(testType+ "_clusters" + str(genderGraphCount) + ".png")
    genderGraphCount+= 1
    plt.show()
    sums_over_iterations = []


