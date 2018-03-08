import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
TEST_SET='zip.test'
TRAIN_SET='zip.train'

def plotData(datafile):
    with open(datafile, 'r', newline='') as csv_file:
        for row in csv.reader(csv_file,delimiter=' '):
            # The first column is the label
            label = row[0]

            # The rest of columns are pixels
            # Note: the ending 'pixel' is actually a ''
            pixels = row[1: ]

            # This array will be of 1D with length 256
            # The pixel intensity values are integers from 0 to 255

            pixels=[round((float(pixel)+1)/2*255) for pixel in pixels if pixel and (not pixel.isspace())]
            pixels = np.array(pixels, dtype=int)
            # Reshape the array into 28 x 28 array (2-dimensional array)
            pixels = pixels.reshape((16, 16))

            # Plot
            plt.title('Label is {label}'.format(label=label))
            plt.imshow(pixels, cmap='gray')
            plt.show()

            #break # This stops the loop -- I just want to see one

def readData(datafile,flatten=True, discrete=False):
    # Param: datafile -- the filename of the data to read
    # Param: flatten-- decidee whether to return the value as a one dimentional vector
    # Param: discrete -- decide whether to return a grayscale integer value from 0-255

    data=[]
    labels=[]
    with open(datafile, 'r') as csv_file:
        for row in csv.reader(csv_file, delimiter=' '):
            # The first column is the label
            label = row[0]

            # The rest of columns are pixels
            pixels = [float(pixel) for pixel in row[1:] if pixel and (not pixel.isspace())]
            # This array will be of 1D with length 256
            # The pixel intensity values are integers from 0 to 255
            if discrete:
                pixels = [round((float(pixel) + 1) / 2 * 255) for pixel in pixels]
            pixels = np.array(pixels)
            # Reshape the array into 28 x 28 array (2-dimensional array)
            # could remove this line if want the pixels to be flattened
            if not flatten:
                pixels = pixels.reshape((16, 16))
            data.append(pixels)
            labels.append(label)

    return data,labels

def applyLDA(training_data, training_labels,testing_data, testing_labels):
    lda_1 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(training_data, training_labels)
    score=lda_1.score(testing_data,testing_labels)
    print("LDA accuracy: ",score)
    return


def applyQDA(training_data, training_labels, testing_data, testing_labels):
    lda_1 = QuadraticDiscriminantAnalysis(store_covariance=True).fit(training_data, training_labels)
    score=lda_1.score(testing_data,testing_labels)
    print("QDA accuracy: ",score)
    return



if __name__=="__main__":
    #plotData(TRAIN_SET)
    training_data, training_labels=readData(TRAIN_SET, flatten=True, discrete=False)
    print(training_data[0][:])
    print(training_labels)

    testing_data, testing_labels = readData(TEST_SET, flatten=True, discrete=False)

    applyLDA(training_data, training_labels,testing_data, testing_labels)
    applyQDA(training_data, training_labels, testing_data, testing_labels)



