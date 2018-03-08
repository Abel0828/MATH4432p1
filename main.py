import numpy as np
import csv
import matplotlib.pyplot as plt

TEST_SET='zip.test'
TRAIN_SET='zip.train'

def plotData(datafile):
    with open(datafile, 'r', newline='') as csv_file:
        for row in csv.reader(csv_file,delimiter=' '):
            # The first column is the label
            label = int(float(row[0]))

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

            #break # This stops the loop, I just want to see one

def readData(datafile,flatten):
    data=[]
    labels=[]
    with open(datafile, 'r') as csv_file:
        for row in csv.reader(csv_file, delimiter=' '):
            # The first column is the label
            label = int(float(row[0]))

            # The rest of columns are pixels
            pixels = row[1:]
            # This array will be of 1D with length 256
            # The pixel intensity values are integers from 0 to 255
            pixels = [round((float(pixel) + 1) / 2 * 255) for pixel in pixels if pixel and (not pixel.isspace())]
            pixels = np.array(pixels, dtype=int)
            # Reshape the array into 28 x 28 array (2-dimensional array)
            # could remove this line if want the pixels to be flattened
            if flatten:
                pixels = pixels.reshape((16, 16))
            data.append(pixels)
            labels.append(label)
            break
    return data,labels

def applyLDA():
    #ToDo


def applyQDA():
    # ToDo



if __name__=="__main__":
    #plotData(TRAIN_SET)
    training_data, training_labels=readData(TRAIN_SET, flatten=True)
    # print(training_data[0][:])
    # print(training_labels[0])
    applyLDA()
    applyQDA()

    testing_data, testing_labels = readData(TEST_SET)


