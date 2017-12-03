import csv
import random
import pandas as pd
import numpy as np

def parseData(fname):
    data = []
    with open(fname, "rb") as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            data.append(line)

    return data

def splitData(data):
    train = []
    test = []

    trainFile = open('trainFile.txt', 'w')
    testFile = open('testFile.txt', 'w')


    while len(data) > 0:
        rand = random.randrange(0, len(data) - 1)
        train.append(data[rand])
        trainFile.write(str(data[rand]) + '\n')
        data.remove(data[rand])

        rand = random.randrange(0, len(data) - 1)
        test.append(data[rand])
        testFile.write(str(data[rand]) + '\n')
        data.remove(data[rand])

        
	
    trainFile.close()
    testFile.close()
    return [train, test]


def feature(datum):
    feat = []
    feat.append(1)

    return feat



def main():
    print("Reading data...")

    #read in csv and shuffle dataframe and only use 100k rows
    data = pd.read_csv("./lyrics.csv")
    shuffled_data = data.sample(frac=1)
    shuffled_data = shuffled_data[:100000]
    #split dataframe into train, val, and test sets
    train_set = shuffled_data[:int(len(shuffled_data)*.5)]
    validation_set = shuffled_data[int(len(shuffled_data)*.5):int(len(shuffled_data)*.8)]
    test_set = shuffled_data[int(len(shuffled_data)*.8):]

    #from dataframe to np array
    train_set = train_set.as_matrix()
    validation_set = validation_set.as_matrix()
    test_set = test_set.as_matrix()


    print(train_set.shape)
    print(validation_set.shape)
    print(test_set.shape)

if __name__ == "__main__": main()
