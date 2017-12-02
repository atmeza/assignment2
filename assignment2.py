import csv
import random

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

        print len(data)

    trainFile.close()
    testFile.close()
    return [train, test]


def feature(datum):
    feat = []
    feat.append(1)

    return feat



def main():
    print("Reading data...")

    trainX = [d for d in open('../trainFile.txt', 'r')]
    testX = [d for d in open('../testFile.txt', 'r')]


if __name__ == "__main__": main()
