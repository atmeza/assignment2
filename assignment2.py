import csv

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



def feature(datum):
    feat = []
    feat.append(1)

    return feat



def main():
    print("Reading data...")
    musicData = parseData('../lyrics.csv')
    print len(musicData)


if __name__ == "__main__": main()
