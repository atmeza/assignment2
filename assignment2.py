import urllib

def parseData(fname):
    for l in urllib.urlopen(fname):
        yield eval(l)

def feature(datum):
    feat = []
    feat.append(1)

    return feat



def main():
    print("Reading data...")
    musicData = list(parseData('file.json'))


if __name__ == "__main__": main()