import pandas as pd
import numpy as np
from collections import defaultdict
import string


def feature(datum):
    feat = []
    feat.append(1)

    return feat


def countWords(train, validation, test):
    wordCounts = defaultdict(int)

    data = np.concatenate([train, validation, test], axis=0)

    # Calculates frequency of top 500 words in train, validation, and test datasets
    print("Calculating frequency of top 500 words...")
    for d in data:
        lyrics = str(d[5])
        predicate = lambda x:x not in string.punctuation
        lyrics = filter(predicate, lyrics.lower())
        words = lyrics.split()

        for word in words:
            wordCounts[word] += 1

    # Sort word frequency from highest to lowest
    sort = sorted(wordCounts.items(), key=lambda(k,v): v, reverse=True)
    sort = sort[:500]
    popularWords = [[d[0], d[1]] for d in sort]

    # Calculate percentage of word frequency among all 500 words
    print("Calculating percentage of word frequency...")
    freqWords = defaultdict(int)
    i = 0
    for word in popularWords:
        count = 0
        j = 0
        for w in popularWords:
            count += popularWords[j][1]
            j += 1

        freqWords[word[0]] = popularWords[i][1] * 1.0 / count
        i += 1

    # Sort percentages from highest to lowest
    sort = sorted(freqWords.items(), key=lambda(k,v): v, reverse=True)
    popularFreq = [[d[0], d[1]] for d in sort]

    # Calculate frequency of words per genre
    print("Calculating genre frequency of top 500 words")
    genreCount = defaultdict(int)
    for d in data:
        genre = d[4]
        if genre == 'Pop':
            lyrics = str(d[5])
            predicate = lambda x: x not in string.punctuation
            lyrics = filter(predicate, lyrics.lower())
            words = lyrics.split()

            for word in popularWords:
                count = words.count(word[0])
                genreCount[word[0]] += count

    # Sort percentages from highest to lowest
    sort = sorted(genreCount.items(), key=lambda(k,v): v, reverse=True)
    popularGenre = [[d[0], d[1]] for d in sort]

    # Calculate percentage of word frequency per genre among all 500 words
    print("Calculating percentage of genre word frequency...")
    freqGenre = defaultdict(int)
    i = 0
    for word in popularGenre:
        count = 0
        j = 0
        for w in popularGenre:
            count += popularGenre[j][1]
            j += 1

        freqGenre[word[0]] = popularGenre[i][1] * 1.0 / count
        i += 1

    # Sort percentages from highest to lowest
    sort = sorted(freqGenre.items(), key=lambda (k, v): v, reverse=True)
    genreFreq = [[d[0], d[1]] for d in sort]

    # Calculate difference between genre word percentage and word percentage
    print("Calculating difference...")
    # diffFreq = defaultdict(int)
    # for word in popularFreq:
    #     diff = genreFreq[word][1] - popularFreq[word][1]
    #     diffFreq[word] = diff

    # Sort percentages from highest to lowest
    # sort = sorted(diffFreq.items(), key=lambda(k,v): v, reverse=True)
    # diffFreq = [[d[0], d[1]] for d in sort]

    print
    print(popularFreq[:10])
    print
    print(genreFreq[:10])
    print
    print(diffFreq[:10])


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

    countWords(train_set, validation_set, test_set)

    # print(train_set.shape)
    # print(validation_set.shape)
    # print(test_set.shape)

if __name__ == "__main__":
    main()
