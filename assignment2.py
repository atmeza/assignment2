import pandas as pd
import numpy as np
from collections import defaultdict
import string


def feature(datum):
    feat = []
    feat.append(1)

    return feat


def countWords(data):
    wordCounts = defaultdict(int)
    genres = defaultdict(int)

    # Calculates frequency of top 500 words in train, validation, and test datasets
    print("Calculating frequency of top 500 words...")
    for d in data:
        lyrics = str(d[5])
        translator = lyrics.maketrans('', '', string.punctuation)
        lyrics = lyrics.translate(translator)
        words = str(lyrics.lower()).split()

        for word in words:
            wordCounts[word] += 1

        genre = d[4]
        genres[genre] += 1

    # Sort word frequency from highest to lowest
    sort = sorted(wordCounts.items(), key=lambda kv: kv[1], reverse=True)
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
    sort = sorted(freqWords.items(), key=lambda kv: kv[1], reverse=True)
    popularFreq = [[d[0], d[1]] for d in sort]

    # Calculate frequency of words per genre
    print("Calculating genre frequency of top 500 words...")
    genreCount = defaultdict(defaultdict(int).copy)
    for d in data:
        genre = d[4]
        lyrics = str(d[5])
        translator = lyrics.maketrans('', '', string.punctuation)
        lyrics = lyrics.translate(translator)
        words = str(lyrics).split()

        for word in popularWords:
            count = words.count(word[0])
            genreCount[genre][word[0]] += count

    # Sort percentages from highest to lowest
    for gc in genres:
        sort = sorted(genreCount[gc].items(), key=lambda kv: kv[1], reverse=True)
        genreCount[gc] = sort

    # Calculate percentage of word frequency per genre among all 500 words
    print("Calculating percentage of genre word frequency...")
    freqGenre = defaultdict(defaultdict(int).copy)
    for gc in genres:
        for word in genreCount[gc]:
            count = 0
            for w in genreCount[gc]:
                count += w[1]

            freqGenre[gc][word[0]] = word[1] * 1.0 / count

    # Sort percentages from highest to lowest
    for gc in genres:
        sort = sorted(freqGenre[gc].items(), key=lambda kv: kv[1], reverse=True)
        freqGenre[gc] = sort

    # Calculate difference between genre word percentage and word percentage
    print("Calculating difference...")
    diffFreq = defaultdict(defaultdict(int).copy)
    for gc in genres:
        i = 0
        for word in popularFreq:
            j = 0
            for g in freqGenre[gc]:
                if word[0] == g[0]:
                    diff = g[1] - popularFreq[i][1]
                    diffFreq[gc][word[0]] = diff
                j += 1
            i += 1

    # Sort percentages from highest to lowest
    for gc in genres:
        sort = sorted(diffFreq[gc].items(), key=lambda kv: kv[1], reverse=True)
        diffFreq[gc] = sort

    # for gc in genres:
    #     print('Genre:', gc)
    #     for d in diffFreq[gc]:
    #         print(d)
    #     print()

    return diffFreq


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

    genrePopularWords = countWords(train_set)

    # print(train_set.shape)
    # print(validation_set.shape)
    # print(test_set.shape)

if __name__ == "__main__":
    main()
