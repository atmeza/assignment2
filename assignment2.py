import pandas as pd
import operator
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

        words = str(lyrics).split()

        for word in words:
            wordCounts[word] += 1

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
    print("Calculating genre frequency of top 500 words")
    genreCount = defaultdict(int)
    for d in data:
        genre = d[4]
        if genre == 'Pop':
            lyrics = str(d[5])
            predicate = lambda x: x not in string.punctuation
            lyrics = filter(predicate, lyrics.lower())
            words = str(lyrics).split()

            for word in popularWords:
                count = words.count(word[0])
                genreCount[word[0]] += count

    # Sort percentages from highest to lowest
    sort = sorted(genreCount.items(), key=lambda kv: kv[1], reverse=True)
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
    sort = sorted(freqGenre.items(), key=lambda kv:kv[1], reverse=True)
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

    print(popularFreq[:10])

    print(genreFreq[:10])

    print(diffFreq[:10])
def baseline_score(X, y, genre_word_freq):

	genre_freq = defaultdict(int)
	for genre, word_freq in genre_word_freq:
		genre_freq[genre]+=1

	most_popular_genre = max(genre_freq.iteritems(), key=operator.itemgetter(1))[0]
	punctuation = string.punctuation

	for x,y in zip(X,y):
		lyrics = x[4]
		lyrics =''.join([c for c in lyrics.lower() if not c in punctuation])
		for word in lyrics.split():
			if word is 
	return 0


def main():
    print("Reading data...")

    #read in csv and shuffle dataframe and only use 100k rows
    data = pd.read_csv("./lyrics.csv")
    data = data.dropna(how='any')

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

    #split sets into features and labels
    train_features = np.column_stack((train_set[:,:4], train_set[:,5]))
    train_labels = train_set[:,4]
    val_features = np.column_stack((validation_set[:,:4], validation_set[:,5]))
    val_labels = validation_set[:,4]
    test_features = np.column_stack((test_set[:,:4], test_set[:,5]))
    test_labels = test_set[:,4]

    print(train_features[0])
    print(train_labels[0])

    # get all words and their frequencies
    #genre_word_freq = countWords(train_set, validation_set, test_set)

    baseline_score(test_features, test_labels, genre_word_freq)







if __name__ == "__main__":
    main()
