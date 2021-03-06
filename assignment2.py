import pandas as pd
import operator
import numpy as np
from collections import defaultdict
import string
from sklearn.svm import SVR

def dataStats(data):

    # Calculate genre counts
    genreCounts = defaultdict(int)
    genres = []
    for d in data:
        genreCounts[d[4]] += 1

    sort = sorted(genreCounts.items(), key=lambda kv: kv[1], reverse=True)
    print()
    print('Genre Counts')
    for gc in sort:
        print(str(gc[0]) + str(': ') + str(gc[1]))
        genres.append(str(gc[0]).lower())

    # Calculate average lyric length
    # and the number of times a genre name is in the song lyrics
    count = 0
    total = 0
    genreInLyrics = 0
    for d in data:
        lyrics = str(d[5])
        translator = lyrics.maketrans('', '', string.punctuation)
        lyrics = lyrics.translate(translator)
        words = str(lyrics.lower()).strip().split()
        count += len(words)

        for word in words:
            if word in genres:
                genreInLyrics += 1

        if len(words) > 0:
            total += 1

    print()
    print('Average lyric length')
    print(count * 1.0 / total)
    print()
    print('Number of times a genre name is in the song lyrics')
    print(genreInLyrics)
    print()

    # Calculate number of unique artists, average number of songs per artist, and most popular artist
    artistCount = defaultdict(int)
    count = 0
    for d in data:
        artist = d[3]
        artistCount[artist] += 1
        count += 1

    sort = sorted(artistCount.items(), key=lambda kv: kv[1], reverse=True)

    print("Number of unique artists")
    print(len(artistCount))
    print()
    print("Average number of songs per artist")
    print(count * 1.0 / len(artistCount))
    print()
    print("Most popular artist")
    print(str(sort[0][0]) + ', ' + str(sort[0][1]))
    print()


def countWords(data):
    wordCounts = defaultdict(int)
    genres = defaultdict(int)

    # Calculates frequency of top 500 words in train, validation, and test datasets
    print("Calculating frequency of top 500 words...")
    for d in data:
        lyrics = str(d[5])
        translator = lyrics.maketrans('', '', string.punctuation)
        lyrics = lyrics.translate(translator)
        words = str(lyrics.lower()).strip().split()

        for word in words:
            wordCounts[word] += 1

        genre = d[4]
        genres[genre] += 1

    # Sort word frequency from highest to lowest
    sort = sorted(wordCounts.items(), key=lambda kv: kv[1], reverse=True)
    sort = sort[:230]
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
        words = str(lyrics).lower().strip().split()

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

    # Uncomment to see what the data looks like
    # for gc in genres:
    #     print('Genre:', gc)
    #     for d in diffFreq[gc]:
    #         print(d)
    #     print()

    return diffFreq, genres


def predictGenre(testSet, data, genres):
    genreCount = defaultdict(int)
    predictions = []

    i = 1
    print("Starting predictions...")

    for test in testSet:
        for g in genres:
            count = 0
            genre = data[g]
            lyrics = str(test[5])
            translator = lyrics.maketrans('', '', string.punctuation)
            lyrics = lyrics.translate(translator)
            words = str(lyrics).lower().strip().split()

            for w in words:
                for val in genre:
                    if w == val[0]:
                        if w in genres:
                            count += val[1] * 4
                        else:
                            count += val[1]
                        break

            genreCount[g] = count

        sort = sorted(genreCount.items(), key=lambda kv: kv[1], reverse=True)
        predictions.append([sort[0][0], test[4]])
        
        i += 1

    correct = 0
    for p in predictions:
        if p[0] == p[1]:
            correct += 1

    accuracy = correct * 1.0 / len(predictions) * 100
    print('Accuracy:', accuracy)
    return accuracy


def baseline_score(ytrain, X, y):

    genre_freq = defaultdict(int)
    genres = []
    for genre in ytrain:
        genre_freq[genre]+=1
        if genre not in genres:
            genres.append(genre)
    most_popular_genre = max(genre_freq.items(), key=operator.itemgetter(1))[0]
    punctuation = string.punctuation
    accurate = 0
    total = 0
    for x,y in zip(X,y):
        total+=1
		
        lyrics = x[4]
        lyrics =''.join([c for c in lyrics.lower() if not c in punctuation])
        predict = str()
		
        for w in genres:
            if(w.lower() in lyrics.split()):
                predict = w
                break
        
        if not predict:
            predict = most_popular_genre
		
        if(predict == y):
            accurate+=1

    return accurate / total * 100.0
def create_feature_matrix(X, popularWords):
	X = []
	x = []
	song_lyrics = X[:,4]
	for lyrics in song_lyrics:
		lyrics =''.join([c for c in lyrics.lower() if not c in punctuation]) 
		for word in popularWords.items()[:500]:
			if(word in lyrics):
				x.append(popularWords[word])
			else:
				x.append(0)
		X.append(x)


	return X
def create_labels(Y, genre):
	label = []
	for y in Y:
		if(y == genre):
			label.append(1)
		else:
			label.append(0)
	return label

def main():
    print("Reading data...")

    # read in csv and shuffle dataframe and only use 100k rows
    data = pd.read_csv("./lyrics.csv")
    data = data.dropna(how='any')
    shuffled_data = data.sample(frac=1)
    shuffled_data = shuffled_data[:10000]

    for ind, row in shuffled_data.iterrows():
        if row['genre'] == 'Not Available':
            shuffled_data.drop(ind, inplace=True)

    # split dataframe into train, val, and test sets
    train_set = shuffled_data[:int(len(shuffled_data)*.5)]
    validation_set = shuffled_data[int(len(shuffled_data)*.5):int(len(shuffled_data)*.8)]
    test_set = shuffled_data[int(len(shuffled_data)*.8):]

    # from dataframe to np array
    train_set = train_set.as_matrix()
    validation_set = validation_set.as_matrix()
    test_set = test_set.as_matrix()


    # split sets into features and labels
    train_features = np.column_stack((train_set[:,:4], train_set[:,5]))
    train_labels = train_set[:,4]
    val_features = np.column_stack((validation_set[:,:4], validation_set[:,5]))
    val_labels = validation_set[:,4]
    test_features = np.column_stack((test_set[:,:4], test_set[:,5]))
    test_labels = test_set[:,4]

    # Pre-processing data
    data = np.concatenate([train_set, validation_set, test_set], axis=0)
    # dataStats(data)
	
    # calculate most popular words and their frequency by genre
    genrePopularWords, genres = countWords(train_set)
    predictGenre(test_set, genrePopularWords, genres)
	
    # calculate accuracy of baseline
    b_score = baseline_score(train_labels, test_features, test_labels)

    print(b_score)

    clf0 = 
	

if __name__ == "__main__":
    main()