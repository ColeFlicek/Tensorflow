import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle
import random
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000

def Create_Lexicon(pos, neg):

    lexicon = []
    with open(pos, 'r') as P:
        contents = P.readlines()

        for l in contents[:hm_lines]:
            all_words = word_tokenize(l)
            lexicon += list(all_words)

    with open(neg, 'r') as N:
        contents = N.readlines()

        for l in contents[:hm_lines]:
            all_words = word_tokenize(l)
            lexicon += list(all_words)

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]

    w_counts = Counter(lexicon)
    l2 = []
    for w in w_counts:
        if 1000 > w_counts[w] > 50:
            l2.append(w)

    return l2

def Sample_Handleing(sample, lexicon, classificaiton):
    featureset = []

    with open(sample, 'r') as f:
        contents = f.readlines()

        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1

            features = list(features)
            featureset.append([features, classificaiton])

    return featureset

def Create_Featureset_and_Labels(pos, neg, test_size = 0.1):
    lexicon = Create_Lexicon(pos, neg)
    features = []
    features += Sample_Handleing('pos.txt', lexicon, [1, 0])
    features += Sample_Handleing('neg.txt', lexicon, [0, 1])
    random.shuffle(features)

    features = np.array(features)

    testing_size = int(test_size*len(features))

    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])
    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])

    return train_x,train_y,test_x,test_y


train_x, train_y, test_x, test_y = Create_Featureset_and_Labels('pos.txt', 'neg.txt', test_size = 0.1)
if __name__ == '__main__':

    with open('Sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x,train_y,test_x,test_y], f)










