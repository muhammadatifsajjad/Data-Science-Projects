import random
import re
import nltk
import numpy as np
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

random.seed(10)

# store tweets in a list as strings 
allTweets = twitter_samples.strings()

# iterate over the tweets and count the number of characters across all the tweets
charCount = 0
for tweet in allTweets:
    charCount += len(tweet)

# calculate and print average length, in characters, of the tweets in the corpus    
avgLenOfCharsPerTweet = charCount / len(allTweets)
print('Average length, in characters, of the tweets in the corpus is', avgLenOfCharsPerTweet)

# regular expression to find hashtags
regex = r'(?<=^#)[a-z]{8,}$|(?<=^#)[a-z]{8,}(?=\s)|(?<=\s#)[a-z]{8,}$|(?<=\s#)[a-z]{8,}(?=\s)'

# extract all hashtags using the above regex
listOfHashtags = [hashtag for hashtag in [re.findall(regex, tweet) for tweet in allTweets] if hashtag != []]

# count and print total number of hashtags collected
hashtagsCount = 0
for hashtags in listOfHashtags:
    hashtagsCount += len(hashtags)
    
print('Total number of hashtags across all the tweets in the corpus is', hashtagsCount)
                     
# code for lemmatization
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
def lemmatize(word):
    lemma = lemmatizer.lemmatize(word,'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word,'n')
    return lemma
    
# words is a Python list
wordsCorpus = nltk.corpus.words.words()

# reverse MaxMatch algorithm
def reverseMaxMatch(word):
    remWordsList = ""
    n = len(word)

    for i in range(n):
        if lemmatize(word[i:]) in wordsCorpus:      # if the lemmatized word is found in corpus, add it to the list
            tokenizedHashtags.append(word[i:])      
            if(remWordsList != ""):                 # recursive call on words that have not yet been compared
                reverseMaxMatch(remWordsList)       
                break
            else:
                break
        else:
            remWordsList += word[i]                 # add left out alphabet to remaining words list to be compared in the recursive call

# initialize final list of tokenized hashtags            
listOfTokenizedHashtags = []

# call reverse MaxMatch algorithm hashtags to get tokenized hashtags 
for hashtags in listOfHashtags:
    tokenizedHashtags = []  
    for hashtag in hashtags:         
        reverseMaxMatch(hashtag)
    listOfTokenizedHashtags.append(tokenizedHashtags)

# print out the last 20 tokenized hashtags
print('Last 20 hashtags:\n', listOfTokenizedHashtags[-20:])

# get tokenized positive and negative tweets from twitter_samples
posTweets = nltk.corpus.twitter_samples.tokenized("positive_tweets.json")
negTweets = nltk.corpus.twitter_samples.tokenized("negative_tweets.json")

# randomly split each subcorpus in 80:10:10 ratio for training:development:testing
from sklearn.model_selection import train_test_split
posTweetsTrain, posTweetsDev = train_test_split(posTweets, test_size=0.2)
posTweetsDev, posTweetsTest = train_test_split(posTweetsDev, test_size=0.5)

negTweetsTrain, negTweetsDev = train_test_split(negTweets, test_size=0.2)
negTweetsDev, negTweetsTest = train_test_split(negTweetsDev, test_size=0.5)

# combine positive and negative tweets data
# positive tweets are labelled as 1, and negative tweets are labelled as 0
xTrain = list(posTweetsTrain)
xTrain.extend(negTweetsTrain)

yTrain = list(np.repeat(1, 4000))
yTrain.extend(list(np.repeat(0, 4000)))

xDev = list(posTweetsDev)
xDev.extend(negTweetsDev)

yDev = list(np.repeat(1, 500))
yDev.extend(list(np.repeat(0, 500)))

xTest = list(posTweetsTest)
xTest.extend(negTweetsTest)

yTest = list(np.repeat(1, 500))
yTest.extend(list(np.repeat(0, 500)))

# remove unnecessary variables from memory
del posTweets, negTweets, negTweetsTrain, negTweetsDev, negTweetsTest, posTweetsTrain, posTweetsDev, posTweetsTest

# get bag of words after doing lower-casing, removing stopwords and non-alphabetic characters
stopwords = set(stopwords.words('english'))
def getBOWLoweredNoStopwords(tweet):
    BOW = {}
    for word in tweet:
        word = word.lower()
        regex = r'^[a-z]+$'         #Remove non-alphabetic tokens
        word = str(re.findall(regex, word))
        if word not in stopwords and word != '[]' :
            BOW[word] = BOW.get(word,0) + 1
    return BOW

def prepareTweetsData(allTweets):
    feature_matrix = []
    for tweet in allTweets:
        feature_dict = getBOWLoweredNoStopwords(tweet) 
        feature_matrix.append(feature_dict)
    return feature_matrix

vectorizer = DictVectorizer()
trainDataset = vectorizer.fit_transform(prepareTweetsData(xTrain))
devDataset = vectorizer.transform(prepareTweetsData(xDev))
testDataset = vectorizer.transform(prepareTweetsData(xTest))

# create logistic regression classifier
def logisticRegressionModel(penaltyVal, cVal, predDataset):
    logReg = LogisticRegression(penalty = penaltyVal, C = cVal) # initialize logistic regression model with parameter values
    logReg.fit(trainDataset, yTrain)    # train the logistic regression model using training set
    return logReg.predict(predDataset)   # predict using development dataset

# create multinomial naive bayes classifier
def naiveBayesModel(alphaVal, predDataset):
    naiveBayes = MultinomialNB(alpha = alphaVal) # initialize logistic regression model with parameter values
    naiveBayes.fit(trainDataset, yTrain) # train the naive bayes model using training set
    return naiveBayes.predict(predDataset) # predict using development dataset

# Hyperparameter tuning for Naive Bayes  
devPredNaiveBayes = naiveBayesModel(1.5, devDataset)
print('Naive Bayes Classifier Accuracy with parameter, alpha=1.5 : ', accuracy_score(yDev, devPredNaiveBayes))
 
devPredNaiveBayes = naiveBayesModel(1.6, devDataset)
print('Naive Bayes Classifier Accuracy with parameter, alpha=1.6 : ', accuracy_score(yDev, devPredNaiveBayes))
 
devPredNaiveBayes = naiveBayesModel(1.7, devDataset)
print('Naive Bayes Classifier Accuracy with parameter, alpha=1.7 : ', accuracy_score(yDev, devPredNaiveBayes))

devPredNaiveBayes = naiveBayesModel(1.8, devDataset)
print('Naive Bayes Classifier Accuracy with parameter, alpha=1.8 : ', accuracy_score(yDev, devPredNaiveBayes))
  
devPredNaiveBayes = naiveBayesModel(1.9, devDataset)
print('Naive Bayes Classifier Accuracy with parameter, alpha=1.9 : ', accuracy_score(yDev, devPredNaiveBayes))

devPredNaiveBayes = naiveBayesModel(2, devDataset)
print('Naive Bayes Classifier Accuracy with parameter, alpha=2 : ', accuracy_score(yDev, devPredNaiveBayes))
  
devPredNaiveBayes = naiveBayesModel(2.1, devDataset)
print('Naive Bayes Classifier Accuracy with parameter, alpha=2.1 : ', accuracy_score(yDev, devPredNaiveBayes))
       
# Hyperparameter tuning for logistic regression   
devPredLogReg = logisticRegressionModel('l1', 1, devDataset)
print('\n\nLogistic Regression Classifier Accuracy with parameters, penalty=l1, C=1 : ', accuracy_score(yDev, devPredLogReg))

devPredLogReg = logisticRegressionModel('l2', 1, devDataset)
print('Logistic Regression Classifier Accuracy with parameters, penalty=l2, C=1 : ', accuracy_score(yDev, devPredLogReg))

devPredLogReg = logisticRegressionModel('l2', 0.9, devDataset)
print('Logistic Regression Classifier Accuracy with parameters, penalty=l2, C=0.9 : ', accuracy_score(yDev, devPredLogReg))

devPredLogReg = logisticRegressionModel('l2', 0.8, devDataset)
print('Logistic Regression Classifier Accuracy with parameters, penalty=l2, C=0.8 : ', accuracy_score(yDev, devPredLogReg))

devPredLogReg = logisticRegressionModel('l2', 0.7, devDataset)
print('Logistic Regression Classifier Accuracy with parameters, penalty=l2, C=0.7 : ', accuracy_score(yDev, devPredLogReg))

devPredLogReg = logisticRegressionModel('l2', 0.6, devDataset)
print('Logistic Regression Classifier Accuracy with parameters, penalty=l2, C=0.6 : ', accuracy_score(yDev, devPredLogReg))

devPredLogReg = logisticRegressionModel('l2', 0.5, devDataset)
print('Logistic Regression Classifier Accuracy with parameters, penalty=l2, C=0.5 : ', accuracy_score(yDev, devPredLogReg))
 
# predict using test dataset
testPredLogReg = logisticRegressionModel('l2', 0.6, testDataset)
print('Logistic Regression Classifier Accuracy with parameters, penalty=l2, C=0.6 : Accuracy - ', accuracy_score(yTest, testPredLogReg), ' f1-score - ', f1_score(yTest, testPredLogReg))

testPredNaiveBayes = naiveBayesModel(1.6, testDataset)
print('Naive Bayes Classifier Accuracy and f1-score with parameter, alpha=1.6 : Accuracy - ', accuracy_score(yTest, testPredNaiveBayes), ' f1-score - ', f1_score(yTest, testPredNaiveBayes))