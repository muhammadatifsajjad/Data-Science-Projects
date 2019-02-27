import string
import pandas as pd
import operator
from operator import itemgetter
import os
import nltk 
import json
import nltk
import pickle 
import numpy as np
import json
from gensim.summarization.bm25 import BM25
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from collections import Counter, defaultdict
from math import log
import spacy
import en_core_web_md
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
import os

###################### LOAD DATA FILES ###############################  

with open('documents.json') as json_data:
    docs = json.load(json_data)
    
with open('training.json') as json_data:
    train = json.load(json_data)

with open('testing.json') as json_data:
    test = json.load(json_data)
    
with open('devel.json') as json_data:
    dev = json.load(json_data)

# increase train examples by add questions from dev
for devQuest in dev:
    train.append(devQuest)
    
# exploratory data analysis: check answer mostly comprises of how many terms
answerLength = {}
for q in train:
    answerLength[len(word_tokenize(q['text']))] = answerLength.get(len(word_tokenize(q['text'])), 0) + 1 
        
###################### PART 1 - PRE-PROCESSING ###############################  

# for stop-words, punctuation removal and stemming    
stopwords = set(nltk.corpus.stopwords.words('english'))
stemmer = nltk.stem.PorterStemmer() 
translation = str.maketrans("","", string.punctuation)

# perform following pre-processing:
#    (1) Remove punctuation, 
#    (2) Remove stopwords 
#    (3) Lower-casing
#    (4) Stemming

# store corpus in following formats:
#   (1) corpusTokenized - List of word tokens of sentences for each document - to use by BM25 to compute relevance
#   (2) corpusSentencesIndexedByDoc - List of sentences for each document - to get the predicted sentence after performing BM25 similarity
#   (3) corpusSentencesIndexedByDocPara - List of sentences for each paragraph of each document - to get the actual sentence after performing named entity recognition
corpusSentencesIndexedByDoc = [] 
corpusSentencesIndexedByDocPara = []   
corpusTokenized = []
for docid in docs:
    sentences = []
    para = []
    for paragraphs in docid['text']:
        parasentences = sent_tokenize(paragraphs)
        sentences += parasentences
        para.append(parasentences)
    corpusSentencesIndexedByDocPara.append(para)
    
    corpusSentencesIndexedByDoc.append(sentences)
    
    tokenizedSentence = []    
    for sent in sentences:        
        remPunctSent = sent.translate(translation)         
        wordTokens = word_tokenize(remPunctSent)
        
        listOfWordTokens = []
        for token in wordTokens:
            if token not in stopwords:
                listOfWordTokens.append(stemmer.stem(token.lower()))
        
        tokenizedSentence.append(listOfWordTokens)
    
    corpusTokenized.append(tokenizedSentence)
	
###################### PART 2 - NAMED ENTITY TAGGING ############################### 
    
# named entity tagging for every sentence in documents

#   (1) listOfTaggedSentencesIndexedByDocs - List of named entity tagged word tokens for sentences of each document in the same format as 'corpusTokenized' - to use in BM25 sentence relevance to get tags in corresponding sentence given sentence index
#   (2) listOfTaggedSentencesIndexedByDocsPara - List named entity tagged word tokens for sentences for each paragraph of each document - to compare actual answer with each tagged word token text to get the corresponding actual tag

# 13,672 untagged without pre-processing
# 13,913 untagged with stop-words removal
# 27,642 untagged with lower-casing
# 28,521 untagged with stemming
# 15,721 untagged with punctuation removal
    
nlp = en_core_web_md.load()
listOfTaggedSentencesIndexedByDocsPara = []
listOfTaggedSentencesIndexedByDocs = []
for docid in docs:
    listOfTaggedParas = []
    listOfTaggedSentences = []
    for paragraphs in docid['text']:
        sentences = []
        sentences += sent_tokenize(paragraphs)
        taggedSentence = []
        for sentence in sentences:                  
            listOftags=[]
            neSent = nlp(sentence)
            for entity in neSent.ents:
                listOftags.append([entity.text, entity.label_])
            taggedSentence.append(listOftags)  
            listOfTaggedSentences.append(listOftags)
        listOfTaggedParas.append(taggedSentence)
    listOfTaggedSentencesIndexedByDocs.append(listOfTaggedSentences)
    listOfTaggedSentencesIndexedByDocsPara.append(listOfTaggedParas)

# in below code, modifying tags in listOfTaggedSentencesIndexedByDocs will also modify tags in listOfTaggedSentencesIndexedByDocsPara
# combine 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL' tags into a single 'NUMERIC' tag
numericTagsList = ['PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL']
for doc in listOfTaggedSentencesIndexedByDocs:
    for sent in doc:
        for tag in sent:
            if tag[1] in numericTagsList:
                tag[1] = 'NUMERIC'

# combine 'DATE', 'TIME' tags into a single 'DATETIME' tag                
dateTimeTagsList = ['DATE', 'TIME']
for doc in listOfTaggedSentencesIndexedByDocs:
    for sent in doc:
        for tag in sent:
            if tag[1] in dateTimeTagsList:
                tag[1] = 'DATETIME'                

# combine 'WORK_OF_ART', 'LAW', 'PRODUCT' tags into a single 'DOCUMENT' tag                
documentTagsList = ['WORK_OF_ART', 'LAW', 'PRODUCT', 'EVENT', 'LANGUAGE']
for doc in listOfTaggedSentencesIndexedByDocs:
    for sent in doc:
        for tag in sent:
            if tag[1] in documentTagsList:
                tag[1] = 'DOCUMENT' 

# combine 'ORG', 'GPE', 'NORP', 'LOC', 'FAC' tags into a single 'INSTITUTION' tag  
institutionTagsList = ['ORG', 'GPE', 'NORP', 'LOC', 'FAC']
for doc in listOfTaggedSentencesIndexedByDocs:
    for sent in doc:
        for tag in sent:
            if tag[1] in institutionTagsList:
                tag[1] = 'INSTITUTION' 
                
# store frequency of each tags for manual checking
tagFreqDict = {}
for doc in listOfTaggedSentencesIndexedByDocs:
    for sent in doc:
        for tag in sent:
            tagFreqDict[tag[1]] = tagFreqDict.get(tag[1], 0) + 1 
                      
# store named entity tags and actual sentences containing answers for every question by exact matching the answer text with tagged terms (in sentences of the specified paragraph of the specified document)
for question in train:
    found = False
    question['actual answer tag'] = 'NA'
    question['actual sentence'] = 'NA'
    i = 0
    for sent in listOfTaggedSentencesIndexedByDocsPara[question['docid']][question['answer_paragraph']]:
        for tag in sent:
            if question['text'].lower() == tag[0].lower():
                question['actual answer tag'] = tag[1]
                question['actual sentence'] = corpusSentencesIndexedByDocPara[question['docid']][question['answer_paragraph']][i]
                found = True
                break
        if found == True:
            break
        i += 1

# for remaining untagged answers, split the answer terms and tagged terms (in sentences of the specified paragraph of the specified document) and match each term in both with each other
for question in train:
    found = False
    if question['actual answer tag'] == 'NA':
        i = 0
        for sent in listOfTaggedSentencesIndexedByDocsPara[question['docid']][question['answer_paragraph']]:
            for tag in sent:
                for term in question['text'].split():  
                    for tagWords in tag[0].lower().split():
                        if term.lower() == tagWords:
                            question['actual answer tag'] = tag[1]
                            question['actual sentence'] = corpusSentencesIndexedByDocPara[question['docid']][question['answer_paragraph']][i]                           
                            found = True
                            break
                    if found == True:
                        break
                if found == True:
                    break
            i += 1      
            
# for remaining untagged answers, split the answer terms on '♠' symbol and match first part of each term with tag terms (in sentences of the specified paragraph of the specified document)
for question in train:
    found = False
    if question['actual answer tag'] == 'NA':
        i = 0
        for sent in listOfTaggedSentencesIndexedByDocsPara[question['docid']][question['answer_paragraph']]:
            for tag in sent:
                for term in question['text'].split():              
                    if term.split('♠')[0].lower() == tag[0].lower():
                        question['actual answer tag'] = tag[1]
                        question['actual sentence'] = corpusSentencesIndexedByDocPara[question['docid']][question['answer_paragraph']][i]
                        found = True
                        break
                if found == True:
                    break
            if found == True:
                break
            i += 1
       
# for remaining untagged answers, remove the space between first two tag terms and then match with answer terms (in sentences of the specified paragraph of the specified document)
for question in train:
    found = False
    if question['actual answer tag'] == 'NA':
        i = 0
        for sent in listOfTaggedSentencesIndexedByDocsPara[question['docid']][question['answer_paragraph']]:
            for tag in sent:
                terms = question['text'].split()
                if len(terms) >= 2:
                    term = ''.join([terms[0], terms[1]])
                    for z in range(2, len(terms)):
                        term = ' '.join([term, terms[z]])
                    if term.lower() == tag[0].lower():
                        question['actual answer tag'] = tag[1]
                        question['actual sentence'] = corpusSentencesIndexedByDocPara[question['docid']][question['answer_paragraph']][i]
                        found = True
                        break
                if found == True:
                    break
            i += 1  
              
# store list of remaining untagged answers for manual checking (time permits)          
unTaggedQuestions = []
for question in train:
    if question['actual answer tag'] == 'NA':
        unTaggedQuestions.append(question)  

# for each term, store the number of times it was assigned a specific tag
termTagFreqDict = defaultdict(dict)
for doc in listOfTaggedSentencesIndexedByDocs:
    for sent in doc:
        for tag in sent:
            termTagFreqDict[tag[0].lower()][tag[1]] = termTagFreqDict[tag[0].lower()].get(tag[1], 0) + 1

# example of occasionally misclassified term
print('Example of occasionally misclassified term: \n\n \'Einstein\' - ', termTagFreqDict['einstein'])  

# re-tag terms by voting
for doc in listOfTaggedSentencesIndexedByDocs:
    for sent in doc:
        for tag in sent:
            mostFrequestTagOfTerm = max(termTagFreqDict[tag[0].lower()].items(), key=operator.itemgetter(1))[0]
            if tag[1] != mostFrequestTagOfTerm:
                tag[1] = mostFrequestTagOfTerm 
                
# exploratory data analysis: for each answer tag, store the frequency
tagFreqDict = {}
for question in train:
        tagFreqDict[question['actual answer tag']] = tagFreqDict.get(question['actual answer tag'], 0) + 1   
   
    
###################### PART 3 - SENTENCE RETRIEVAL ###############################  
                    
# function to return the most relevant sentence using BM25 ranking 
# which contains the expected answer tag 
# such that the tagged answer term(s) does not occur in question     
def BM25SentenceRelevance(query, docID, answerTag, corpusTokenized, listOfTaggedSentencesIndexedByDocs):
    remPunctQuery = query.translate(translation)         
    queryTokens = word_tokenize(remPunctQuery)    

    tokenizedQuery = []
    for token in queryTokens:
        if token not in stopwords:
            tokenizedQuery.append(stemmer.stem(token.lower()))
        
    bm25 = BM25(corpusTokenized[docID])
    average_idf = sum(float(val)  for val in bm25.idf.values()) / len(bm25.idf)
    scores = bm25.get_scores(tokenizedQuery, average_idf)

    # return the highest scoring sentence which has the expected answer tag     
    found = False
    finalScoreIndex = scores.index(max(scores))
    predictedAnswer = 'NA'
    if answerTag != 'NA':
        sortedScores = scores.copy()
        sortedScores.sort(reverse=True)
        for scoreValue in sortedScores:
            if scoreValue == 0:         
                zeroScores = []
                i = 0
                for zeroScoreValue in scores:
                    if zeroScoreValue == 0:
                        zeroScores.append(i)
                    i += 1            
               
                for zeroScoreIndex in zeroScores:
                    for tag in listOfTaggedSentencesIndexedByDocs[docID][zeroScoreIndex]:
                        if (tag[1] == answerTag) and (tag[0].translate(translation) != ''):
                            foundinQuest = False
                            for firstTerm in queryTokens:
                                if tag[0].translate(translation) in remPunctQuery:
                                    foundinQuest = True
                                    break
                            if foundinQuest == False:
                                found = True
                                finalScoreIndex = zeroScoreIndex
                                predictedAnswer = tag[0]
                                break  
                    if found == True:
                        break
                found = True            
            else:
                scoreIndex = scores.index(scoreValue)
                for tag in listOfTaggedSentencesIndexedByDocs[docID][scoreIndex]:
                    if (tag[1] == answerTag) and (tag[0].translate(translation) != ''):
                        foundinQuest = False
                        for firstTerm in queryTokens:
                            if tag[0].translate(translation) in remPunctQuery:
                                foundinQuest = True
                                break
                        if foundinQuest == False:
                            found = True
                            finalScoreIndex = scoreIndex
                            predictedAnswer = tag[0]
                            break
            if found == True:
                break
     
    return((docID, finalScoreIndex, predictedAnswer))

# randomly split train into 5 sets for validation
trainDup = train.copy()
np.random.shuffle(trainDup)
trainSplits = []
window = 1000
for j in range(0,5):
    start = window*j
    trainSplits.append(trainDup[start:(start + window)].copy())
    
# check accuracy of retrieved sentences on each of 5 train splits using BM25 ranking
splitAccSentRet = []
splitAccAnsRet = []
k = 0
for split in trainSplits:
    correctSentRet = 0
    correctAnsRet = 0
    total = 0
    for question in split:
        total += 1    
        docID, sentID, predictedAnswer = BM25SentenceRelevance(question['question'], question['docid'], question['actual answer tag'], corpusTokenized, listOfTaggedSentencesIndexedByDocs)      
        predictedSentence = corpusSentencesIndexedByDoc[docID][sentID]
        question['predicted sentence'] = predictedSentence
        question['predicted answer'] = predictedAnswer
        if question['text'] in predictedSentence:
            correctSentRet += 1
        if question['text'] == predictedAnswer:
            correctAnsRet += 1

    splitAccSentRet.append((float(correctSentRet)/total) * 100)
    splitAccAnsRet.append((float(correctAnsRet)/total) * 100)
    k += 1

print('--------Sentence Retrieval Accuracy--------')
for k in range(0,len(trainSplits)):
    print('Split: ', k, '          Accuracy: ', splitAccSentRet[k])
    
print('\n\n--------Answer Retrieval Accuracy--------')
for k in range(0,len(trainSplits)):
    print('Split: ', k, '          Accuracy: ', splitAccAnsRet[k])
    
# check accuracy of retrieved sentences and predicted answers on overall train set using BM25 ranking
total = 0
correctSentRet = 0
correctAnsRet = 0
for question in train:
    total += 1    
    question['predicted sentence'] = 'NA'
    docID, sentID, predictedAnswer = BM25SentenceRelevance(question['question'], question['docid'], question['actual answer tag'], corpusTokenized, listOfTaggedSentencesIndexedByDocs)      
    predictedSentence = corpusSentencesIndexedByDoc[docID][sentID]
    question['predicted sentence'] = predictedSentence
    question['predicted answer'] = predictedAnswer
    if question['text'] in predictedSentence:
        correctSentRet += 1
    if question['text'] == predictedAnswer:
        correctAnsRet += 1
print('Overall Train:          Sentence Retrieval Accuracy: ', (float(correctSentRet)/total) * 100)
print('Overall Train:          Answer Retrieval Accuracy: ', (float(correctAnsRet)/total) * 100)

###################### PART 4 - EXPECTED ANSWER TYPE PREDICTION ######################### 
  
# for each tag, build a dictionary of unigram term frequency 
neTagQuestTermDict = defaultdict(dict)
for question in train:
    remPunctQuest = question['question'].translate(translation) 
    for token in word_tokenize(remPunctQuest):     
        first = token.lower()
        neTagQuestTermDict[question['actual answer tag']][first] = neTagQuestTermDict[question['actual answer tag']].get(first, 0) + 1   

# for each tag, add bigram term frequency to the dictionary
for question in train:
    remPunctQuest = question['question'].translate(translation) 
    tokens = word_tokenize(remPunctQuest)
    for i in range(0, len(tokens)-1):
        first = tokens[i].lower()
        second = tokens[i+1].lower()
        neTagQuestTermDict[question['actual answer tag']][(first, second)] = neTagQuestTermDict[question['actual answer tag']].get((first, second), 0) + 1   

# for each tag, add trigram term frequency to the dictionary
for question in train:
    remPunctQuest = question['question'].translate(translation) 
    tokens = word_tokenize(remPunctQuest)
    for i in range(0, len(tokens)-2):
        first = tokens[i].lower()
        second = tokens[i+1].lower()
        third = tokens[i+2].lower()
        neTagQuestTermDict[question['actual answer tag']][(first, second, third)] = neTagQuestTermDict[question['actual answer tag']].get((first, second, third), 0) + 1   

# for each tag, sort terms according to frequency in descending order
sortedNETagQuestTermDict = {}
listofTuples = []
for tags in neTagQuestTermDict:
    for elements in neTagQuestTermDict[tags]:
        listofTuples.append((elements, neTagQuestTermDict[tags][elements]))
    listofTuples = sorted(listofTuples, key=itemgetter(1), reverse = True)
    sortedNETagQuestTermDict[tags] = listofTuples

# select top 800 terms of each tag as features
features = []
for tags in sortedNETagQuestTermDict:
    for i in range(0, 800):
        features.append(sortedNETagQuestTermDict[tags][i][0])

# remove duplicates from features
featList = []
for f in features:
    if f not in featList:
        featList.append(f)
   
# prepare dataset for training the classifier to predict answer tags from the feature terms occurring in sentence
# if a feature term is present in the question, mark it as 1 else 0
# target column is the named entity tag of the answer to question
allRecords = []
for question in train:
    if question['actual answer tag'] != 'NA':
        record = []
        q = question['question'].translate(translation)
        queryTokens = nltk.word_tokenize(q.lower())
        
        for f in featList:
            found = False
            for token in queryTokens:
                
                first = queryTokens[queryTokens.index(token)]
                second = ''
                third = ''
                
                if queryTokens.index(token)+1 <= len(queryTokens)-1:
                    second = queryTokens[queryTokens.index(token)+1]
                if queryTokens.index(token)+2 <= len(queryTokens)-1:
                    third = queryTokens[queryTokens.index(token)+2]
                
                if f == first:
                    found = True
                    record.append(1)
                    break
                elif f == (first,second):
                    found = True
                    record.append(1)
                    break
                elif f == (first,second,third):
                    found = True
                    record.append(1)
                    break
                
            if found == False:
                record.append(0)
        
        record.append(question['actual answer tag'])
        allRecords.append(record)

featList.append('target')
lgbmTrainDF = pd.DataFrame.from_records(allRecords, columns = featList)

####### using Light GBM classifier to predict answer tags

# transform target column from string to categorical codes and retain mapping to convert back the predicted codes
lgbmTrainDF.target = pd.Categorical(lgbmTrainDF.target)
catToCodeMapping = list(set(zip(lgbmTrainDF.target.cat.codes, lgbmTrainDF.target)))
lgbmTrainDF['target'] = lgbmTrainDF.target.cat.codes

# separate features and target
yTrain = lgbmTrainDF.pop('target')
xTrain = lgbmTrainDF
      
# transform tuples in feature names to conactenated strings for use in Light GBM
featXTrain = list(xTrain)
xgbFeatNames = []
for i in range(0,len(featXTrain)):
    if type(featXTrain[i]) == tuple:
        xgbFeatNames.append(''.join(list(featXTrain[i])))
    else:
        xgbFeatNames.append(featXTrain[i])

# Light GBM compatible training data set        
dTrainLGB = lgbm.Dataset(data = xTrain, label = yTrain, feature_name = xgbFeatNames)

# setting up Light GBM parameters for multi-class classification
params = {}
params['objective'] = 'multiclass'
params['metric'] = 'multi_logloss'
params['num_class'] = 6
# tuning parameters of Light GBM
params['learning_rate'] = 0.06
params['colsample_bytree'] = 0.8
#find optimum value of num_boost_round using 3-fold cross-validation
cvResults = lgbm.cv(
        params,
        dTrainLGB,
        nfold=3,
        stratified=True,
        num_boost_round = 10000,
        early_stopping_rounds = 50,
        seed = 3,
        verbose_eval=20
        )
    
bestNRound = cvResults['multi_logloss-mean'].index(np.min(cvResults['multi_logloss-mean']))
bestScore = np.min(cvResults['multi_logloss-mean'])

###### parameter tuning of LightGBM using 3-fold cross-validation
#paramTunningResultsDict = {}

# set different values of parameter to be tuned to see what works best
#paramValue = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]

# for each value of the parameter, get the 3-fold cross-validation results
#for value in paramValue:
#    params['colsample_bytree'] = value

#    cvResults = lgbm.cv(
#            params,
#            dTrainLGB,
#            nfold=3,
#            stratified=True,
#           num_boost_round = 10000,
#           early_stopping_rounds = 50,
#            seed = 3,
#            verbose_eval=20
#            )
    
#    bestNRound = cvResults['multi_logloss-mean'].index(np.min(cvResults['multi_logloss-mean']))
#    bestScore = np.min(cvResults['multi_logloss-mean'])
    
#    print(bestNRound)
#    print(bestScore)
    
 #   paramTunningResultsDict[' '.join(['colsample_bytree: ', str(params['colsample_bytree'])])] = (bestNRound, bestScore)
###### end of parameter tuning code
   
###### use stratified sampling to split full train into train and test splits and check accuracy of the model on test split
# perform stratified splits for train and test
x_train, x_test, y_train, y_test = train_test_split(xTrain, yTrain, test_size=0.33, random_state=42, stratify=yTrain)

# Light GBM compatible training data set        
dTrainLGB = lgbm.Dataset(data = x_train, label = y_train, feature_name = xgbFeatNames)

# train the Light GBM model on entire train dataset
model = lgbm.train(params, dTrainLGB, num_boost_round=bestNRound)

# use the trained model to get prediction probablities for each class on test split
preds = model.predict(x_test)

# select the class with highest probability as the predicted class in test split
predictions = []
for x in preds:
    predictions.append(np.argmax(x))

# calculate accuracy of test split
y_test = list(y_test)
correct = 0
total = len(predictions)
for i in range(0, total):
    if predictions[i] == y_test[i]:
        correct += 1

print('Answwer Tag Prediction Accuracy on Test Split: ', (float(correct)/total) * 100)
###### end of checking accuracy code
   
################# PART 5 - PREDICT ANSWERS OF QUESTIONS IN TEST SET ##################### 

# predict expected answer tag of test set questions
# train Light GBM model on full training dataset with tuned parameters
dTrainLGB = lgbm.Dataset(data = xTrain, label = yTrain, feature_name = xgbFeatNames)
model = lgbm.train(params, dTrainLGB, num_boost_round=bestNRound)

# prepare test dataset for use in Light GBM to predict answer tags from feature terms occurring in question
# if a feature term is present in the question, mark it as 1 else 0
allRecords = []
for question in test:
    record = []
    q = question['question'].translate(translation)
    queryTokens = nltk.word_tokenize(q.lower())
    
    for f in featList:
        found = False
        
        for token in queryTokens:          
            first = queryTokens[queryTokens.index(token)]
            second = ''
            third = ''
            
            if queryTokens.index(token)+1 <= len(queryTokens)-1:
                second = queryTokens[queryTokens.index(token)+1]
                
            if queryTokens.index(token)+2 <= len(queryTokens)-1:
                third = queryTokens[queryTokens.index(token)+2]
            
            if f == first:
                found = True
                record.append(1)
                break
            elif f == (first,second):
                found = True
                record.append(1)
                break
            elif f == (first,second,third):
                found = True
                record.append(1)
                break
            
        if found == False:
            record.append(0)
    
    allRecords.append(record)
lgbmTestDF = pd.DataFrame.from_records(allRecords, columns = featList)

# predict answer tags
preds = model.predict(lgbmTestDF)
# select the class with highest probability as the predicted class in test split
predictions = []
for x in preds:
    predictions.append(np.argmax(x))
    
# convert categorical codes back to categories
for i in range(0, len(predictions)):
    for cat in catToCodeMapping:
        if cat[0] == predictions[i]:
            predictions[i] = cat[1]

# save predicted answer tags for each question in test set
for i in range(0, len(test)):
    test[i]['predicted answer tag'] = predictions[i]
    
# predict answers of test set questions
questID = 0
submission = pd.DataFrame(columns = ['id','answer'])
for question in test: 
    question['predicted sentence'] = 'NA'
    question['predicted answer'] = 'NA'
    docID, sentID, predictedAnswer = BM25SentenceRelevance(question['question'], question['docid'], question['predicted answer tag'], corpusTokenized, listOfTaggedSentencesIndexedByDocs)      
    question['predicted sentence'] = corpusSentencesIndexedByDoc[docID][sentID]
    
    if len(predictedAnswer) > 3:
        splitted = predictedAnswer.split()
        predictedAnswer = ' '.join(splitted[0:4])        
    
    question['predicted answer'] = predictedAnswer

    submission = submission.append(pd.DataFrame({"id": [questID], "answer": [predictedAnswer]}))
    questID += 1

# generate kaggle submission
submission.to_csv('Submission.csv', index=False, columns=["id", "answer"])


