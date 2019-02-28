import nltk
import operator
import math
from nltk.corpus import brown
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import cosine as cos_distance
from gensim.models import Word2Vec 
from scipy.stats.stats import pearsonr

# Load 'combined.tab' file in dictionary
with open('../combined.tab') as tabFile:
    next(tabFile)
    tabSepWords = (line.split('\t') for line in tabFile)
    wordSimDict = {(words[0],words[1]):float(words[2]) for words in tabSepWords}  
    
# for each paragraph in brown corpus, store a list of lower-cased, lemmatized word types
lemmatizer = WordNetLemmatizer()
brownParas = []
for paragraphs in brown.paras():
    wordTypes = set()
    wordTypes.update([lemmatizer.lemmatize(words.lower()) for sentences in paragraphs for words in sentences])
    brownParas.append(wordTypes)

# create a dictionary of document frequency for word types in brown corpus
wordTypeDocFreqDict = {}
for paragraphs in brownParas:
    for word in paragraphs:
        wordTypeDocFreqDict[word] = wordTypeDocFreqDict.get(word,0) + 1

# filter word pairs where frequency of either one of them is less than 10
for word1, word2 in list(wordSimDict):
    if (wordTypeDocFreqDict.get(word1,0) < 10) | (wordTypeDocFreqDict.get(word2,0) < 10):
        wordSimDict.pop((word1, word2), None)

# store single noun primary sense of words in dictionary, and
# filter word pairs where single noun primary sense is not found
primarySenseSynsetDict = {}
for wordPairs in list(wordSimDict):
    for word in wordPairs:
        haveNounPrimarySense = False
        synsets = wn.synsets(word)
        if (len(synsets) == 1 and synsets[0].pos() == 'n'):
            haveNounPrimarySense = True
            primarySenseSynsetDict[word] = synsets[0]
        elif (len(synsets) > 1):
            lemmaCounts = {}
            for synset in synsets:
                for lemma in synset.lemmas():
                    lemmaName = lemma.name()
                    if lemmaName == word:
                        lemmaCounts[lemma, synset] = lemma.count()  
            lemmaCounts = sorted(lemmaCounts.items(), key=operator.itemgetter(1), reverse=True)            
            if (len(lemmaCounts) == 1) and (lemmaCounts[0][0][1].pos() == 'n') and (lemmaCounts[0][1] >= 5):
                haveNounPrimarySense = True
            elif len(lemmaCounts) > 1:
                if(lemmaCounts[0][1] >= 5) and (lemmaCounts[0][1] >= lemmaCounts[1][1]*5) and (lemmaCounts[0][0][1].pos() == 'n'):
                    haveNounPrimarySense = True
            if haveNounPrimarySense == True:   
                primarySenseSynsetDict[word] = lemmaCounts[0][0][1]
        if haveNounPrimarySense == False:
            wordSimDict.pop(wordPairs, None) 
            break

print(wordSimDict.keys())

# create dictionary of wordpair/Wu-Palmer-similarity mappings for filtered word pairs
wuPalmerSimilarityDict={}
for word1, word2 in wordSimDict:
    wuPalmerSimilarityDict[word1,word2] = primarySenseSynsetDict[word1].wup_similarity(primarySenseSynsetDict[word2])

print(wuPalmerSimilarityDict)

# create dictionary of wordpair/PPMI-similarity mappings for filtered word pairs
pmiSimilarityDict={}
totalParasCount = float(len(brownParas))
for word1, word2 in wordSimDict:
    wordCount1 = 0
    wordCount2 = 0
    bothWordCount = 0
    for paras in brownParas:
        if word1 in paras:
            wordCount1 += 1
            if word2 in paras:
                bothWordCount += 1
        if word2 in paras:
            wordCount2 += 1
    probCalc = (bothWordCount/totalParasCount)/((wordCount1/totalParasCount)*(wordCount2/totalParasCount))
    pmiSimilarityDict[word1, word2] = 0.0 if probCalc==0 else math.log(probCalc, 2)

print(pmiSimilarityDict)

# bag-of-words implementation
def get_BOW(text):
    BOW = {}
    for word in text:
        BOW[word.lower()] = BOW.get(word.lower(),0) + 1
    return BOW
    
# get frequency of words in paras using word types list created above 
# matrix of 0's and 1's depending on if the word exists in a paragraph or not
texts = []
for paras in brownParas:
    texts.append(get_BOW(paras))

# create words-paragraph frequency matrix
vectorizer = DictVectorizer()
brownMatrix = vectorizer.fit_transform(texts).transpose()

# get dense vectors of length 500 using truncated SVD
svd = TruncatedSVD(n_components=500)
brownMatrixSVD = svd.fit_transform(brownMatrix)

# create dictionary of wordpair/cosine-similarity mappings using LSA method for filtered word pairs
cosineSimilarityDict = {}
for word1, word2 in wordSimDict:
    word1Index = vectorizer.feature_names_.index(word1)
    word2Index = vectorizer.feature_names_.index(word2)
    cosSim = 1 - cos_distance(brownMatrixSVD[word1Index,:], brownMatrixSVD[word2Index,:])
    cosineSimilarityDict[word1, word2] = cosSim

print(cosineSimilarityDict)   
    
# create dictionary of wordpair/word2vec-similarity mappings for filtered word pairs using sentences from brown corpus
brownSentences = nltk.corpus.brown.sents()
model = Word2Vec(brownSentences, min_count=5, size=500, iter=50)
word2vecSimilarityDict = {}
for word1, word2 in wordSimDict:
    word2vecSimilarityDict[word1, word2] = model.wv.similarity(word1, word2)

print(word2vecSimilarityDict)

# compare similarities with the gold standard using pearson correlation co-efficient
wordSimGoldStanardList = list(wordSimDict.values())
def pearsonCorrelation(wordSimilarityDict):
    wordSimilarityList = list(wordSimilarityDict.values())
    return pearsonr(wordSimGoldStanardList, wordSimilarityList)[0]

print('Pearson correlation coefficient compared with Gold Standard:')
print('------------------------------------------------------------')
print('Cosine: ', pearsonCorrelation(cosineSimilarityDict))
print('PMI: ', pearsonCorrelation(pmiSimilarityDict))
print('Wu-Palmer: ', pearsonCorrelation(wuPalmerSimilarityDict))
print('Word2Vec: ', pearsonCorrelation(word2vecSimilarityDict))