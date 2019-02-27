import nltk
import numpy as np
from nltk.corpus import brown    

############################## HANGMAN CODE ################################################

# allowing better python 2 & python 3 compatibility 
from __future__ import print_function 

def hangman(secret_word, guesser, max_mistakes=8, verbose=True, **guesser_args):
    """
        secret_word: a string of lower-case alphabetic characters, i.e., the answer to the game
        guesser: a function which guesses the next character at each stage in the game
            The function takes a:
                mask: what is known of the word, as a string with _ denoting an unknown character
                guessed: the set of characters which already been guessed in the game
                guesser_args: additional (optional) keyword arguments, i.e., name=value
        max_mistakes: limit on length of game, in terms of allowed mistakes
        verbose: be chatty vs silent
        guesser_args: keyword arguments to pass directly to the guesser function
    """
    secret_word = secret_word.lower()
    mask = ['_'] * len(secret_word)
    guessed = set()
    if verbose:
        print("Starting hangman game. Target is", ' '.join(mask), 'length', len(secret_word))
    
    mistakes = 0
    while mistakes < max_mistakes:
        if verbose:
            print("You have", (max_mistakes-mistakes), "attempts remaining.")
        guess = guesser(mask, guessed, **guesser_args)

        if verbose:
            print('Guess is', guess)
        if guess in guessed:
            if verbose:
                print('Already guessed this before.')
            mistakes += 1
        else:
            guessed.add(guess)
            if guess in secret_word:
                for i, c in enumerate(secret_word):
                    if c == guess:
                        mask[i] = c
                if verbose:
                    print('Good guess:', ' '.join(mask))
            else:
                if verbose:
                    print('Sorry, try again.')
                mistakes += 1
                
        if '_' not in mask:
            if verbose:
                print('Congratulations, you won.')
            return mistakes
        
    if verbose:
        print('Out of guesses. The word was', secret_word)    
    return mistakes

def human(mask, guessed, **kwargs):
    """
    simple function for manual play
    """
    print('Enter your guess:')
    try:
        return raw_input().lower().strip() # python 3
    except NameError:
        return input().lower().strip() # python 2
    
############################## Q1 ################################################

# unique word types of lower-cased, alphabetic characters from brown corpus
wordTypes = set()
for words in brown.words():
    word = words.lower()
    if word.isalpha():
        wordTypes.add(word)       
uniqueWordTypes = list(wordTypes)

# randomly shuffle word types and split in train and test sets
np.random.shuffle(uniqueWordTypes)
testSet = uniqueWordTypes[0:1000]
trainSet = uniqueWordTypes[1000:]

# print size of train and test sets
print("Number of word types in Test Set: ", len(testSet))
print("Number of word types in Train Set: ", len(trainSet))

############################## Q2 ################################################

# randomly return an alphabet which has not already been guessed
def randomGuesser(mask, guessed, **kwargs):
    alphabets = set(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
             'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
    alphabets = alphabets - guessed 
    return np.random.choice(list(alphabets))

# return average number of mistakes made by a given model in solving hangman across entire list of words
def hangmanPerformance(listOfWords, method, **kwargs):
    totalNoOfMistakes = 0
    for word in listOfWords:
        totalNoOfMistakes += hangman(word, method, 26, False, **kwargs)  
    return(totalNoOfMistakes / len(listOfWords))

print('Average number of mistakes for random guesser: ', hangmanPerformance(testSet, randomGuesser))  
  
############################## Q3 ################################################

import operator

# build alphabet frequency dictionary
totalCharacters = 0
charFreqDict = {}
for words in trainSet:
    for character in words:
        charFreqDict[character] = charFreqDict.get(character, 0) + 1 
        totalCharacters += 1              

# convert character frequency into probability                 
for alphabet in charFreqDict:
    charFreqDict[alphabet] = charFreqDict[alphabet] / totalCharacters

# store ordered list of alphabets based on frequency in descending order
charFreqDict = sorted(charFreqDict.items(), key=operator.itemgetter(1), reverse=True) 
orderedCharsList = []
for characters in charFreqDict:
    orderedCharsList.append(characters[0])

# return an alphabet with highest probability across all alphabets (appearing in words) in train set
def unigramGuesser(mask, guessed, **kwargs):
    remCharsList = [chars for chars in orderedCharsList if chars not in guessed]
    return remCharsList[0]

print('Average number of mistakes for unigram guesser: ', hangmanPerformance(testSet, unigramGuesser))  
     
############################## Q4 ################################################

import math
from collections import Counter

# get total number of different length words
lenOfWords=[]
for words in trainSet:
    lenOfWords.append(len(words))
wordLengthCounts = Counter(lenOfWords)

# get total count of alphabets for given length of word
charFreqDictByLength = {}
for words in trainSet:
    for character in words:
        charFreqDictByLength[character, len(words)] = charFreqDictByLength.get((character, len(words)),0) + 1

# convert alphabet frequencies per word length into probabilities, as follows:
# P(alphabet | wordLength) = (number of times an alphabet appeared in a word length) / (number of total alphabets across all words in that word length)
for a in charFreqDictByLength:
    charFreqDictByLength[a[0], a[1]] = charFreqDictByLength.get((a[0], a[1]), 0) / (a[1] * wordLengthCounts[a[1]])

# return probability given an alphabet and word length
# if alphabet, wordlength pair is not found check recursively in wordlength-1 
# until a pair is found or wordlength becomes 0
def getCharProbGivenLength(char,length):
    if length == 0:
        return 0.0
    else:
        if (char, length) in charFreqDictByLength:
            return charFreqDictByLength[char, length]
        else:
            return getCharProbGivenLength(char, length-1)

# return an alphabet with highest probability across all alphabets appearing in words of the same length as the secret word in train set
def lengthCondUnigramGuesser(mask, guessed):
    alphabets = set(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
             'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])    
    alphabets = alphabets - guessed 
    maskLength = len(mask)
    maxFreq = (-1,'_')
    for char in alphabets:
       if getCharProbGivenLength(char,maskLength) > maxFreq[0]:
        maxFreq = (getCharProbGivenLength(char,maskLength),char)
    return maxFreq[1]

print('Average number of mistakes for length conditioned unigram guesser: ', hangmanPerformance(testSet, lengthCondUnigramGuesser))  
  
############################## Q5 ################################################

from collections import defaultdict
from collections import Counter

def convertWord(word):
    return ["<s>","<s>","<s>","<s>"] + [alphabet for alphabet in word] + ["</s>"]

def getNgramProb(dataset):
    # initialize variables
    unigramProb = Counter()
    bigramProb = defaultdict(Counter)
    trigramProb = defaultdict(lambda: defaultdict(Counter))
    quadgramProb = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    pentgramProb = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(Counter))))

    # collect unigram counts
    for word in dataset:
        word = convertWord(word)
        for alphabet in word:
            unigramProb[alphabet] += 1
    
    # using alphabet frequency, compute probability of a given alphabet across all alphabets (appearing in words) in train set
    for alphabet in unigramProb:
        unigramProb[alphabet] = unigramProb[alphabet] / totalCharacters
    
    # collect bigram counts
    for word in dataset:
        word = convertWord(word)
        # generate a list of bigrams
        bigram1 = [word[i] for i in range(len(word)-1)]
        bigram2 = [word[i+1] for i in range(len(word)-1)]
        bigramList = zip(bigram1, bigram2)
        # iterate over bigrams
        for bigram in bigramList:
            first, second = bigram
            bigramProb[first][second] += 1
    
    # compute probability for second alphabet given first alphabet
    for first in bigramProb:
        for second in bigramProb[first]:
            bigramProb[first][second] = bigramProb[first][second] / sum(bigramProb[first].values())

    # collect trigram counts
    for word in dataset:
        word = convertWord(word)
        # generate a list of trigrams
        trigram1 = [word[i] for i in range(len(word)-2)]
        trigram2 = [word[i+1] for i in range(len(word)-2)]
        trigram3 = [word[i+2] for i in range(len(word)-2)]
        trigramList = zip(trigram1, trigram2, trigram3)
        # iterate over trigrams
        for trigram in trigramList:
            first, second, third = trigram
            trigramProb[first][second][third] += 1
    
    # compute probability for third alphabet given first and second alphabets
    for first in trigramProb:
        for second in trigramProb[first]:
            bigramCount = sum(trigramProb[first][second].values())
            for third in trigramProb[first][second]:
                trigramProb[first][second][third] = trigramProb[first][second][third] / bigramCount

    # collect quadgram counts
    for word in dataset:
        word = convertWord(word)
        # generate a list of quadgrams
        quadgram1 = [word[i] for i in range(len(word)-3)]
        quadgram2 = [word[i+1] for i in range(len(word)-3)]
        quadgram3 = [word[i+2] for i in range(len(word)-3)]
        quadgram4 = [word[i+3] for i in range(len(word)-3)]
        quadgramList = zip(quadgram1, quadgram2, quadgram3, quadgram4)
        # iterate over trigrams
        for quadgram in quadgramList:
            first, second, third, fourth = quadgram
            quadgramProb[first][second][third][fourth] += 1
    
    # compute probability for fourth alphabet given first, second, and third alphabets
    for first in quadgramProb:
        for second in quadgramProb[first]:
            for third in quadgramProb[first][second]:
                trigramCount = sum(quadgramProb[first][second][third].values())
                for fourth in quadgramProb[first][second][third]:
                    quadgramProb[first][second][third][fourth] = quadgramProb[first][second][third][fourth] / trigramCount
        
    # collect pentgram counts
    for word in dataset:
        word = convertWord(word)
        # generate a list of quadgrams
        pentgram1 = [word[i] for i in range(len(word)-4)]
        pentgram2 = [word[i+1] for i in range(len(word)-4)]
        pentgram3 = [word[i+2] for i in range(len(word)-4)]
        pentgram4 = [word[i+3] for i in range(len(word)-4)]
        pentgram5 = [word[i+4] for i in range(len(word)-4)]
        pentgramList = zip(pentgram1, pentgram2, pentgram3, pentgram4, pentgram5)
        # iterate over trigrams
        for pentgram in pentgramList:
            first, second, third, fourth, fifth = pentgram
            pentgramProb[first][second][third][fourth][fifth] += 1
    
    # compute probability for fifth alphabet given first, second, third, and fourth alphabets
    for first in pentgramProb:
        for second in pentgramProb[first]:
            for third in pentgramProb[first][second]:
                for fourth in pentgramProb[first][second][third]:
                    quadgramCount = sum(pentgramProb[first][second][third][fourth].values())
                    for fifth in pentgramProb[first][second][third][fourth]:
                        pentgramProb[first][second][third][fourth][fifth] = pentgramProb[first][second][third][fourth][fifth] / quadgramCount

    return(unigramProb, bigramProb, trigramProb, quadgramProb, pentgramProb)

# get probabilities for unigram, bigram, trigram, quadgram, and pentgram
unigramProb, bigramProb, trigramProb, quadgramProb, pentgramProb = getNgramProb(trainSet)

def trigramGuesser(mask, guessed, **kwargs):
    # initialize variables
    uniProb = defaultdict(Counter)
    biProb = defaultdict(Counter) 
    triProb = defaultdict(Counter) 
    interpProb = defaultdict(Counter) 
    alphabets = set(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
         'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
    alphabets = alphabets - guessed 
    mask = ["<s>","<s>"] + [char for char in mask]
    maskLength = len(mask)
    
    # get unigram, bigram, and trigram probabilities for each position
    for pos in range(2, maskLength):
        if (mask[pos] == '_'):
            for alphabet in alphabets:
                uniProb[pos][alphabet] = unigramProb[alphabet]
                biProb[pos][alphabet] = bigramProb[mask[pos-1]][alphabet]
                triProb[pos][alphabet] = trigramProb[mask[pos-2]][mask[pos-1]][alphabet]
    
    # parameters for linear interpolation
    lambdaUnigram = kwargs['lambdaUnigram']
    lambdaBigram = kwargs['lambdaBigram']
    lambdaTrigram = 1 - lambdaUnigram - lambdaBigram
    
    # sum all the vector of probabilities over each alphabet, to get a vector of scores for each alphabet     
    for alphabet in alphabets:
        interpProb[alphabet] = 0
        for pos in uniProb:
            interpProb[alphabet] += (lambdaUnigram * uniProb[pos][alphabet]) + (lambdaBigram * biProb[pos][alphabet]) + (lambdaTrigram * triProb[pos][alphabet])
    
    # return the alphabet which has the highest score
    return(max(interpProb, key=interpProb.get))    
        
print('Average number of mistakes for trigram guesser: ', hangmanPerformance(testSet, trigramGuesser, lambdaUnigram=0.1, lambdaBigram=0.0))  

def quadgramGuesser(mask, guessed, **kwargs):
    # initialize variables
    uniProb = defaultdict(Counter)
    biProb = defaultdict(Counter) 
    triProb = defaultdict(Counter) 
    quadProb = defaultdict(Counter) 
    interpProb = defaultdict(Counter) 
    alphabets = set(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
         'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
    alphabets = alphabets - guessed 
    mask = ["<s>","<s>","<s>"] + [char for char in mask]
    maskLength = len(mask)
    
    # get unigram, bigram, trigram, and quadgram probabilities for each position
    for pos in range(3, maskLength):
        if (mask[pos] == '_'):
            for alphabet in alphabets:
                uniProb[pos][alphabet] = unigramProb[alphabet]
                biProb[pos][alphabet] = bigramProb[mask[pos-1]][alphabet]
                triProb[pos][alphabet] = trigramProb[mask[pos-2]][mask[pos-1]][alphabet]
                quadProb[pos][alphabet] = quadgramProb[mask[pos-3]][mask[pos-2]][mask[pos-1]][alphabet]

    # parameters for linear interpolation
    lambdaUnigram = kwargs['lambdaUnigram']
    lambdaBigram = kwargs['lambdaBigram']
    lambdaTrigram = kwargs['lambdaTrigram']
    lambdaQuadgram = 1 - lambdaUnigram - lambdaBigram - lambdaTrigram
    
    # sum all the vector of probabilities over each alphabet, to get a vector of scores for each alphabet   
    for alphabet in alphabets:
        interpProb[alphabet] = 0
        for pos in uniProb:
            interpProb[alphabet] += (lambdaUnigram * uniProb[pos][alphabet]) + (lambdaBigram * biProb[pos][alphabet]) + (lambdaTrigram * triProb[pos][alphabet]) + (lambdaQuadgram * quadProb[pos][alphabet])

    # return the alphabet which has the highest score
    return(max(interpProb, key=interpProb.get))    

print('Average number of mistakes for quadgram guesser: ', hangmanPerformance(testSet, quadgramGuesser, lambdaUnigram=0.1, lambdaBigram=0.0, lambdaTrigram=0.3))  

def pentgramGuesser(mask, guessed, **kwargs):
    # initialize variables
    uniProb = defaultdict(Counter)
    biProb = defaultdict(Counter) 
    triProb = defaultdict(Counter) 
    quadProb = defaultdict(Counter) 
    pentProb = defaultdict(Counter) 
    interpProb = defaultdict(Counter) 
    alphabets = set(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
         'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
    alphabets = alphabets - guessed 
    mask = ["<s>","<s>","<s>","<s>"] + [char for char in mask]
    maskLength = len(mask)
    
    # get unigram, bigram, trigram, quadgram, and pentgram probabilities for each position
    for pos in range(4, maskLength):
        if (mask[pos] == '_'):
            for alphabet in alphabets:
                uniProb[pos][alphabet] = unigramProb[alphabet]
                biProb[pos][alphabet] = bigramProb[mask[pos-1]][alphabet]
                triProb[pos][alphabet] = trigramProb[mask[pos-2]][mask[pos-1]][alphabet]
                quadProb[pos][alphabet] = quadgramProb[mask[pos-3]][mask[pos-2]][mask[pos-1]][alphabet]
                pentProb[pos][alphabet] = pentgramProb[mask[pos-4]][mask[pos-3]][mask[pos-2]][mask[pos-1]][alphabet]

    # parameters for linear interpolation
    lambdaUnigram = kwargs['lambdaUnigram']
    lambdaBigram = kwargs['lambdaBigram']
    lambdaTrigram = kwargs['lambdaTrigram']
    lambdaQuadgram = kwargs['lambdaQuadgram']
    lambdaPentgram = 1 - lambdaUnigram - lambdaBigram - lambdaTrigram - lambdaQuadgram
    
    # sum all the vector of probabilities over each alphabet, to get a vector of scores for each alphabet       
    for alphabet in alphabets:
        interpProb[alphabet] = 0
        for pos in uniProb:
            interpProb[alphabet] += (lambdaUnigram * uniProb[pos][alphabet]) + (lambdaBigram * biProb[pos][alphabet]) + (lambdaTrigram * triProb[pos][alphabet]) + (lambdaQuadgram * quadProb[pos][alphabet]) + (lambdaPentgram * pentProb[pos][alphabet])

    # return the alphabet which has the highest score    
    return(max(interpProb, key=interpProb.get))    

print('Average number of mistakes for pentgram guesser: ', hangmanPerformance(testSet, pentgramGuesser, lambdaUnigram=0.1, lambdaBigram=0.0, lambdaTrigram=0.2, lambdaQuadgram=0.1))  

###########Finding best combination of lambdas########################
###Find best params for trigram
#lambdaUnigram=0.1, lambdaBigram=0.0; Avg. Mistakes: 8.642
for a in range(0, 11):
    for b in range(0, (11-a)):
        print(a/10, ' ', b/10)
        avgMist = hangmanPerformance(testSet, trigramGuesser, lambdaUnigram=a/10, lambdaBigram=b/10)
        print(avgMist)
        if avgMist>10:
            break

###Find best params for quadgram
#lambdaUnigram=0.1, lambdaBigram=0.0, lambdaTrigram=0.3 ; Avg. Mistakes: 8.181
for a in range(3, 11):
    for b in range(0, (11-a)):
        for c in range(0, (11-a-b)):
            print(a/10, ' ', b/10, '', c/10)
            avgMist = hangmanPerformance(testSet, quadgramGuesser, lambdaUnigram=a/10, lambdaBigram=b/10, lambdaTrigram=c/10)
            print(avgMist)
            if avgMist>10:
                break

###Find best params for pentgram
#lambdaUnigram=, lambdaBigram=, lambdaTrigram=, lambdaQuadgram= ; Avg. Mistakes: 
for a in range(0, 11):
    for b in range(0, (11-a)):
        for c in range(0, (11-a-b)):
            for d in range(0, (11-a-b-c)):
                print(a/10, ' ', b/10, '', c/10, '', d/10)
                avgMist = hangmanPerformance(testSet, pentgramGuesser, lambdaUnigram=a/10, lambdaBigram=b/10, lambdaTrigram=c/10, lambdaQuadgram=d/10)
                print(avgMist)
                if avgMist>10:
                    break                    
