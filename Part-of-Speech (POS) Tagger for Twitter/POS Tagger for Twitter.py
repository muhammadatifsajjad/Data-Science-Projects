import nltk
import re
import urllib
import numpy as np
from sklearn.metrics import classification_report
from nltk.corpus import treebank, twitter_samples
from sklearn.metrics import accuracy_score as acc

urllib.request.urlretrieve("https://github.com/aritter/twitter_nlp/raw/master/data/annotated/pos.txt","pos.txt")
corpus = treebank.tagged_sents()
vocab = {}
tagset = {}

preProcessedCorpusPTBTrain = []
for sent in corpus:
    num_sent = []
    for word, tag in sent:
        wi = vocab.setdefault(word.lower().strip(), len(vocab))
        ti = tagset.setdefault(tag, len(tagset))
        num_sent.append((wi, ti))
    preProcessedCorpusPTBTrain.append(num_sent)
    
print('\nFirst sentence in preprocessed PTB train corpus: \n', preProcessedCorpusPTBTrain[0])
print('\nIndex for the word electricity: \n', vocab['electricity'])
print('\nLength of the full tagset: \n', len(tagset))

preProcessedCorpusTwitterTrain = []
for tokens in twitter_samples.tokenized():
    num_sent = []
    for word in tokens:
        #pre-processing on words
        word = word.lower().strip()
        word = re.sub(r'^@.*', 'USER_TOKEN', word)
        word = re.sub(r'^#.*', 'HASHTAG_TOKEN', word)
        word = re.sub(r'^http(s)?://.*', 'URL_TOKEN', word)
        word = re.sub('^rt$', 'RETWEET_TOKEN', word)

        wi = vocab.setdefault(word, len(vocab))
        num_sent.append(wi)   
    preProcessedCorpusTwitterTrain.append(num_sent)

print('\nFirst sentence in preprocessed twitter_samples train corpus: \n', preProcessedCorpusTwitterTrain[0])
print('\nIndex for the word electricity: \n', vocab['electricity'])
print('\nIndex for HASHTAG_TOKEN: \n', vocab['HASHTAG_TOKEN'])

tagset.setdefault('USR', len(tagset))
tagset.setdefault('HT', len(tagset))
tagset.setdefault('RT', len(tagset))
tagset.setdefault('URL', len(tagset))
tagset.setdefault('VPP', len(tagset))
tagset.setdefault('TD', len(tagset))
tagset.setdefault('O', len(tagset))

vocab.setdefault('<unk>', len(vocab))

invVocab = [None] * len(vocab)
for word, index in vocab.items():
    invVocab[index] = word
invTagset = [None] * len(tagset)
for tag, index in tagset.items():
    invTagset[index] = tag
    
print('\nIndex for ''<unk>'': \n', vocab['<unk>'])
print('\nLength of resulting tagset: \n', len(invTagset))

preProcessedCorpusTwitterTest = []
with open('pos.txt') as f:
    wordsTags = []
    for line in f:
        if line.strip() == '':
            preProcessedCorpusTwitterTest.append(wordsTags)
            wordsTags = []
        else:
            word, tag = line.strip().split()          
            #pre-processing on words
            word = word.lower().strip()
            word = re.sub(r'^@.*', 'USER_TOKEN', word)
            word = re.sub(r'^#.*', 'HASHTAG_TOKEN', word)
            word = re.sub(r'^http(s)?://.*', 'URL_TOKEN', word)
            word = re.sub('^rt$', 'RETWEET_TOKEN', word)
            #pre-processing on tags
            tag = tag.replace("(", "-LRB-")
            tag = tag.replace(")", "-RRB-")
            tag = tag.replace("NONE", "-NONE-")
         
            wi = vocab.get(word, vocab.get('<unk>'))
            ti = tagset.get(tag)
            wordsTags.append((wi, ti))
            
print('\nFirst sentence in preprocessed twitter test corpus: \n', preProcessedCorpusTwitterTest[0])

def count(corpus, vocab, tagset):
    S = len(tagset)
    V = len(vocab)
    
    # initalise
    eps = 0.1
    pi = eps * np.ones(S)
    A = eps * np.ones((S, S))
    O = eps * np.ones((S, V))
    
    # count
    for sent in corpus:
        last_tag = None
        for word, tag in sent:
            O[tag, word] += 1
            if last_tag == None:
                pi[tag] += 1
            else:
                A[last_tag, tag] += 1
            last_tag = tag
            
    # normalise
    pi /= np.sum(pi)
    for s in range(S):
        O[s,:] /= np.sum(O[s,:])
        A[s,:] /= np.sum(A[s,:])
    
    return pi, A, O
    
[initialMatrix, transitionMatrix, emissionMatrix] = count(preProcessedCorpusPTBTrain, vocab, tagset)

def viterbi(params, observations):
    pi, A, O = params
    M = len(observations)
    S = pi.shape[0]
    
    alpha = np.zeros((M, S))
    alpha[:,:] = float('-inf')
    backpointers = np.zeros((M, S), 'int')
    
    # base case
    alpha[0, :] = pi * O[:,observations[0]]
    
    # recursive case
    for t in range(1, M):
        for s2 in range(S):
            for s1 in range(S):
                score = alpha[t-1, s1] * A[s1, s2] * O[s2, observations[t]]
                if score > alpha[t, s2]:
                    alpha[t, s2] = score
                    backpointers[t, s2] = s1
    
    # now follow backpointers to resolve the state sequence
    ss = []
    ss.append(np.argmax(alpha[M-1,:]))
    for i in range(M-1, 0, -1):
        ss.append(backpointers[i, ss[-1]])
        
    return list(zip(observations, list(reversed(ss))))

predictions = []
for sent in preProcessedCorpusTwitterTest:
    encoded_sent = [wordTags[0] for wordTags in sent]
    pred = viterbi((initialMatrix, transitionMatrix, emissionMatrix), encoded_sent)
    predictions.append(pred)

print('\nFirst sentence of predicted list: \n', predictions[0])

def getOrigTags(sentences, tagInvIndex):
    tagSequence = []
    for wordTagsList in sentences:
        for wordTags in wordTagsList:
            tagSequence.append(tagInvIndex[wordTags[1]])
    return tagSequence

tagSequenceTest = getOrigTags(preProcessedCorpusTwitterTest, invTagset)
tagSequencePred = getOrigTags(predictions, invTagset)

print('\nAccuracy:\n', round(acc(tagSequenceTest, tagSequencePred) * 100, 1), '%')

emissionMatrix[tagset['USR']] = 0.0
emissionMatrix[tagset['USR']][vocab['USER_TOKEN']] = 1.0

emissionMatrix[tagset['HT']] = 0.0
emissionMatrix[tagset['HT']][vocab['HASHTAG_TOKEN']] = 1.0

emissionMatrix[tagset['URL']] = 0.0
emissionMatrix[tagset['URL']][vocab['URL_TOKEN']] = 1.0

emissionMatrix[tagset['RT']] = 0.0
emissionMatrix[tagset['RT']][vocab['RETWEET_TOKEN']] = 1.0

print('\nEmission Matrix:\n', emissionMatrix)

predictions = []
for sent in preProcessedCorpusTwitterTest:
    encoded_sent = [wordTags[0] for wordTags in sent]
    pred = viterbi((initialMatrix, transitionMatrix, emissionMatrix), encoded_sent)
    predictions.append(pred)
    
tagSequenceTest = getOrigTags(preProcessedCorpusTwitterTest, invTagset)
tagSequencePred = getOrigTags(predictions, invTagset)

print('\nAccuracy:\n', round(acc(tagSequenceTest, tagSequencePred) * 100, 1), '%')    
print('\nClasification report:\n', classification_report(tagSequenceTest, tagSequencePred))
print("\nBest tags having F1-Score >= 0.9:\n\"TO\", \"WRB\", \",\", \"CC\", \"URL\", \"USR\", \"HT\", \"RT\"") 
print("\nWorst tags having F1-Score <= 0.1:\n\"#\", \"$\", \"-LRB-\", \"-NONE-\", \"FW\", \"LS\", \"NNPS\", \"O\", \"PDT\", \"SYM\", \"TD\", \"UH\", \"VPP\", \"WP$\", \"``\", \"''\", \"-RRB-\"")
print("\nTraining is done on a subset of Penn Treebank (PTB) corpus which is freely available with NLTK. As such, there are a lot of words which are not present in the PTB corpus we used - word vocabulary from PTB corpus consists of 11387 word types which increases to 26069 words when we add new words from twitter training dataset (which does not have a corresponding POS tag). Also, there are tags that either do not appear in PTB dataset at all e.g. \"TD\", \"VPP\", \"O\", or they appear with very low frequency e.g. \"UH\" which appears only 3 times in PTB. Further, all the worse performing tags have a very low support meaning that they occur very infrequently in the test dataset. One exception, however, is the \"UH\" (interjection) tag which appears 493 times in test dataset, and the reason for it's poor performance is because twitter dataset is expected to have smileys and internet language slangs such as 'lol', 'hahaha', ':)', 'omg' etc. having \"UH\" tags which are not present in PTB tagged corpus (PTB only has three words 'OK', 'no', 'Oh' having \"UH\" tags).\n\nSome ways to improve the tagger are (1) to include smileys and common internet language expression slangs, such as the ones mentioned above, with \"UH\" tag in our dictionaries and explicitly set the emission probabilities just like we did for special tokens like \"URL_TOKEN\", (2) to use a more comprehensive tagged corpus, and (3) instead of bigrams, use model based on trigrams i.e. compute probability of a tag given its last two tags.")