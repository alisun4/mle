from nltk.tokenize import regexp_tokenize
import numpy as np
from collections import defaultdict
from tqdm import tqdm

default_pattern =  r"""(?x)                  
                        (?:[A-Z]\.)+          
                        |\$?\d+(?:\.\d+)?%?    
                        |\w+(?:[-']\w+)*      
                        |\.\.\.               
                        |(?:[.,;"'?():-_`])    
                    """

def tokenize(text, pattern = default_pattern):
    """Tokenize senten with specific pattern
    
    Arguments:
        text {str} -- sentence to be tokenized, such as "I love NLP"
    
    Keyword Arguments:
        pattern {str} -- reg-expression pattern for tokenizer (default: {default_pattern})
    
    Returns:
        list -- list of tokenized words, such as ['I', 'love', 'nlp']
    """
    # text = text
    return text.split()

class FeatureExtractor(object):
    """Base class for feature extraction.
    """
    def __init__(self):
        pass
    def fit(self, text_set):
        pass
    def transform(self, text):
        pass  
    def transform_list(self, text_set):
        pass



class UnigramFeature(FeatureExtractor):
    """Example code for unigram feature extraction
    """
    def __init__(self, smoothing_alpha):
        self.unigram = {}
        self.smoothing_alpha = smoothing_alpha
        self.num_tokens = 0
        
    def zero(self):
        return 0
    
    
    def two(self):
        return 2
        
        
    def fit(self, text_set: list):
        """Fit a feature extractor based on given data 
        
        Arguments:
            text_set {list} -- list of tokenized sentences and words are lowercased, such as [["I", "love", "nlp"], ["I", "like", "python"]]
        """
        self.unigram["<UNK>"] = 0
        index = 1
        count = defaultdict(self.zero)
        for i in range(0, len(text_set)):
            for j in range(0, len(text_set[i])):
                count[text_set[i][j]] += 1
                if count[text_set[i][j]] == 3:
                    self.unigram[text_set[i][j]] = index
                    index += 1
                else:
                    continue
              
                    
    def transform(self, text: list):
        """Transform a given sentence into vectors based on the extractor you got from self.fit()
        
        Arguments:
            text {list} -- a tokenized sentence (list of words), such as ["I", "love", "nlp"]
        
        Returns:
            array -- an unigram feature array, such as array([1,1,1,0,0,0])
        """
        feature = np.zeros(len(self.unigram))
        # feature[0] += 1
        for i in range(0, len(text)):
            if text[i] in self.unigram:
                feature[self.unigram[text[i]]] += 1
            else:
                feature[1] += 1
        
        self.num_tokens = np.sum(feature[1:])

        return feature
    
    
    def transform_list(self, text_set):
        # Add your code here!
        features = []
        for i in range(0, len(text_set)):
            features.append(self.transform(text_set[i]))
        
        return np.array(features)


    def token_log_probs(self, features, smoothing = True):
        # print(features.shape)
        # print(np.sum(features))
        prob = np.log(np.sum(features, axis = 0) + self.smoothing_alpha) - np.log((np.sum(features) + features.shape[0]) + self.smoothing_alpha * ((features.shape[1])+1))
        # print(prob)
        return prob

class BigramFeature(FeatureExtractor):
    """Example code for unigram feature extraction
    """
    def __init__(self, smoothing_alpha=0):
        self.unigram = {}
        self.smoothing_alpha = smoothing_alpha
        self.unigram_counter = np.array([])
        self.not_trained = True
        self.bigrams = defaultdict(self.zero)
        
    def zero(self):
        return 0
    
    def num_tokens(self):
        return np.sum(self.unigram_counter) - self.unigram_counter[0]
    
    def bigram_count(self, ind = None):
        if ind == None:
            return sum(self.bigrams.values())
        else:
            return self.bigrams[ind]
    
    
    def index(self, token):
        if token in self.unigram:
            return self.unigram[token]
        
        return self.unigram["<UNK>"]
        
    def fit(self, text_set: list):
        """Fit a feature extractor based on given data 
        
        Arguments:
            text_set {list} -- list of tokenized sentences and words are lowercased, such as [["I", "love", "nlp"], ["I", "like", "python"]]
        """
        self.unigram["<START>"] = 0
        self.unigram["<STOP>"] = 1
        self.unigram["<UNK>"] = 2
        index = 3
        count = defaultdict(self.zero)
        for i in range(0, len(text_set)):
            for j in range(0, len(text_set[i])):
                count[text_set[i][j]] += 1
                if count[text_set[i][j]] == 3:
                    self.unigram[text_set[i][j]] = index
                    index += 1
                else:
                    continue
        
        self.unigram_counter = np.zeros(len(self.unigram))
                    
    def transform(self, text: list):
        """Transform a given sentence into vectors based on the extractor you got from self.fit()
        
        Arguments:
            text {list} -- a tokenized sentence (list of words), such as ["I", "love", "nlp"]
        
        Returns:
            array -- an unigram feature array, such as array([1,1,1,0,0,0])
        """
        copied_text = text[:]
        word_count = len(self.unigram)
        copied_text.insert(0, "<START>")
        copied_text.append("<STOP>")
        
        feature = []
        self.unigram_counter[0] += 1
        # self.unigram_counter[self.index(copied_text[1])] += 1

        # print(self.unigram)

        for i in range(2, len(copied_text)):

            bigram_index = self.index(copied_text[i-1])*word_count + self.index(copied_text[i])

            # print(copied_text[i-1], copied_text[i], bigram_index)

            self.bigrams[bigram_index] += 1

            if self.not_trained:
                self.unigram_counter[self.index(copied_text[i-1])] += 1

            feature.append(bigram_index)
        # print()
        # print(self.bigrams)
            
        if self.not_trained:
            self.unigram_counter[1] += 1

        return np.array(feature)
    
    def transform_list(self, text_set):
        features = []
        for i in range(0, len(text_set)):
            features.append(self.transform(text_set[i]))
        
        return np.array(features)
    
    def token_log_probs(self, features, smoothing = False):
        probabilities = {}
        word_count = len(self.unigram)
        print(self.unigram_counter)
        for feature_vect in features:
            for index in feature_vect:
                if index != -1 and index not in probabilities:
                    bigram_count = self.bigram_count(index)
                    unigram_count = self.unigram_counter[index//word_count]
                    # print(index)
                    # print(bigram_count)
                    # print(unigram_count)
                    # print()
                    probabilities[index] = np.log(bigram_count + self.smoothing_alpha) - np.log(self.unigram_counter[index//word_count] + word_count*self.smoothing_alpha)

        # print(probabilities)
        # print()

        return probabilities


class TrigramFeature(FeatureExtractor):
    """Example code for unigram feature extraction
    """
    def __init__(self):
        self.unigram = {}
        self.smoothing_alpha = 1
        self.unigram_counter = np.array([])
        self.not_trained = True
        self.bigrams = defaultdict(self.zero)
        self.trigrams = defaultdict(self.zero)
        
    def zero(self):
        return 0
    
    def num_tokens(self):
        return np.sum(self.unigram_counter) - self.unigram_counter[0]
    
    def bigram_count(self, ind = None):
        if ind == None:
            return sum(self.bigrams.values())
        else:
            return self.bigrams[ind]
    
    def trigram_count(self, ind = None ):
        if ind == None:
            return sum(self.trigrams.values())
        else:
            return self.trigrams[ind]
    
    def index(self, token):
        if token in self.unigram:
            return self.unigram[token]
        
        return -1
        
    def fit(self, text_set: list):
        """Fit a feature extractor based on given data 
        
        Arguments:
            text_set {list} -- list of tokenized sentences and words are lowercased, such as [["I", "love", "nlp"], ["I", "like", "python"]]
        """
        self.unigram["<START>"] = 0
        self.unigram["<STOP>"] = 1
        self.unigram["<UNK>"] = 2
        index = 3
        count = defaultdict(self.zero)
        for i in range(0, len(text_set)):
            for j in range(0, len(text_set[i])):
                count[text_set[i][j]] += 1
                if count[text_set[i][j]] == 3:
                    self.unigram[text_set[i][j]] = index
                    index += 1
                else:
                    continue
        
        self.unigram_counter = np.zeros(len(self.unigram))
                    
    def transform(self, text: list):
        """Transform a given sentence into vectors based on the extractor you got from self.fit()
        
        Arguments:
            text {list} -- a tokenized sentence (list of words), such as ["I", "love", "nlp"]
        
        Returns:
            array -- an unigram feature array, such as array([1,1,1,0,0,0])
        """
        copied_text = text[:]
        word_count = len(self.unigram)
        copied_text.insert(0, "<START>")
        copied_text.append("<STOP>")

        
        
        if self.index(copied_text[1]) == -1:
            copied_text[1] = "<UNK>"
        feature = np.array([[-1, self.index(copied_text[1])]])
        self.unigram_counter[0] += 1
        self.unigram_counter[self.index(copied_text[1])] += 1

        for i in range(2, len(copied_text)):
            if self.index(copied_text[i]) == -1:
                copied_text[i] = "<UNK>"

            trigram_index = -1
            bigram_index = self.index(copied_text[i-2])*word_count + self.index(copied_text[i-1])
            trigram_index = self.index(copied_text[i-2])*word_count**2 + self.index(copied_text[i-1])*word_count + self.index(copied_text[i])

            if trigram_index != -1:
                self.trigrams[trigram_index] += 1
            self.bigrams[bigram_index] += 1

            self.unigram_counter[self.index(copied_text[i])] += 1

            feature = np.append(feature, [[trigram_index, bigram_index]], axis=0)

        
        return feature
    
    def transform_list(self, text_set):
        features = []
        for i in range(0, len(text_set)):
            features.append(self.transform(text_set[i]))
        
        return np.array(features)
    
    def token_log_probs(self, features, smoothing = False):
        probabilities = []
        word_count = len(self.unigram)
        for feature_vect in features:
            log_prob_sum = 0
            for indexes in feature_vect:
                if indexes[0] != -1:
                    trigram_count = self.trigram_count(indexes[0])
                    bigram_count = self.bigram_count(indexes[1])
                    if smoothing:
                        log_prob_sum += (np.log(trigram_count + 1) - np.log(bigram_count + word_count))
                    else:
                        log_prob_sum += (np.log(trigram_count) - np.log(bigram_count))
                    
                else:
                    bigram_count = self.bigram_count(indexes[1])
                    unigram_count = self.unigram_counter[0]
                    if smoothing:
                        log_prob_sum += (np.log(bigram_count + 1) - np.log(unigram_count + word_count))
                    else:
                        log_prob_sum += (np.log(bigram_count) - np.log(unigram_count))
            probabilities.append(log_prob_sum)
        return probabilities
                