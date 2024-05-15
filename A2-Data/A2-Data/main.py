from utils import *
import numpy as np
import argparse

def get_features(filename, feat_extractor):
    f = open(filename)
    text_set = f.readlines()
    f.close()
    
    features = []
    i = 0
    for text in text_set:
        feature_vect = feat_extractor.transform(tokenize(text))
        if len(feature_vect) > 23729:
            print(f"New words added on example {i}.")
            print(len(feature_vect))
            return np.array(features)
        features.append(feature_vect)
        i += 1
    
    # print(features[:5])
    
    return np.array(features)

def perplexity(features, probs):
    return np.exp(np.dot(features, np.transpose(probs)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', '-f', type=str, default='unigram',
                        choices=['unigram', 'bigram', 'trigram'])
    args = parser.parse_args()
    if args.feature == "unigram":
        feat_extractor = UnigramFeature()
    elif args.feature == "bigram":
        feat_extractor = BigramFeature()
        
    f = open("1b_benchmark.train.tokens")
    train_set = f.readlines()
    f.close()
    
    training_data = []
    
    for text in train_set:
        training_data.append(tokenize(text))
    
    feat_extractor.fit(training_data)
    
    # print(len(feat_extractor.unigram))

    train_features = get_features("1b_benchmark.train.tokens", feat_extractor)
    
    # print(train_features.shape)
    
    train_log_probs = feat_extractor.token_log_probs(train_features)
    
    perp = perplexity(train_features, train_log_probs)
    
    print(perp)
    
    

if __name__ == "__main__":
    main()