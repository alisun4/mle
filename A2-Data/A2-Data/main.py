from utils import *
import numpy as np
import argparse
from tqdm import tqdm

def get_features(filename, feat_extractor, args_feature):
    f = open(filename, 'r', encoding="utf-8")
    text_set = f.readlines()
    f.close()
    features = []
    i = 0
    for text in tqdm(text_set):
        feature_vect = feat_extractor.transform(tokenize(text))
        # print(len(feature_vect))
        features.append(feature_vect)
        i += 1
    
    # print(features[:5])
    if args_feature == "bigram" or args_feature == "trigram":
        return features
    
    return np.array(features)

def perplexity(features, log_probs, args_feature, smoothing,feat_extractor = None):
    if args_feature == "bigram":
        log_prob_sum = 0
        bigram_count = 0
        for feature_vect in tqdm(features):
            log_prob_sum -= np.sum(log_probs[feature_vect[:,0]]*feature_vect[:,1])
            bigram_count += np.sum(feature_vect[:,1])
        return np.exp(log_prob_sum/bigram_count)
    
    if args_feature == "trigram":
        log_prob_sum = 0
        log_prob_sum -= sum(log_probs)
        total_count = feat_extractor.num_tokens()
        return np.exp(log_prob_sum/total_count)

    dims = features.shape
    dim1 = 1
    for i in range(1, len(dims)):
        dim1 *= dims[i]
    return np.exp(-np.sum(np.dot(features.reshape(dims[0], dim1), log_probs))/np.sum(features))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', '-f', type=str, default='unigram',
                        choices=['unigram', 'bigram', 'trigram'])
    parser.add_argument('--smoothing', '-s', type=bool, default=False)
    args = parser.parse_args()
    if args.feature == "unigram":
        feat_extractor = UnigramFeature()
    elif args.feature == "bigram":
        feat_extractor = BigramFeature()
    elif args.feature == "trigram":
        feat_extractor = TrigramFeature()
        
    f = open("1b_benchmark.train.tokens", 'r', encoding="utf-8")
    train_set = f.readlines()
    f.close()
    
    training_data = []
    
    for text in train_set:
        training_data.append(tokenize(text))
    
    feat_extractor.fit(training_data)
    
    print(len(feat_extractor.unigram))

    train_features = get_features("1b_benchmark.train.tokens", feat_extractor, args.feature)
    
    # print(train_features)
    if args.feature == "bigram" or args.feature == "trigram":
        print("Calculating bigram probabilities.")
        train_log_probs = feat_extractor.token_log_probs(train_features)
    else:
        train_log_probs = feat_extractor.token_log_probs(train_features)
    
    if args.feature != "trigram":
        perp = perplexity(train_features, train_log_probs, args.feature, args.smoothing)
    else:
        perp = perplexity(train_features, train_log_probs, args.feature, args.smoothing, feat_extractor)
    
    print(perp)
    
    

if __name__ == "__main__":
    main()