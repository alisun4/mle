from utils import *
import numpy as np
import argparse
from tqdm import tqdm

def get_features(filename, feat_extractor, args_feature):
    f = open(filename, 'r', encoding="utf-8")
    text_set = f.readlines()
    f.close()
    features = []
    # i = 0
    for text in tqdm(text_set):
        feature_vect = feat_extractor.transform(tokenize(text))
        # print(len(feature_vect))
        features.append(feature_vect)
        # i += 1

    
    if args_feature == "bigram":
        feat_extractor.not_trained = False
    
    # print(features[:5])
    if args_feature == "bigram" or args_feature == "trigram":
        return features
    
    return np.array(features)

def perplexity(features, log_probs, args_feature, smoothing, feat_extractor = None):
    if args_feature != "unigram":
        log_prob_sum = 0
        total_count = 0
        for feature_vect in features:
            for feature in feature_vect:
                log_prob_sum -= log_probs[feature]
            total_count += len(feature_vect) + 1
        # total_count = feat_extractor.num_tokens()
        # print(log_prob_sum)
        # print(total_count)
        # print(log_prob_sum/total_count)
        return np.exp(log_prob_sum/total_count)
    
    # print(features)
    # print(np.dot(features, log_probs))
    # print(np.sum(features))
    # print(features.shape)

    s_log_prob = np.log(features.shape[0]) - np.log(np.sum(features) + 2*features.shape[0])

    return np.exp(-(np.sum(np.dot(features, log_probs)) + s_log_prob)/(np.sum(features) + 2*features.shape[0]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', '-f', type=str, default='unigram',
                        choices=['unigram', 'bigram', 'trigram'])
    parser.add_argument('--smoothing', '-s', type=float, default=0)
    parser.add_argument('--debug', '-d', type=bool, default=False)
    parser.add_argument('--test', '-t', type=str, default='train')
    args = parser.parse_args()
    if args.feature == "unigram":
        feat_extractor = UnigramFeature(smoothing_alpha=args.smoothing)
    elif args.feature == "bigram":
        feat_extractor = BigramFeature(smoothing_alpha=args.smoothing)
    elif args.feature == "trigram":
        feat_extractor = TrigramFeature()
    
    train = "train"
    test = args.test

    if args.debug:
        train = "tiny"
        test = "tiny"
        
    f = open(f"1b_benchmark.{train}.tokens", 'r', encoding="utf-8")
    train_set = f.readlines()
    f.close()
    
    training_data = []
    
    for text in train_set:
        training_data.append(tokenize(text))
    
    feat_extractor.fit(training_data)
    
    print(len(feat_extractor.unigram))

    train_features = get_features(f"1b_benchmark.{test}.tokens", feat_extractor, args.feature)
    
    # print(train_features)
    if args.feature == "bigram" or args.feature == "trigram":
        print("Calculating bigram probabilities.")
        train_log_probs = feat_extractor.token_log_probs(train_features)
    else:
        train_log_probs = feat_extractor.token_log_probs(train_features)
    
    perp = perplexity(train_features, train_log_probs, args.feature, args.smoothing, feat_extractor)
    
    print(perp)
    
    

if __name__ == "__main__":
    main()