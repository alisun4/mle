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
        # # print(len(feature_vect))
        features.append(feature_vect)
        # i += 1

    

    feat_extractor.not_trained = False
    

    return features


def perplexity(features, log_probs, args_feature, smoothing, feat_extractor = None):
    log_prob_sum = 0
    total_count = 0
    for feature_vect in features:
        if args_feature == "trigram":
            total_count += 1
            try:
                first_bigram_prob = feat_extractor.start_probs[feat_extractor.extract_bigram_index(feature_vect[0])]
            except KeyError:
                first_bigram_prob = feat_extractor.zero_prob_bigram(feat_extractor.extract_bigram_index(feature_vect[0]))
            # # print(first_bigram_prob)
            log_prob_sum -= first_bigram_prob
        # # print(feature_vect)
        for feature in feature_vect:
            # # print(log_probs[feature])
            try:
                log_prob_sum -= log_probs[feature]
            except KeyError:
                log_prob_sum -= feat_extractor.zero_prob(feature)
        total_count += len(feature_vect)
        # # print()
    # # print(log_prob_sum)
    # # print(total_count)
    # # print(log_prob_sum/total_count)
    return np.exp(log_prob_sum/total_count)



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
        feat_extractor = TrigramFeature(smoothing_alpha=args.smoothing)
    
    train = "train"
    test = args.test

        
    f = open(f"1b_benchmark.{train}.tokens", 'r', encoding="utf-8")
    train_set = f.readlines()
    f.close()
    
    training_data = []
    
    for text in train_set:
        training_data.append(tokenize(text))
    
    feat_extractor.fit(training_data)
    
    # print(len(feat_extractor.unigram))
    

    train_features = get_features(f"1b_benchmark.{train}.tokens", feat_extractor, args.feature)
    
    # # print(train_features)
    if args.feature == "bigram" or args.feature == "trigram":
        # print("Calculating bigram probabilities.")
        train_log_probs = feat_extractor.token_log_probs(train_features)
    else:
        train_log_probs = feat_extractor.token_log_probs(train_features)
        
    if args.debug:
        test_filename = "hdtv.txt"
    else:
        test_filename = f"1b_benchmark.{test}.tokens"
    
    test_features = get_features(test_filename, feat_extractor, args.feature)
    
    perp = perplexity(test_features, train_log_probs, args.feature, args.smoothing, feat_extractor)
    
    print(f"{args.feature} perplexity for {test}:", perp)
    
    

if __name__ == "__main__":
    main()