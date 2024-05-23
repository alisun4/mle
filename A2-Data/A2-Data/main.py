from utils import *
import numpy as np
import argparse
from tqdm import tqdm

def init_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', '-f', type=str, default='unigram',
                        choices=['unigram', 'bigram', 'trigram'])
    parser.add_argument('--smoothing', '-s', type=float, default=0)
    parser.add_argument('--debug', '-d', type=bool, default=False)
    parser.add_argument('--test', '-t', type=str, default='train')
    parser.add_argument('--interpolate', '-i', type=bool, default=False)
    return parser.parse_args()
    
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

    if args_feature == "bigram":
        feat_extractor.not_trained = False
    
    # # print(features[:5])
    if args_feature == "bigram" or args_feature == "trigram":
        return features
    
    return np.array(features)

def perplexity(features, log_probs, args_feature, smoothing, feat_extractor = None):
    if args_feature != "unigram":
        log_prob_sum = 0
        total_count = 0
        for feature_vect in features:
            if args_feature == "trigram":
                total_count += 1
                first_bigram_prob = feat_extractor.start_probs[feat_extractor.extract_bigram_index(feature_vect[0])]
                # # print(first_bigram_prob)
                log_prob_sum -= first_bigram_prob
            # # print(feature_vect)
            for feature in feature_vect:
                # # print(log_probs[feature])
                try:
                    log_prob_sum -= log_probs[feature]
                except(KeyError):
                    pass
                
                    
                    
            total_count += len(feature_vect)
            # # print()
        # # print(log_prob_sum)
        # # print(total_count)
        # # print(log_prob_sum/total_count)
        return np.exp(log_prob_sum/total_count)
    
    # # print(features)
    # # print(np.dot(features, log_probs))
    # # print(np.sum(features))
    # # print(features.shape)

    # # print(s_log_prob)

    log_prob_sum = np.sum(np.dot(features, log_probs))

    if log_prob_sum > 0:
        return "inf"

    return np.exp(-(log_prob_sum)/(np.sum(features)))

def main():
    training_data = []
    args = init_arg_parser()
    
    if args.feature == "unigram":
        feat_extractor = UnigramFeature(smoothing_alpha=args.smoothing)
    elif args.feature == "bigram":
        feat_extractor = BigramFeature(smoothing_alpha=args.smoothing)
    elif args.feature == "trigram":
        feat_extractor = TrigramFeature(smoothing_alpha=args.smoothing)
    
    train, text = ("tiny", "tiny") if (args.debug) else ("train", args.test)
    
    with open(f"1b_benchmark.{train}.tokens", 'r', encoding="utf-8") as f:
        train_set = f.readlines()
    
    for text in train_set:
        training_data.append(tokenize(text))
        
    feat_extractor.fit(training_data)
    train_features = get_features(f"1b_benchmark.{train}.tokens", feat_extractor, args.feature)
    train_log_probs = feat_extractor.token_log_probs(train_features)
    test_features = get_features(f"hdtv.txt", feat_extractor, args.feature)
    perp = perplexity(test_features, train_log_probs, args.feature, args.smoothing, feat_extractor)
    print(perp) 
    

if __name__ == "__main__":
    main()