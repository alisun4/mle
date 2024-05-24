from utils import *
import numpy as np
import argparse
from tqdm import tqdm

def init_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', '-f', type=str, default='unigram',
                        choices=['unigram', 'bigram', 'trigram', 'interpolate'])
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


def linear_interpolation(trigram_features, lambdas, tri_log_probs, bi_log_probs, uni_log_probs):
    assert (sum(lambdas) == 1, "Weights must sum to 1")
    
    interpolated_log_probs = {}
    for p in tri_log_probs:
        bi_index = p // len(uni_log_probs)
        uni_index = bi_index //
    
    interpolated_log_probs = []
    interpolated_log_probs = [lambdas[0]*p1 + lambdas*p2 + lambdas*p3 for p1, p2, p3 in zip(uni_log_probs, bi_log_probs, tri_log_probs)]
    print(interpolated_log_probs)

def main():
    train_data = []
    # test_data = 'hdtv.txt'
    test_data = '1b_benchmark.dev.tokens'
    train, text = ("tiny", "tiny") if (args.debug) else ("train", args.test)
    args = init_arg_parser()
    
    if args.feature == "unigram":
        feat_extractor = UnigramFeature(smoothing_alpha=args.smoothing)
    elif args.feature == "bigram":
        feat_extractor = BigramFeature(smoothing_alpha=args.smoothing)
    elif args.feature == "trigram":
        feat_extractor = TrigramFeature(smoothing_alpha=args.smoothing)
    elif args.feature == "interpolate":
        uni_feat_extractor = UnigramFeature(smoothing_alpha=0)
        bi_feat_extractor = BigramFeature(smoothing_alpha=0)
        tri_feat_extractor = TrigramFeature(smoothing_alpha=0)
    
    
    if args.feature == "interpolate":
        
        uni_features = get_features(f"1b_benchmark.{train}.tokens", uni_feat_extractor, "unigram")
        bi_features = get_features(f"1b_benchmark.{train}.tokens", bi_feat_extractor, "bigram")
        tri_features = get_features(f"1b_benchmark.{train}.tokens", tri_feat_extractor, "trigram")
        
        uni_feat_extractor.fit(uni_features)
        bi_feat_extractor.fit(bi_features)
        tri_feat_extractor.fit(tri_features)
        
        uni_feat_extractor.transform(train_features)
        bi_feat_extractor.transform(train_features)
        tri_feat_extractor.transform(train_features)
        
        uni_log_probs = uni_feat_extractor.token_log_probs(uni_features)
        bi_log_probs = bi_feat_extractor.token_log_probs(bi_features)
        tri_log_probs = tri_feat_extractor.token_log_probs(tri_features)
        
        linear_interpolation(tri_features, [0.1, 0.3, 0.6], tri_log_probs, bi_log_probs, uni_log_probs)
        
    else:
        with open(f"1b_benchmark.{train}.tokens", 'r', encoding="utf-8") as f:
            train_set = f.readlines()
        
        for text in train_set:
            train_data.append(tokenize(text))
            
        train_features = get_features(f"1b_benchmark.{train}.tokens", feat_extractor, args.feature)
        feat_extractor.fit(train_data)
        train_log_probs = feat_extractor.token_log_probs(train_features)
        test_features = get_features(test_data, feat_extractor, args.feature)
        perp = perplexity(test_features, train_log_probs, args.feature, args.smoothing, feat_extractor)
        print(perp)
    

if __name__ == "__main__":
    main()