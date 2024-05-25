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

    feat_extractor.not_trained = False
    
    # # print(features[:5])

    return features

def perplexity(features, log_probs, args_feature, smoothing, feat_extractor = None):

    log_prob_sum = 0
    total_count = 0
    # print(len(features))
    for feature_vect in features:
        # print(len(feature_vect))
        if args_feature == "trigram" or args_feature == "interpolate":
            total_count += 1
            try:
                first_bigram_prob = feat_extractor.start_probs[feat_extractor.extract_bigram_index(feature_vect[0])]
            except(KeyError):
                first_bigram_prob = feat_extractor.zero_prob_bigram(feat_extractor.extract_bigram_index(feature_vect[0]))
            # # print(first_bigram_prob)
            log_prob_sum -= first_bigram_prob
        # print(feature_vect)
        for feature in feature_vect:
            try:
                # print(log_probs[feature])
                log_prob_sum -= log_probs[feature]
            except(KeyError):
                if args_feature == "interpolate":
                    log_prob_sum -= -np.inf
                else:
                    log_prob_sum -= feat_extractor.zero_prob(feature)
            
                
                
        total_count += len(feature_vect)
        # # print()
    # # print(log_prob_sum)
    # # print(total_count)
    # # print(log_prob_sum/total_count)
    return np.exp(log_prob_sum/total_count)


def linear_interpolation(trigram_features, lambdas, uni_log_probs, bi_log_probs, tri_log_probs, type_count):
    # assert (sum(lambdas) == 1, "Weights must sum to 1")
    print(f'\tInterpolating with lambdas: {lambdas}')
    
    i = 1
    
    interpolated_log_probs = {}

    for trigram_feature_list in tqdm(trigram_features):
        for trigram_feature in trigram_feature_list:
            bigram_feature = trigram_feature % (type_count**2)
            unigram_feature = bigram_feature % type_count
            
            trigram_log_prob = tri_log_probs[trigram_feature]
            try:
                bigram_log_prob = bi_log_probs[bigram_feature]
            except(KeyError):
                bigram_log_prob = -np.inf
            try:
                unigram_log_prob = uni_log_probs[unigram_feature]
            except(KeyError):
                unigram_log_prob = -np.inf
            
            if i == 39721:
                # print(bigram_feature)
                print(unigram_log_prob, bigram_log_prob, trigram_log_prob)
            
            interpolated_prob = unigram_log_prob*lambdas[0] + bigram_log_prob*lambdas[1] + trigram_log_prob*lambdas[2]
            interpolated_log_probs[trigram_feature] = interpolated_prob
        
        i += 1
    
    # Filter out -inf values
    # interpolated_log_probs = {k: v for k, v in interpolated_log_probs.items() if v != -np.inf}
    
    return interpolated_log_probs

def main():
    train_data = []
    test_data = 'hdtv.txt'
    # test_data = '1b_benchmark.dev.tokens'
    print(test_data)
    args = init_arg_parser()
    train, text = ("tiny", "tiny") if (args.debug) else ("train", args.test)
    
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
    
    with open(f"1b_benchmark.{train}.tokens", 'r', encoding="utf-8") as f:
        train_set = f.readlines()
    
    for text in train_set:
        train_data.append(tokenize(text))
        
    # Poor way to handle "this or that", but it works
    if args.feature == "interpolate":
        uni_feat_extractor.fit(train_data)
        bi_feat_extractor.fit(train_data)
        tri_feat_extractor.fit(train_data)
        
        type_count = len(uni_feat_extractor.unigram)
        
        uni_features = get_features(f"1b_benchmark.{train}.tokens", uni_feat_extractor, "unigram")
        bi_features = get_features(f"1b_benchmark.{train}.tokens", bi_feat_extractor, "bigram")
        tri_features = get_features(f"1b_benchmark.{train}.tokens", tri_feat_extractor, "trigram")
        
        uni_log_probs = uni_feat_extractor.token_log_probs(uni_features)
        bi_log_probs = bi_feat_extractor.token_log_probs(bi_features)
        tri_log_probs = tri_feat_extractor.token_log_probs(tri_features)
        
        # print(bi_feat_extractor.bigram_index(".", "<STOP>"))
        # print(bi_log_probs[bi_feat_extractor.bigram_index(".", "<STOP>")])
        
        interpolated_log_probs = linear_interpolation(tri_features, [0.1, 0.3, 0.6], uni_log_probs, bi_log_probs, tri_log_probs, type_count)
        
        test_tri_features = get_features(test_data, tri_feat_extractor, "trigram")
        
        perp = perplexity(test_tri_features, interpolated_log_probs, "interpolate", args.smoothing, tri_feat_extractor)
        print("Interpolated Trigram Perplexity: ", perp)
        
    else:  
        feat_extractor.fit(train_data)
        train_features = get_features(f"1b_benchmark.{train}.tokens", feat_extractor, args.feature)
        train_log_probs = feat_extractor.token_log_probs(train_features)
        test_features = get_features(test_data, feat_extractor, args.feature)
        perp = perplexity(test_features, train_log_probs, args.feature, args.smoothing, feat_extractor)
        print(perp)
    

if __name__ == "__main__":
    main()