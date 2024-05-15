from utils import *
import numpy as np
import argparse
from tqdm import tqdm

def get_features(filename, feat_extractor):
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
    
    return np.array(features)

def perplexity(features, probs):
    dims = features.shape
    dim1 = 1
    for i in range(1, len(dims)):
        dim1 *= dims[i]
    return np.exp(np.dot(features.reshape(dims[0], dim1), probs))

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
        
    f = open("1b_benchmark.train.tokens", 'r', encoding="utf-8")
    train_set = f.readlines()
    f.close()
    
    training_data = []
    
    for text in train_set:
        training_data.append(tokenize(text))
    
    feat_extractor.fit(training_data)
    
    print(len(feat_extractor.unigram))

    train_features = get_features("1b_benchmark.train.tokens", feat_extractor)
    
    print(train_features.shape)
    
    train_log_probs = np.transpose(np.ravel(feat_extractor.token_log_probs(train_features)))
    
    perp = perplexity(train_features, train_log_probs)
    
    print(perp)
    
    

if __name__ == "__main__":
    main()