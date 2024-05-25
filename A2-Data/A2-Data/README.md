# CSE 143 (Winter 2020) HW 1 data

This folder contains 3 files, a subset of the 1 Billion Word Benchmark's
heldout set.

Specifically, `1b_benchmark.train.tokens` is taken from sections 0-9,
`1b_benchmark.dev.tokens` is taken from sections 10 and 11, and
`1b_benchmark.test.tokens` is taken from sections 12 and 13.

To recreate this data (download the raw 1 Billion Word Benchmark and generate the split), run:

```
./subsample_1b_benchmark.sh
```

# Command Line Arguments
The script accepts the following command line arguments:

`--feature` or `-f`: This argument specifies the type of feature to use. The options are 'unigram', 'bigram', 'trigram', and 'interpolate'. The default value is 'unigram'.

`--smoothing` or `-s`: This argument is used to specify the smoothing value. It should be a float. The default value is 0.

`--debug` or `-d`: This argument is a boolean flag used to turn on or off the debug mode. The default value is False.

`--test` or `-t`: This argument is used to specify the test set. It should be a string. The default value is 'train'.

`--interpolate` or `-i`: This argument is a boolean flag used to turn on or off the interpolation. The default value is False.

Usage
Here is an example of how to use these arguments:
```
python main.py --feature bigram --smoothing 0.1 --debug True --test test --interpolate True
```
