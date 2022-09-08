#!/usr/bin/env python
# Copyright (c) 2022 David A. Randolph.
# Copyright (c) 2015 Seong-Jin Kim.
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
# Some methods and program structure inspired by (borrowed from)
# Seong-Jin Kim's crf package (https://github.com/lancifollia/crf).
# Many thanks.
import argparse
import math
from datetime import datetime
import itertools
from sklearn.model_selection import KFold, train_test_split

from xycrf import XyCrf, START_TAG, STOP_TAG
from util.corpus import read_corpus, split_corpus

HYPHEN_TAG = '-'
NO_HYPHEN_TAG = '*'
SUFFIX = 'suffix'
PREFIX = 'prefix'
SPLIT = 'split'
BOUND = 'bound'

VOWELS = [x for x in "aeiou"]
IS_VOWEL = dict.fromkeys(VOWELS, True)

ngram_counts = dict()
ngram_sums = dict()
ngram_norms = dict()


def normalize(counts):
    """
    Normalize n-gram counts to mean 0 and variance 1.
    :param counts: Dictionary mapping n-gram tuple keys to their counts.
    :return: A dictionary of normalized values.
    """
    norms = dict()
    ngram_count = len(counts)
    count_sum = 0
    for gram in counts:
        count_sum += counts[gram]

    avg = count_sum / ngram_count
    sum_of_squared_deviations = 0
    for gram in counts:
        gram_count = counts[gram]
        deviation = gram_count - avg
        sum_of_squared_deviations += deviation**2
    variance = sum_of_squared_deviations / len(counts)
    std_deviation = math.sqrt(variance)
    for gram in counts:
        count = counts[gram]
        norm = (count - avg) / std_deviation
        norms[gram] = norm
    return norms


def normalize_simple(counts):
    """
    Normalize counts to scores between 0 and 1.
    :param counts: Dictionary mapping n-gram tuple keys to their counts.
    :return: Dictionary of normalized scores for each ngram.
    """
    norms = dict()
    ngram_count = len(counts)
    largest_count = 0
    for gram in counts:
        if counts[gram] > largest_count:
            largest_count = counts[gram]

    normalizer = largest_count * largest_count
    for gram in counts:
        normed_count = counts[gram] * largest_count
        norm = normed_count / normalizer
        norms[gram] = norm
    return norms


def load_ngram_dicts(n, training_data, fix, hyphenated):
    if fix not in ngram_counts:
        ngram_counts[fix] = dict()
        ngram_norms[fix] = dict()
        ngram_sums[fix] = dict()
    if hyphenated not in ngram_counts[fix]:
        ngram_counts[fix][hyphenated] = dict()
        ngram_norms[fix][hyphenated] = dict()
        ngram_sums[fix][hyphenated] = dict()
    if n not in ngram_counts[fix][hyphenated]:
        ngram_counts[fix][hyphenated][n] = dict()
        ngram_sums[fix][hyphenated][n] = 0
        ngram_norms[fix][hyphenated][n] = dict()

    counts = ngram_counts[fix][hyphenated]
    sums = ngram_sums[fix][hyphenated]
    norms = ngram_norms[fix][hyphenated]

    for example in training_data:
        x_bar = list(itertools.chain(*example[0]))
        y_bar = list(itertools.chain(example[1]))

        if fix == SUFFIX:
            for i in range(1, len(y_bar)):
                if i + n > len(y_bar):
                    continue
                if hyphenated == SPLIT:
                    if y_bar[i-1] != HYPHEN_TAG:
                        continue
                else:
                    if y_bar[i-1] == HYPHEN_TAG:
                        continue
                tuple_start = i
                tuple_end = i + n
                new_gram = tuple(x_bar[tuple_start:tuple_end])
                if new_gram not in counts[n]:
                    counts[n][new_gram] = 0
                counts[n][new_gram] += 1
                sums[n] += 1
        else:
            for i in range(n - 1, len(y_bar)):
                if i - n + 1 < 0:
                    continue
                if hyphenated == SPLIT:
                    if y_bar[i-1] != HYPHEN_TAG:
                        continue
                else:
                    if y_bar[i-1] == HYPHEN_TAG:
                        continue
                tuple_start = i - n + 1
                tuple_end = i + 1
                new_gram = tuple(x_bar[tuple_start:tuple_end])
                if new_gram not in counts[n]:
                    counts[n][new_gram] = 0
                counts[n][new_gram] += 1
                sums[n] += 1

    # Normalize function return values.
    # norms[n] = normalize(counts[n])
    # norms[n] = normalize_simple(counts[n])


def load_all_n_gram_dicts(training_data, ns):
    print("\nLoading n-grams...")
    for n in ns:
        for fix in (PREFIX, SUFFIX):
            for hyphenate in (SPLIT, BOUND):
                load_ngram_dicts(n, training_data, fix, hyphenate)
    print("N-grams loaded.")


def get_ngram_count(x_bar, i, n, fix, hyphenated):
    flat_x_bar = list(itertools.chain(*x_bar))
    count = 0
    if fix == SUFFIX:
        if 0 < i < len(x_bar) + n - 2:
            start = i
            end = i + n
            gram = tuple(flat_x_bar[start:end])
            if gram in ngram_counts[fix][hyphenated][n]:
                count = ngram_counts[fix][hyphenated][n][gram]
    else:
        if i - n + 1 >= 0 and i + 1 < len(flat_x_bar):
            start = i - n + 1
            end = i + 1
            gram = tuple(flat_x_bar[start:end])
            if gram in ngram_counts[fix][hyphenated][n]:
                count = ngram_counts[fix][hyphenated][n][gram]
    return count


def get_count_ratio(x_bar, i, n, fix):
    split_count = get_ngram_count(x_bar=x_bar, i=i, n=n, fix=fix, hyphenated=SPLIT)
    if split_count == 0:
        return 0
    bound_count = get_ngram_count(x_bar=x_bar, i=i, n=n, fix=fix, hyphenated=BOUND)
    value = split_count / (split_count + bound_count)
    return value


def combined_ngram_func_factory(n, fix):
    def f(y_prev, y, x_bar, i):
        if i == 0:
            return 0
        if y_prev != HYPHEN_TAG:
            return 0
        if y == HYPHEN_TAG:
            return 0
        if y == STOP_TAG and i != len(x_bar) - 1:
            return 0
        if y != STOP_TAG and i == len(x_bar) - 1:
            return 0

        value = get_count_ratio(x_bar=x_bar, i=i, n=n, fix=fix)
        return value

    return f


def double_consonant(y_prev, y, x_bar, i):
    if 1 < i < len(x_bar) - 1:
        ch = x_bar[i][0]
        ch_prev = x_bar[i-1][0]
        if ch_prev == ch and ch_prev not in IS_VOWEL and ch not in IS_VOWEL and y_prev == HYPHEN_TAG:
            return 1
    return 0


def one_letter_syllable(y_prev, y, x_bar, i):
    # One-letter syllables are (almost?) always vowels.
    last_index = len(x_bar) - 1
    ch = x_bar[i][0]
    if last_index > i > 1 and y_prev == HYPHEN_TAG and ch not in IS_VOWEL and y == HYPHEN_TAG:
        return 1
    return 0


def leading_one_letter_syllable(y_prev, y, x_bar, i):
    # One-letter syllables are (almost?) always vowels.
    ch = x_bar[i][0]
    if i == 1 and ch not in IS_VOWEL and y == HYPHEN_TAG:
        return 1
    return 0


def valid_internal(y_prev, y, x_bar, i):
    if 0 < i < len(x_bar) - 1 and y in (HYPHEN_TAG, NO_HYPHEN_TAG):
        return 1
    return 0


def valid_start(y_prev, y, x_bar, i):
    if i == 1 and y_prev == START_TAG:
        return 1
    return 0


def valid_stop(y_prev, y, x_bar, i):
    if i == len(x_bar) - 1 and y == STOP_TAG:
        return 1
    return 0


def add_functions(xycrf, training_data, ns=(2, 3, 4, 5)):
    load_all_n_gram_dicts(training_data=training_data, ns=ns)
    print("Adding feature functions...")

    # xycrf.add_feature_function(func=valid_start, name="valid_start")
    # xycrf.add_feature_function(func=valid_stop, name="valid_stop")
    xycrf.add_feature_function(func=valid_internal, name="valid_internal")
    xycrf.add_feature_function(func=double_consonant, name="double_consonant")
    xycrf.add_feature_function(func=one_letter_syllable, name="one_letter_syllable")
    xycrf.add_feature_function(func=leading_one_letter_syllable, name="leading_one_letter_syllable")

    for n in ns:
        for fix in (SUFFIX, PREFIX):
            name = f'{n}-gram_{fix}'
            func = combined_ngram_func_factory(n=n, fix=fix)
            xycrf.add_feature_function(func=func, name=name)
    print(f"{xycrf.feature_count} feature functions added")


def train_from_file(xycrf, corpus_path, model_path,
                    epochs, learning_rate, attenuation,
                    test_size=0.0, seed=None, ns=(2, 3, 4, 5)):
    """
    Estimates parameters using stochastic gradient ascent.
    """
    start_dt = datetime.now()
    print('[%s] Start training' % start_dt)

    # Read the training corpus
    print("* Reading training data... ", end="")
    if test_size == 0 or seed is None:
        training_data, tag_set, _ = read_corpus(corpus_path, ns=ns)
    else:
        splits = split_corpus(file_path=corpus_path, ns=ns, seed=seed, test_size=test_size)
        training_data = splits['train']['data']
        tag_set = splits['train']['tag_set']

    xycrf.set_tags(tag_set=tag_set)
    add_functions(xycrf=xycrf, training_data=training_data, ns=ns)
    xycrf.training_data = training_data
    print("Done reading data")

    print("* Number of tags: {}".format(xycrf.tag_count))
    print("* Number of features: {}".format(xycrf.feature_count))
    print("* Number of training examples: {}".format(len(training_data)))

    # gradient, big_z = xycrf.gradient_for_all_training()
    xycrf.stochastic_gradient_ascent_train(epochs=epochs, learning_rate=learning_rate, attenuation=attenuation)
    print(xycrf.weights)
    # Estimates parameters to maximize log-likelihood of the corpus.
    # xycrf.train()

    xycrf.pickle(model_path)

    end_dt = datetime.now()
    execution_duration_minutes = (end_dt - start_dt)
    print("Total training time (wall clock): {}".format(execution_duration_minutes))
    print('* [%s] Training done' % end_dt)

    return xycrf.weights


def test_from_file(xycrf, corpus_path, test_size=0.0, seed=None, ns=(2, 3, 4, 5)):
    if test_size == 0:
        test_data, tag_set, _ = read_corpus(corpus_path, ns=ns)
    else:
        splits = split_corpus(file_path=corpus_path, ns=ns, seed=seed, test_size=test_size)
        test_data = splits['test']['data']
        tag_set = splits['test']['tag_set']

    total_count = 0
    correct_count = 0
    for x_bar, y_bar in test_data:
        y_hat = xycrf.infer(x_bar)
        flat_x_bar = list(itertools.chain(*x_bar))
        print('WORD:      ', end='')
        print("".join(flat_x_bar))
        print('PREDICTED: ', end='')
        print("".join(y_hat))
        print('CORRECT:   ', end='')
        print("".join(y_bar))
        print("")
        for i in range(1, len(y_bar) - 1):
            total_count += 1
            if y_bar[i] == y_hat[i]:
                correct_count += 1

    print('Correct: %d' % correct_count)
    print('Total: %d' % total_count)
    print('Performance: %f' % (correct_count / total_count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Data file for training input.")
    parser.add_argument("--output", help="Path of model pickle file name to output.")
    parser.add_argument("--method", help="One of sga or lbfgs. Default: sga.", choices=['lbfgs', 'sga'])
    parser.add_argument("--rate", help="Learning rate for sga (stochastic gradient ascent) method. " +
                        "Default: 1.", type=float)
    parser.add_argument("--epochs", help="Number of epochs (runs over all training) for sga method. " +
                        "Default: 3.", type=int)
    parser.add_argument("--attenuation", help="Rate multiplier for subsequent epochs of sga training. " +
                        "Default: 0.1.", type=float)
    parser.add_argument("--test", help="Data file for evaluation.")
    parser.add_argument("--input", help="Model pickle file path.")

    parser.add_argument("--seed", help="Integer to seed randomization of input data." +
                        "Default: 1965", type=int)
    parser.add_argument("--test_size", help="Proportion (0.00 - 1.00) of records to include in test set." +
                        "Default: 0.0", type=float)
    # parser.add_argument("--validation_size", help="Proportion (0.00 - 1.00) of records to include in validation set." +
                        # "Default: 0.0", type=float)
    # parser.add_argument("--k_folds", help="Number of folds to include for cross-validation." +
                        # "Default: 0.0", type=int)
    # ns = (2,)
    ns = (2, 3, 4, 5)
    args = parser.parse_args()

    if args.train:
        epochs = 1
        learning_rate = 0.01
        attenuation = 1
        if args.epochs:
            epochs = args.epochs
        if args.rate:
            learning_rate = args.rate
        if args.attenuation:
            attenuation = args.attenuation
        crf = XyCrf(optimize=False)
        if args.test_size:
            if not args.seed:
                seed = 1066
            else:
                seed = args.seed
            serial_weights = train_from_file(xycrf=crf, corpus_path=args.train, model_path=args.output,
                                             epochs=epochs, learning_rate=learning_rate, attenuation=attenuation,
                                             test_size=args.test_size, seed=seed,
                                             ns=ns)
        else:
            serial_weights = train_from_file(xycrf=crf, corpus_path=args.train, model_path=args.output,
                                             epochs=epochs, learning_rate=learning_rate, attenuation=attenuation,
                                             ns=ns)
        # crf = XyCrf(optimize=False)
        # parallel_weights = train_from_file(xycrf=crf, corpus_path=args.train, model_path=args.output)
        # for i in range(len(serial_weights)):
            # if serial_weights[i] != parallel_weights[i]:
                # print("Weights are DIFFERENT.")
                # break
        # print("Weights are the SAME.")
    if args.test:
        crf = XyCrf.unpickle(args.input)
        if args.test_size:
            if not args.seed:
                seed = 1066
            else:
                seed = args.seed
            test_from_file(xycrf=crf, corpus_path=args.test, test_size=args.test_size, seed=seed, ns=ns)
        else:
            test_from_file(xycrf=crf, corpus_path=args.test, ns=ns)

    exit(0)
