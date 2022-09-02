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

from xycrf import XyCrf
from util.corpus import read_corpus, split_corpus

ngram_counts = dict()
ngram_norms = dict()
ngram_sums = dict()


def ngram_func_factory(n, training_data, suffixing: bool, hyphenated: bool):
    if suffixing not in ngram_counts:
        ngram_counts[suffixing] = dict()
        ngram_norms[suffixing] = dict()
        ngram_sums[suffixing] = dict()
    if hyphenated not in ngram_counts[suffixing]:
        ngram_counts[suffixing][hyphenated] = dict()
        ngram_norms[suffixing][hyphenated] = dict()
        ngram_sums[suffixing][hyphenated] = dict()
    if n not in ngram_counts[suffixing][hyphenated]:
        ngram_counts[suffixing][hyphenated][n] = dict()
        ngram_sums[suffixing][hyphenated][n] = 0
        ngram_norms[suffixing][hyphenated][n] = dict()

    counts = ngram_counts[suffixing][hyphenated]
    sums = ngram_sums[suffixing][hyphenated]
    norms = ngram_norms[suffixing][hyphenated]

    for example in training_data:
        x_bar = list(itertools.chain(*example[0]))
        y_bar = list(itertools.chain(example[1]))

        if suffixing:
            for i in range(1, len(y_bar)):
                if i + n > len(y_bar):
                    continue
                if hyphenated:
                    if y_bar[i-1] != '-':
                        continue
                else:
                    if y_bar[i-1] == '-':
                        continue
                tuple_start = i
                tuple_end = i + n
                new_gram = tuple(x_bar[tuple_start:tuple_end])
                if new_gram not in counts[n]:
                    counts[n][new_gram] = 0
                counts[n][new_gram] += 1
                sums[n] += 1
        else:
            for i in range(n - 1, len(y_bar) - 1):
                if i - n < 0:
                    continue
                if hyphenated:
                    if y_bar[i-1] != '-':
                        continue
                else:
                    if y_bar[i-1] == '-':
                        continue
                tuple_start = i - n
                tuple_end = i + 1
                new_gram = tuple(x_bar[tuple_start:tuple_end])
                if new_gram not in counts[n]:
                    counts[n][new_gram] = 0
                counts[n][new_gram] += 1
                sums[n] += 1

    # Normalize function return values.
    ngram_sum = sums[n]
    ngram_count = len(counts[n])
    avg = ngram_sum / ngram_count
    sum_of_squared_deviations = 0
    for gram in counts[n]:
        gram_count = counts[n][gram]
        deviation = gram_count - avg
        sum_of_squared_deviations += deviation**2
    variance = sum_of_squared_deviations / len(counts)
    std_deviation = math.sqrt(variance)
    for gram in counts[n]:
        count = counts[n][gram]
        norm = (count - avg) / std_deviation
        norms[n][gram] = norm

    def f(y_prev, y, x_bar, i):
        if y_prev != '-':
            return 0
        if y == '-':
            return 0

        if y == 'START':
            if i == 0:
                return 1
            else:
                return 0
        if y == 'STOP':
            if i == len(x_bar) - 1:
                return 1
            else:
                return 0

        flat_x_bar = list(itertools.chain(*x_bar))
        if suffixing:
            if 0 < i < len(x_bar) + n - 2:
                start = i
                end = i + n
                gram = tuple(flat_x_bar[start:end])
                if gram in ngram_norms[n]:
                    # return ngram_counts[n][gram]
                    norm = ngram_norms[n][gram]
                    return norm
        else:
            if i - n + 1 >= 0 and i + 1 < len(flat_x_bar):
                start = i - n + 1
                end = i + 1
                gram = tuple(flat_x_bar[start:end])
                if gram in ngram_norms[n]:
                    # return ngram_counts[n][gram]
                    norm = ngram_norms[n][gram]
                    return norm
        return 0

    return f


def add_functions(xycrf, training_data, ns=(2,3,4,5)):
    for n in ns:
        for suffixing in (True, False):
            direction = 'prefix'
            if suffixing:
                direction = 'suffix'
            for hyphenated in (True, False):
                name = f'{n}-gram_{direction}_{hyphenated}'
                func = ngram_func_factory(n=n, training_data=training_data, suffixing=suffixing, hyphenated=hyphenated)
                xycrf.add_feature_function(func=func, name=name)


def train_from_file(xycrf, corpus_path, model_path,
                    epochs, learning_rate, attenuation,
                    test_size=0.0, seed=None, ns=(2, 3, 4, 5)):
    """
    Estimates parameters using conjugate gradient methods.(L-BFGS-B used)
    """
    start_dt = datetime.now()
    print('[%s] Start training' % start_dt)

    # Read the training corpus
    print("* Reading training data ... ", end="")
    if test_size == 0 or seed is None:
        training_data, tag_set, _ = read_corpus(corpus_path, ns=ns)
    else:
        splits = split_corpus(file_path=corpus_path, ns=ns, seed=seed, test_size=test_size)
        training_data = splits['train']['data']
        tag_set = splits['train']['tag_set']

    xycrf.set_tags(tag_list=list(tag_set))
    add_functions(xycrf=xycrf, training_data=training_data)
    xycrf.training_data = training_data
    print("Done reading data")

    print("* Number of labels: {}".format(xycrf.tag_count))
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
        y_bar_pred = xycrf.infer(x_bar)
        print(x_bar)
        print(y_bar_pred)
        for i in range(len(y_bar)):
            total_count += 1
            if y_bar[i] == y_bar_pred[i]:
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
    ns = (2,)
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
