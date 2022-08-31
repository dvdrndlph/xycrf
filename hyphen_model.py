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
from util.corpus import read_corpus

ngram_counts_before = dict()
ngram_counts_after = dict()
ngram_norms_before = dict()
ngram_norms_after = dict()
ngram_sums_before = dict()
ngram_sums_after = dict()


def ngram_func_factory(n, training_data, look_after):
    if look_after:
        ngram_counts = ngram_counts_after
        ngram_sums = ngram_sums_after
        ngram_norms = ngram_norms_after
    else:
        ngram_counts = ngram_counts_before
        ngram_sums = ngram_sums_before
        ngram_norms = ngram_norms_before
    if n not in ngram_counts:
        ngram_counts[n] = dict()
        ngram_sums[n] = 0
        ngram_norms[n] = dict()

    for example in training_data:
        x_bar = list(itertools.chain(*example[0]))
        y_bar = list(itertools.chain(example[1]))

        if look_after:
            for i in range(1, len(y_bar)):
                if i + n > len(y_bar) - 1:
                    continue
                if y_bar[i-1] != '-':
                    continue
                tuple_start = i
                tuple_end = i + n
                new_gram = tuple(x_bar[tuple_start:tuple_end])
                if "" in new_gram:
                    raise Exception(f'No blanks allowed after: {new_gram}.')
                if new_gram not in ngram_counts[n]:
                    ngram_counts[n][new_gram] = 0
                ngram_counts[n][new_gram] += 1
                ngram_sums[n] += 1
        else:
            for i in range(n - 1, len(y_bar) - 1):
                if i - n < 0:
                    continue
                if y_bar[i-1] != '-':
                    continue
                tuple_start = i - n + 1
                tuple_end = i + 1
                new_gram = tuple(x_bar[tuple_start:tuple_end])
                if "" in new_gram:
                    raise Exception(f'No blanks allowed before: {new_gram}.')
                if new_gram not in ngram_counts[n]:
                    ngram_counts[n][new_gram] = 0
                ngram_counts[n][new_gram] += 1
                ngram_sums[n] += 1

    # Normalize function return values.
    ngram_sum = ngram_sums[n]
    ngram_count = len(ngram_counts[n])
    avg = ngram_sum / ngram_count
    sum_of_squared_deviations = 0
    for gram in ngram_counts[n]:
        gram_count = ngram_counts[n][gram]
        deviation = gram_count - avg
        sum_of_squared_deviations += deviation**2
    variance = sum_of_squared_deviations / len(ngram_counts)
    std_deviation = math.sqrt(variance)
    for gram in ngram_counts[n]:
        count = ngram_counts[n][gram]
        norm = (count - avg) / std_deviation
        ngram_norms[n][gram] = norm

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
        if look_after:
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
        for looking_after in (True, False):
            direction = 'before'
            if looking_after:
                direction = 'after'
            name = "{}-gram_{}".format(n, direction)
            func = ngram_func_factory(n=n, training_data=training_data, look_after=looking_after)
            xycrf.add_feature_function(func=func, name=name)


def train_from_file(xycrf, corpus_path, model_path, epochs, learning_rate, attenuation):
    """
    Estimates parameters using conjugate gradient methods.(L-BFGS-B used)
    """
    start_dt = datetime.now()
    print('[%s] Start training' % start_dt)

    # Read the training corpus
    print("* Reading training data ... ", end="")
    training_data, tag_set, ngram_sets = read_corpus(corpus_path, ns=[2,3,4,5])
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


def test_from_file(xycrf, corpus_path):
    test_data, tag_set, ngram_sets = XyCrf.read_corpus(corpus_path, ns=[1, 2, 3])
    total_count = 0
    correct_count = 0
    for x_bar, y_bar in test_data:
        y_bar_pred = xycrf.infer(x_bar)
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

    parser.add_argument("--test_size", help="Number of records to include in test set.", type=int)
    parser.add_argument("--holdout", help="Number of records to exclude from usage. " +
                        "Default: 0.", type=int)
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
        # crf = XyCrf(optimize=False)
        serial_weights = train_from_file(xycrf=crf, corpus_path=args.train, model_path=args.output,
                                         epochs=epochs, learning_rate=learning_rate, attenuation=attenuation)
        # parallel_weights = train_from_file(xycrf=crf, corpus_path=args.train, model_path=args.output)
        # for i in range(len(serial_weights)):
            # if serial_weights[i] != parallel_weights[i]:
                # print("Weights are DIFFERENT.")
                # break
        # print("Weights are the SAME.")
    if args.test:
        crf = XyCrf.unpickle(args.input)
        test_from_file(xycrf=crf, corpus_path=args.test)

    exit(0)
