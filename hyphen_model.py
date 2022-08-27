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
from datetime import datetime
import itertools

import numpy as np
from nltk import ngrams

from xycrf import XyCrf

ngram_counts_after = dict()
ngram_counts_before = dict()


def ngram_func_factory(n, training_data, look_after):
    if look_after:
        ngram_counts = ngram_counts_after
    else:
        ngram_counts = ngram_counts_before
    if n not in ngram_counts:
        ngram_counts[n] = dict()

    for example in training_data:
        x_bar = list(itertools.chain(*example[0]))
        y_bar = list(itertools.chain(example[1]))

        if look_after:
            for i in range(1, len(y_bar)):
                if i + n > len(y_bar):
                    continue
                if y_bar[i-1] != '-':
                    continue
                tuple_start = i
                tuple_end = i + n - 1
                new_gram = tuple(x_bar[tuple_start:tuple_end])
                if new_gram not in ngram_counts[n]:
                    ngram_counts[n][new_gram] = 1
                else:
                    ngram_counts[n][new_gram] += 1
        else:
            for i in range(n - 1, len(y_bar) - 1):
                if i - n < 0:
                    continue
                if y_bar[i-1] != '-':
                    continue
                tuple_start = i - n + 1
                tuple_end = i + 1
                new_gram = tuple(x_bar[tuple_start:tuple_end])
                if new_gram not in ngram_counts[n]:
                    ngram_counts[n][new_gram] = 1
                else:
                    ngram_counts[n][new_gram] += 1

    def f(y_prev, y, x_bar, i):
        if y_prev != '-':
            return 0
        if y == '-':
            return 0

        flat_x_bar = list(itertools.chain(*x_bar))
        if look_after:
            if 0 < i < len(x_bar) + n:
                start = i
                end = i + n
                gram = tuple(flat_x_bar[start:end])
                if gram in ngram_counts[n]:
                    return ngram_counts[n][gram]
        else:
            if i - n + 1 >= 0 and i + 1 < len(flat_x_bar):
                start = i - n + 1
                end = i + 1
                gram = tuple(flat_x_bar[start:end])
                if gram in ngram_counts[n]:
                    return ngram_counts[n][gram]
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


def train_from_file(xycrf, corpus_path, model_path):
    """
    Estimates parameters using conjugate gradient methods.(L-BFGS-B used)
    """
    start_dt = datetime.now()
    print('[%s] Start training' % start_dt)

    # Read the training corpus
    print("* Reading training data ... ", end="")
    training_data, tag_set, ngram_sets = XyCrf.read_corpus(corpus_path, ns=[2,3,4,5])
    xycrf.set_tags(tag_list=list(tag_set))
    add_functions(xycrf=xycrf, training_data=training_data)
    xycrf.training_data = training_data
    print("Done reading data")

    print("* Number of labels: {}".format(xycrf.tag_count))
    print("* Number of features: {}".format(xycrf.feature_count))
    print("* Number of training examples: {}".format(len(training_data)))

    # gradient, big_z = xycrf.gradient_for_all_training()
    xycrf.stochastic_gradient_ascent_train()
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
    parser.add_argument("--train", help="data file for training input")
    parser.add_argument("--output", help="the model pickle file name to output")
    parser.add_argument("--test", help="data file for evaluation")
    parser.add_argument("--input", help="the model pickle file path")
    args = parser.parse_args()

    if args.train:
        crf = XyCrf(optimize=False)
        parallel_weights = train_from_file(xycrf=crf, corpus_path=args.train, model_path=args.output)
        # crf = XyCrf(optimize=False)
        # serial_weights = train_from_file(xycrf=crf, corpus_path=args.train, model_path=args.output)
        # for i in range(len(serial_weights)):
            # if serial_weights[i] != parallel_weights[i]:
                # print("Weights are DIFFERENT.")
                # break
        # print("Weights are the SAME.")
    if args.test:
        crf = XyCrf.unpickle(args.input)
        test_from_file(xycrf=crf, corpus_path=args.test)

    exit(0)
