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

from xycrf import XyCrf


def unigram_func_factory(token, track_index, offset):
    def f(y_prev, y, x_bar, i):
        if i + offset >= len(x_bar):
            return 0
        if x_bar[i+offset][track_index] == token:
            return 1
        else:
            return 0
    return f


def bigram_func_factory(bigram, track_index, offset):
    def f(y_prev, y, x_bar, i):
        length = len(x_bar)
        if i + offset >= length or i + offset + 1 >= length:
            return 0
        if offset >= 0 and \
                i < length - offset and \
                x_bar[i+offset][track_index] == bigram[0] and \
                x_bar[i+offset+1][track_index] == bigram[1]:
            return 1
        elif i > -1 * offset and \
                x_bar[i+offset][track_index] == bigram[0] and \
                x_bar[i+offset+1][track_index] == bigram[1]:
            return 1
        else:
            return 0
    return f


def pos_trigram_func_factory(trigram, offset):
    if offset < 0:
        raise Exception("Bad offset.")
    track_index = 1

    def f(y_prev, y, x_bar, i):
        length = len(x_bar)
        if offset == 0 and i < length - 1 and \
                x_bar[i][track_index] == trigram[0] and \
                x_bar[i+1][track_index] == trigram[1] and \
                x_bar[i+2][track_index] == trigram[2]:
            return 1
        elif offset == 1 and 0 < i < length - 1 and \
                x_bar[i-1][track_index] == trigram[0] and \
                x_bar[i][track_index] == trigram[1] and \
                x_bar[i+1][track_index] == trigram[2]:
            return 1
        elif offset == 2 and i > 1 and \
                x_bar[i-2][track_index] == trigram[0] and \
                x_bar[i-1][track_index] == trigram[1] and \
                x_bar[i][track_index] == trigram[2]:
            return 1
        else:
            return 0
    return f


def add_unigram_functions(xycrf, ngram_sets):
    for (track_index, n) in ngram_sets:
        if n != 1:
            continue
        unigram_set = ngram_sets[(track_index, n)]
        for unigram in unigram_set:
            token = unigram[0]
            # for offset in (-2, -1, 0, 1, 2):
            # FIXME
            for offset in [0]:
                name = "unigram_{}_track_{}_{}".format(offset, track_index, token)
                func = unigram_func_factory(token=token, track_index=track_index, offset=offset)
                # result = func(y_prev=None, y=None, x_bar=[[token, 'foo']], i=0)
                xycrf.add_feature_function(func=func, name=name)


def add_bigram_functions(xycrf, ngram_sets):
    for (track_index, n) in ngram_sets:
        if n != 2:
            continue
        bigram_set = ngram_sets[(track_index, n)]
        for bigram in bigram_set:
            for offset in (-2, -1, 0, 1, 2):
                name = "bigram_{}_track_{}_{}-{}".format(offset, track_index, bigram[0], bigram[1])
                func = bigram_func_factory(bigram=bigram, track_index=track_index, offset=offset)
                xycrf.add_feature_function(func=func, name=name)


def add_trigram_functions(xycrf, ngram_sets):
    for (track_index, n) in ngram_sets:
        if n != 3:
            continue
        if track_index != 1:
            continue  # trigrams only for pos track
        trigram_set = ngram_sets[(track_index, n)]
        for trigram in trigram_set:
            for offset in (0, 1, 2):
                name = "pos_trigram_{}_{}-{}-{}".format(offset, track_index, trigram[0], trigram[1], trigram[2])
                func = pos_trigram_func_factory(trigram=trigram, offset=offset)
                xycrf.add_feature_function(func=func, name=name)


def add_functions(xycrf, ngram_sets):
    add_unigram_functions(xycrf, ngram_sets)
    # add_bigram_functions(xycrf, ngram_sets)
    # add_trigram_functions(xycrf, ngram_sets)


def train_from_file(xycrf, corpus_path, model_path):
    """
    Estimates parameters using conjugate gradient methods.(L-BFGS-B used)
    """
    start_dt = datetime.now()
    print('[%s] Start training' % start_dt)

    # Read the training corpus
    print("* Reading training data ... ", end="")
    training_data, tag_set, ngram_sets = XyCrf.read_corpus(corpus_path, ns=[1, 2, 3])
    xycrf.set_tags(tag_list=list(tag_set))
    add_functions(xycrf, ngram_sets)
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


def test_from_file(xycrf, corpus_path):
    test_data, tag_set, ngram_sets = XyCrf.read_corpus(corpus_path, ns=[1, 2, 3])
    inferences = xycrf.infer_all(test_data)
    print(inferences)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="data file for training input")
    parser.add_argument("--output", help="the model pickle file name to output")
    parser.add_argument("--test", help="data file for evaluation")
    parser.add_argument("--input", help="the model pickle file path")
    args = parser.parse_args()

    if args.train:
        crf = XyCrf()
        train_from_file(xycrf=crf, corpus_path=args.train, model_path=args.output)
    if args.test:
        crf = XyCrf.unpickle(args.input)
        test_from_file(xycrf=crf, corpus_path=args.test)
