__author__ = 'David Randolph'
# Copyright (c) 2022 David A. Randolph.
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
import copy

from conll_feature_functions import unigram_func_factory, bigram_func_factory, pos_trigram_func_factory
from nltk import ngrams
import time
from collections import Counter
from datetime import datetime

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from math import log, exp

SCALING_THRESHOLD = 1e250

ITERATION_NUM = 0
SUB_ITERATION_NUM = 0
TOTAL_SUB_ITERATIONS = 0
GRADIENT = None


def _training_callback(params):
    global ITERATION_NUM
    global SUB_ITERATION_NUM
    global TOTAL_SUB_ITERATIONS
    ITERATION_NUM += 1
    TOTAL_SUB_ITERATIONS += SUB_ITERATION_NUM
    SUB_ITERATION_NUM = 0


def _log_conditional_likelihood(params, *args):
    """
    Calculate likelihood and gradient
    """
    xycrf = args[0]
    return xycrf.log_conditional_likelihood()


def _gradient(params, *args):
    xycrf = args[0]
    return xycrf.get_gradient()


def _augment_ngram_sets(x_bar, ngram_sets, ns):
    track_count = len(x_bar[0])
    for track_index in range(track_count):
        for n in ns:
            if (track_index, n) not in ngram_sets:
                ngram_sets[(track_index, n)] = set()
            token_list = list()
            for vector in x_bar:
                token_list.append(vector[track_index])
            new_grams = ngrams(token_list, n)
            for gram in new_grams:
                ngram_sets[(track_index, n)].add(tuple(gram))


def _append_example(data, ngram_sets, ns, x_bar, y_bar):
    track_count = len(x_bar[0])
    x_0 = list()
    x_last = list()
    for _ in range(track_count):
        x_0.append('')
        x_last.append('')
    x_bar.insert(0, x_0)
    x_bar.append(x_last)
    y_bar.insert(0, 'START')
    y_bar.append('STOP')
    data.append((x_bar, y_bar))
    _augment_ngram_sets(x_bar=x_bar, ngram_sets=ngram_sets, ns=ns)


def _read_corpus(file_name, ns):
    """
    Read a corpus file with a format used in CoNLL.
    """
    data = list()
    data_string_list = list(open(file_name))
    tag_set = set()
    ngram_sets = dict()
    element_size = 0
    x_bar = list()
    y_bar = list()
    for data_string in data_string_list:
        words = data_string.strip().split()
        if len(words) == 0:
            _append_example(data, ngram_sets, ns, x_bar, y_bar)
            x_bar = list()
            y_bar = list()
        else:
            if element_size == 0:
                element_size = len(words)
            elif element_size is not len(words):
                raise Exception("Bad file format.")
            x_bar.append(words[:-1])
            y_bar.append(words[-1])
            tag_set.add(words[-1])
    if len(y_bar) > 1:
        _append_example(data, ngram_sets, ns, x_bar, y_bar)

    return data, tag_set, ngram_sets


class XyCrf():
    def __init__(self):
        self.training_data = None
        self.feature_functions = []
        self.function_by_name = dict()
        self.function_name_index = dict()
        self.function_index_name = dict()
        self.tag_count = 0
        self.tags = []
        self.feature_count = 0  # Number of global feature functions.
        self.weights = []
        self.tag_index_for_name = dict()
        self.tag_name_for_index = dict()
        self.g_matrix_list = []
        self.gradient = None

    def get_gradient(self):
        return self.gradient

    def set_gradient(self, new_values):
        if len(new_values) != self.feature_count:
            raise Exception("Bad gradient settings.")
        self.gradient = np.ndarray((self.feature_count,))
        for i in range(self.feature_count):
            self.gradient[i] = new_values[i]

    def set_tags(self, tag_list: list):
        self.tags = tag_list
        self.tags.append('START')
        self.tags.append('STOP')
        self.tag_count = len(self.tags)
        self.tag_index_for_name = dict()
        self.tag_name_for_index = dict()
        tag_index = 0
        for tag_name in self.tags:
            self.tag_index_for_name[tag_name] = tag_index
            self.tag_name_for_index[tag_index] = tag_name
            tag_index += 1

    def get_g_i(self, y_prev, y, x_bar, i):
        j = 0
        value = 0
        for func in self.feature_functions:
            weight = self.weights[j]
            value += weight * func(y_prev, y, x_bar, i)
        return value

    def get_g_i_dict(self, x_bar, i):
        # Our matrix is a dictionary
        g_i_dict = dict()
        for y_prev in self.tags:
            for y in self.tags:
                if not (y_prev in ('START', 'STOP') and y in ('START', 'STOP')):
                    g_i_dict[(y_prev, y)] = self.get_g_i(y_prev, y, x_bar, i)
        return g_i_dict

    def set_g_matrix_list(self, x_bar):
        self.g_matrix_list = list()
        for i in range(len(x_bar)):
            matrix = np.zeros((self.tag_count, self.tag_count))
            g_i = self.get_g_i_dict(x_bar, i)
            for (y_prev, y) in g_i:
                y_prev_index = self.tag_index_for_name[y_prev]
                y_index = self.tag_index_for_name[y]
                matrix[y_prev_index, y_index] = g_i[(y_prev, y)]
            self.g_matrix_list.append(matrix)
        return self.g_matrix_list

    def get_inference_g_list(self, x_bar):
        g_list = list()
        for i in range(len(x_bar)):
            g_i = self.get_g_i_dict(x_bar, i)
            g_list.append(g_i)
        return g_list

    def add_feature_function(self, func, name=None):
        self.feature_functions.append(func)
        if name is None:
            name = "f_{}".format(self.feature_count)
        self.function_by_name[name] = func
        self.function_name_index[name] = self.feature_count
        self.function_index_name[self.feature_count] = name
        self.feature_count += 1
        self.weights.append(1.0)

    def clear_feature_functions(self):
        self.feature_functions = []
        self.function_by_name = {}
        self.function_name_index = {}
        self.function_index_name = {}
        self.feature_count = 0
        self.weights = []

    def init_weights(self):
        big_j = len(self.feature_functions)
        self.weights = list()
        for j in range(big_j):
            self.weights.append(1.0)

    def add_feature_functions(self, functions):
        for func in functions:
            self.add_feature_function(func=func)

    def big_u(self, k, v_tag_index):
        if k == 0:
            if self.tag_name_for_index[v_tag_index] == 'START':
                return 1.0
            else:
                return 0.0  # ???
        max_value = -1 * float('inf')
        max_tag_index = None
        for u_tag_index in self.tag_name_for_index:
            value = self.big_u(k-1, u_tag_index) + self.g_matrix_list[k][u_tag_index, v_tag_index]
            if value > max_value:
                max_value = value
                max_u_tag_index = u_tag_index
        return max_value, u_tag_index

    def viterbi(self, x_bar, g_list):
        n = len(x_bar)

        big_u_dict = dict()
        for k in range(1, n):
            for v_tag in self.tag_index_for_name:
                v_tag_index = self.tag_index_for_name[v_tag]
                max_value, max_u_tag_index = self.big_u(k=k, v_tag_index=v_tag_index)
                big_u_dict[(k, v_tag)] = (max_value, max_u_tag_index)

        y_bar = list()
        max_value = -1 * float('inf')
        prior_tag = None
        for v_tag in self.tag_index_for_name:
            v_tag_index = self.tag_index_for_name[v_tag]
            (value, u_tag_index) = big_u_dict[(k, v_tag)]
            if value < max_value:
                max_value = value
                prior_tag = self.tag_name_for_index[u_tag_index]
        y_bar.append(prior_tag)
        for k in range(n-1, 1, -1):
            (prior_max, prior_tag) = big_u_dict[(k, prior_tag)]
            y_bar.insert(0, prior_tag)
        return y_bar

    def viterbi_iterative(self, x_bar, g_list):
        # Modeled after Seong-Jin Kim's implementation.
        time_len = len(x_bar)
        max_table = np.zeros((time_len, self.tag_count))
        argmax_table = np.zeros((time_len, self.tag_count), dtype='int64')

        t = 0
        for tag_index in self.tag_index_for_name:
            max_table[t, tag_index] = g_list[t][('START', self.tag_name_for_index[tag_index])]

        for t in range(1, time_len):
            for tag_index in range(1, self.tag_count):
                tag = self.tag_name_for_index[tag_index]
                max_value = -float('inf')
                max_tag_index = None
                for prev_tag_index in range(1, self.tag_count):
                    prev_tag = self.tag_name_for_index[prev_tag_index]
                    value = max_table[t - 1, prev_tag_index] * g_list[t][(prev_tag, tag)]
                    if value > max_value:
                        max_value = value
                        max_tag_index = prev_tag_index
                max_table[t, tag_index] = max_value
                argmax_table[t, tag_index] = max_tag_index

        sequence = list()
        next_tag_index = max_table[time_len - 1].argmax()
        sequence.append(self.tag_name_for_index[next_tag_index])
        for t in range(time_len - 1, -1, -1):
            next_tag_index = argmax_table[t, next_tag_index]
            sequence.append(self.tag_name_for_index[next_tag_index])
        # return [self.label_dic[label_id] for label_id in sequence[::-1][1:]]
        return sequence

    def alpha(self, k_plus_1, v_tag_index):
        if k_plus_1 == 0:
            if v_tag_index == self.tag_index_for_name['START']:
                return 1.0
            else:
                return 0.0
        k = k_plus_1 - 1
        sum_total = 0.0
        for u_tag_index in self.tag_name_for_index:
            sum_total += self.alpha(k, u_tag_index) * \
                (exp(self.g_matrix_list[k+1][u_tag_index, v_tag_index]))
        return sum_total

    def beta(self, u_tag_index, k):
        n = len(self.g_matrix_list)  # Length of the sequence
        if k == n + 1:
            if u_tag_index == self.tag_index_for_name['STOP']:
                return 1.0
            else:
                return 0.0
        sum_total = 0
        for v_tag_index in self.tag_name_for_index:
            sum_total += exp(self.g_matrix_list[k+1][u_tag_index, v_tag_index]) * \
                self.beta(v_tag_index, k+1)

    def zed_forward(self, x_bar):
        n = len(self.g_matrix_list)
        Z = self.alpha(n+1, self.tag_index_for_name['STOP'])
        return Z

    def zed_backward(self, x_bar):
        Z = self.beta(self.tag_index_for_name['START'], 0)
        return Z

    def expectation_for_function(self, function_index, x_bar, y_bar):
        n = len(self.g_matrix_list)
        big_z = self.zed_forward(x_bar)
        expectation = 0.0
        for i in range(1, n+1):
            for y_index_minus_1 in range(0, n+1):
                for y_index in range(1, n+1):
                    y_prev = y_bar[y_index_minus_1]
                    y_prev_tag_index = self.tag_index_for_name[y_prev]
                    y = y_bar[y_index]
                    y_tag_index = self.tag_index_for_name[y]
                    feature_value = self.feature_functions[function_index](y_prev=y_prev, y=y, x_bar=x_bar, i=y_index)
                    alpha_value = self.alpha(k_plus_1=y_index_minus_1, v_tag_index=y_prev_tag_index)
                    exp_g_i_value = exp(self.g_matrix_list[y_index][y_index_minus_1, y_index])
                    beta_value = self.beta(u_tag_index=y_tag_index, k=y_index)
                    expectation += feature_value * ((alpha_value * exp_g_i_value * beta_value) / big_z)
        return expectation, big_z

    def infer(self, x_bar):
        g_list = self.get_inference_g_list(x_bar)
        y_hat = self.viterbi(x_bar, g_list)
        return y_hat

    def big_f(self, function_index, x_bar, y_bar):
        n = len(y_bar)
        func = self.feature_functions[function_index]
        sum_total = 0
        for i in range(1, n):
            val = func(y_prev=y_bar[i-1], y=y_bar[i], x_bar=x_bar, i=i)
            sum_total += val
        print("Returning")
        return sum_total

    def gradient_for_all_training(self):
        function_count = len(self.feature_functions)
        gradient = list()
        big_z = 0
        for example in self.training_data:
            x_bar = example[0]
            y_bar = example[1]
            for j in range(function_count):
                example_val = self.big_f(function_index=j, x_bar=x_bar, y_bar=y_bar)
                expected_val, example_zed = self.expectation_for_function(function_index=j, x_bar=x_bar, y_bar=y_bar)
                if j not in gradient:
                    gradient.append(0)
                gradient[j] += example_val - expected_val
                big_z += example_zed
        return gradient, big_z

    def log_conditional_likelihood(self):
        gradient, big_z = self.gradient_for_all_training()
        weighted_feature_sum = 0
        function_count = len(self.feature_functions)
        for j in range(function_count):
            global_feature_val = 0
            for example in self.training_data:
                x_bar = example[0]
                y_bar = example[1]
                global_feature_val += self.big_f(function_index=j, x_bar=x_bar, y_bar=y_bar)
            weighted_feature_sum += self.weights[j] * global_feature_val
        likelihood = weighted_feature_sum - log(big_z)
        self.set_gradient(gradient)
        return likelihood

    def stochastic_gradient_ascent_train(self, learning_rate=0.01):
        function_count = len(self.feature_functions)
        self.init_weights()
        for example in self.training_data:
            x_bar = example[0]
            y_bar = example[1]
            for j in range(function_count):
                global_feature_val = self.big_f(function_index=j, x_bar=x_bar, y_bar=y_bar)
                expected_val, _ = self.expectation_for_function(function_index=j, x_bar=x_bar, y_bar=y_bar)
                self.weights[j] = self.weights[j] + learning_rate * global_feature_val - expected_val

    def train(self):
        self.init_weights()
        print('* Start L-BGFS')
        print('   ========================')
        print('   iter(sit): likelihood')
        print('   ------------------------')
        self.weights, log_likelihood, information = \
            fmin_l_bfgs_b(func=_log_conditional_likelihood, fprime=_gradient,
                          x0=np.zeros(self.feature_count), args=[self],
                          callback=_training_callback)
        print('   ========================')
        print('   (iter: iteration, sit: sub iteration)')
        print('* Training has been finished with %d iterations' % information['nit'])

        if information['warnflag'] != 0:
            print('* Warning (code: %d)' % information['warnflag'])
            if 'task' in information.keys():
                print('* Reason: %s' % (information['task']))
        print('* Likelihood: %s' % str(log_likelihood))

    def add_unigram_functions(self, ngram_sets):
        for (track_index, n) in ngram_sets:
            if n != 1:
                continue
            unigram_set = ngram_sets[(track_index, n)]
            for unigram in unigram_set:
                token = unigram[0]
                for offset in (-2, -1, 0, 1, 2):
                    name = "unigram_{}_track_{}_{}".format(offset, track_index, token)
                    func = unigram_func_factory(token=token, track_index=track_index, offset=offset)
                    # result = func(y_prev=None, y=None, x_bar=[[token, 'foo']], i=0)
                    self.add_feature_function(func=func, name=name)

    def add_bigram_functions(self, ngram_sets):
        for (track_index, n) in ngram_sets:
            if n != 2:
                continue
            bigram_set = ngram_sets[(track_index, n)]
            for bigram in bigram_set:
                for offset in (-2, -1, 0, 1, 2):
                    name = "bigram_{}_track_{}_{}-{}".format(offset, track_index, bigram[0], bigram[1])
                    func = bigram_func_factory(bigram=bigram, track_index=track_index, offset=offset)
                    self.add_feature_function(func=func, name=name)

    def add_trigram_functions(self, ngram_sets):
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
                    self.add_feature_function(func=func, name=name)

    def train_from_file(self, corpus_filename, model_filename):
        """
        Estimates parameters using conjugate gradient methods.(L-BFGS-B used)
        """
        start_time = time.time()
        print('[%s] Start training' % datetime.now())

        # Read the training corpus
        print("* Reading training data ... ", end="")
        training_data, tag_set, ngram_sets = _read_corpus(corpus_filename, ns=[1, 2, 3])
        self.set_tags(tag_list=list(tag_set))
        self.add_unigram_functions(ngram_sets)
        self.add_bigram_functions(ngram_sets)
        self.add_trigram_functions(ngram_sets)
        self.training_data = training_data
        print("Done reading data")

        print("* Number of labels: {}".format(self.tag_count))
        print("* Number of features: {}".format(self.feature_count))

        # Estimates parameters to maximize log-likelihood of the corpus.
        self.train()

        # self.save_model(model_filename)

        elapsed_time = time.time() - start_time
        print('* Elapsed time: %f' % elapsed_time)
        print('* [%s] Training done' % datetime.now())


if __name__ == '__main__':
    rh_tags = ['>1', '>2', '>3', '>4', '>5']
    xyc = XyCrf()
    xyc.set_tags(tag_list=rh_tags)