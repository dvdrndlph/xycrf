#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
from nltk import ngrams
from sklearn.model_selection import KFold, train_test_split


def augment_ngram_sets(x_bar, ngram_sets, ns):
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


def append_example(data, ngram_sets, ns, x_bar, y_bar):
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
    augment_ngram_sets(x_bar=x_bar, ngram_sets=ngram_sets, ns=ns)


def read_corpus(file_name, ns):
    """
    Read a corpus file with the format used in CoNLL.
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
            append_example(data, ngram_sets, ns, x_bar, y_bar)
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
        append_example(data, ngram_sets, ns, x_bar, y_bar)

    return data, tag_set, ngram_sets


def get_ngram_sets(data, ns):
    ngram_sets = dict()
    for (x_bar, y_bar) in data:
        augment_ngram_sets(x_bar=x_bar, ngram_sets=ngram_sets, ns=ns)
    return ngram_sets


def get_tag_set(data):
    tag_set = set()
    for (_, y_bar) in data:
        for y in y_bar:
            tag_set.add(y)
    return tag_set


def get_dataset_fields(data, ns):
    fields = dict()
    fields['data'] = data
    fields['tag_set'] = get_tag_set(data=data)
    fields['ngram_sets'] = get_ngram_sets(data=data, ns=ns)
    return fields


def split_corpus(file_name, ns, seed: int,
                 train_size=0.7, validation_size=0.15, test_size=0.15):
    splits = {
        'train': dict(),
        'validation': dict(),
        'test': dict()
    }
    data, tag_set, ngram_sets = read_corpus(file_name=file_name, ns=ns)
    (training_data, non_training_data) = train_test_split(data,
                                                          train_size=train_size,
                                                          random_state=seed,
                                                          shuffle=True)
    splits['train'] = get_dataset_fields(data=training_data, ns=ns)

    if validation_size == 0:
        splits['test'] = get_dataset_fields(data=non_training_data, ns=ns)
    else:
        non_training_size = validation_size + test_size
        scaled_validation_size = validation_size / non_training_size
        (validation_data, test_data) = train_test_split(non_training_data,
                                                        train_size=scaled_validation_size,
                                                        random_state=seed,
                                                        shuffle=True)
        splits['validation'] = get_dataset_fields(data=validation_data, ns=ns)
        splits['test'] = get_dataset_fields(data=test_data, ns=ns)

    return splits


def k_fold_corpus(file_name, ns, seed: int, k: int, holdout_size=0.15):
    splits = {
        'folds': list(),
        'holdout': dict()
    }
    data, tag_set, ngram_sets = read_corpus(file_name=file_name, ns=ns)
    if holdout_size != 0:
        (data, holdout_data) = train_test_split(data,
                                                test_size=holdout_size,
                                                random_state=seed,
                                                shuffle=True)
        splits['holdout'] = get_dataset_fields(data=holdout_data, ns=ns)
    folder = KFold(n_splits=k, random_state=seed, shuffle=True)
    for train_indices, test_indices in folder.split(data):
        fold_fields = dict()
        fold_train_data = list()
        fold_test_data = list()
        for index in train_indices:
            fold_train_data.append(data[index])
        for index in test_indices:
            fold_test_data.append(data[index])
        fold_fields['train'] = get_dataset_fields(data=fold_train_data, ns=ns)
        fold_fields['test'] = get_dataset_fields(data=fold_test_data, ns=ns)
        splits['folds'].append(fold_fields)
    return splits



