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

def print_vector(filename, vector):
    fout = open(filename, 'w')
    for k in range(len(vector)):
        fout.write('%s\n' % str(vector[k]))
    fout.close()


def print_vector_history(filename, history):
    if len(history) < 1:
        return
    fout = open(filename, 'w')
    num_iterations = len(history)
    for i in range(num_iterations):
        if i != 0:
            fout.write('\t')
        fout.write('%d' % i)
    fout.write('\n')
    for k in range(len(history[0])):
        for i in range(num_iterations):
            if i != 0:
                fout.write('\t')
            fout.write('%s' % str(history[i][k]))
        fout.write('\n')
    fout.close()


def print_potential_table(num_labels, table, X):
    fout = open('_potential.txt', 'w')
    fout.write('pot\t')
    for t in range(len(X)):
        if t != 0:
            fout.write('\t')
        fout.write('%d' % t)
    fout.write('\n')
    for prev_y in range(num_labels):
        for y in range(num_labels):
            for t in range(len(X)):
                if t == 0:
                    fout.write('(%d, %d)\t' % (prev_y, y))
                else:
                    fout.write('\t')
                fout.write('%s' % str(table[prev_y, y, t]))
            fout.write('\n')
    fout.close()


def print_alpha_beta_table(time_length, num_labels, alpha, beta):
    fout = open('_alpha.txt', 'w')
    fout.write('a\t')
    for t in range(time_length):
        if t != 0:
            fout.write('\t')
        fout.write('%d' % t)
    fout.write('\n')
    for y in range(num_labels):
        for t in range(time_length):
            if t == 0:
                fout.write('%d\t' % y)
            else:
                fout.write('\t')
            fout.write('%s' % str(alpha[t, y]))
        fout.write(''
                   '\n')
    fout.close()

    fout = open('_beta.txt', 'w')
    fout.write('a\t')
    for t in range(time_length):
        if t != 0:
            fout.write('\t')
        fout.write('%d' % t)
    fout.write('\n')
    for y in range(num_labels):
        for t in range(time_length):
            if t == 0:
                fout.write('%d\t' % y)
            else:
                fout.write('\t')
            fout.write('%s' % str(beta[t, y]))
        fout.write('\n')
    fout.close()