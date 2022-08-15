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
from xycrf import XyCrf
from conll_feature_functions import feature_function_set

parser = argparse.ArgumentParser()
parser.add_argument("datafile", help="data file for training input")
parser.add_argument("modelfile", help="the model file name. (output)")


args = parser.parse_args()

crf = XyCrf()

crf.train_from_file(corpus_filename=args.datafile,
                    feature_functions=feature_function_set,
                    model_filename=args.modelfile)
