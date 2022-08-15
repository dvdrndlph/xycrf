# xycrf
A CRF package supporting true feature functions of both observations and labels (X and Y).

## Usage

You can test this code with [CoNLL 2000 chunking data](https://www.clips.uantwerpen.be/conll2000/chunking/).

### Training

```sh
# format
python3 xycrf_train.py <train_file> <model_file>

# example
python3 xycrf_train.py data/chunking_small/small_train.data small_model.json
```

### Test

```sh
# format
python3 xycrf_test.py <test_file> <trained_model_file>

# example
python3 xycrf_test.py data/chunking_small/small_test.data small_model.json
```

## Benchmark Result

- Data: CoNLL corpus
    - [data/chunking_full](https://github.com/dvdrndlph/xycrf/data/chunking_full): original data (8936 sentences)
    - [data/chunking_small](https://github.com/dvdrndlph/xycrf/data/chunking_small): sampled data (77 sentences)
- Compared with [CRF++](http://taku910.github.io/crfpp/)
- Use feature set

**Accuracy**

|                | xycrf    |  CRF++   |
|--------------- |----------| -------- |
| chunking_full  | 0.000000 | 0.960128 |
| chunking_small | 0.000000 | 0.889474 |

## License
MIT

## Credits
Some methods and program structure inspired by (borrowed from) Seong-Jin Kim's [crf package](https://github.com/lancifollia/crf).
Many thanks.

## References
- An Introduction to Conditional Random Fields / Charles Sutton, Andrew McCallum/ 2010
- Log-Linear Models and Conditional Random Fields / Charles Elkan / 2014