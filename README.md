# xycrf
A first-order linear-chain CRF package supporting true feature functions of both observations and tags (X and Y),
per the original (some might say painfully naive) CRF formalism.

At this point, xycrf relies exclusively on stochastic gradient ascent for training. L-BFGS training options are
forthcoming, but performance of L-BFGS promises to be *painfully* slow in this context.

## Usage
 
You can test this code with [Trogkanis/Elkan hyphenation datasets](https://cseweb.ucsd.edu/~elkan/hyphenation/),
provided in the data/hyphen directory.

### Training
```sh
# example
$ python3 hyphen_model.py --train data/hyphen/english/all.data --epochs 3 --rate 0.2 \
     --attenuation 0.1 --output /tmp/hyphen_engl_model_3ep_ts20.dill --test_size 0.20
```

### Testing

```sh
# example
$ python3 hyphen_model.py --test data/hyphen/english/all.data --test_size 0.20 \
    --input /tmp/hyphen_engl_model_3ep_ts20.dill
```

## Benchmark Result

Data: The Trogkanis/Elkan English dataset, provided in [data/hyphen/english](https://github.com/dvdrndlph/xycrf/data/hyphen/english).
This contains 66,001 hyphenated English words. 

Our model compares unfavorably to the published results in
the [Trogkanis/Elkan](https://aclanthology.org/P10-1038/) ACL paper. We also run about four times slower on hardware
that is ten years newer, using a training method that makes no guarantees about convergence to global maximum goodness. But it does inspire some confidence in the underlying CRF implementation.

**Accuracy**

|                   | T/E L-BFGS | Our SGA |
|-------------------|------------|---------|
| Word-level        | 96.33%     | 45.64%  |
| Character-level   |            | 89.83%  |
| Feature functions | 2,916,942  | 11      |


## License
MIT

## Thanks
Inspired by Seong-Jin Kim's [crf package](https://github.com/lancifollia/crf).
Thanks to the lectures and notes of Charles Elkan, who finally explained all this well enough
for me to be able to do this. Thanks to Oscar Laird for the conversations.

## References
- Elkan, C. (2014). <i>Log-linear models and conditional random fields</i>. http://cseweb.ucsd.edu/~elkan/250B/CRFs.pdf
- Elkan, C. (2014). <i>Maximum Likelihood, Logistic Regression, and Stochastic Gradient Training</i>. https://cseweb.ucsd.edu/~elkan/250B/logreg.pdf
- <div class="csl-entry">Sutton, C., &#38; McCallum, A. (2011). An introduction to conditional random fields. <i>Foundations and Trends in Machine Learning</i>, <i>4</i>(4), 267–373. https://doi.org/10.1561/2200000013</div> 
- Trogkanis, N., &#38; Elkan, C. (2010). Conditional random fields for word hyphenation. <i>Proceedings of the 48th Annual Meeting of the Association for Computational Linguistics</i>, 366–374.