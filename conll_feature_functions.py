def unigram_func_factory(token, track_index, offset):
    def f(y_prev, y, X, i):
        if X[i+offset][track_index] == token:
            return 1
        else:
            return 0
    return f


def bigram_func_factory(bigram, track_index, offset):
    def f(y_prev, y, X, i):
        length = len(X)
        if offset >= 0 and \
                i < length - offset and \
                X[i+offset][track_index] == bigram[0] and \
                X[i+offset+1][track_index] == bigram[1]:
            return 1
        elif i > -1 * offset and \
                X[i+offset][track_index] == bigram[0] and \
                X[i+offset+1][track_index] == bigram[1]:
            return 1
        else:
            return 0
    return f


def pos_trigram_func_factory(trigram, offset):
    if offset < 0:
        raise Exception("Bad offset.")
    track_index = 1

    def f(y_prev, y, X, i):
        length = len(X)
        if offset == 0 and i < length and \
                X[i][track_index] == trigram[0] and \
                X[i+1][track_index] == trigram[1] and \
                X[i+2][track_index] == trigram[2]:
            return 1
        elif offset == 1 and 0 < i < length - 1 and \
                X[i-1][track_index] == trigram[0] and \
                X[i][track_index] == trigram[1] and \
                X[i+1][track_index] == trigram[2]:
            return 1
        elif offset == 2 and i > 1 and \
                X[i-2][track_index] == trigram[0] and \
                X[i-1][track_index] == trigram[1] and \
                X[i][track_index] == trigram[2]:
            return 1
        else:
            return 0
    return f

