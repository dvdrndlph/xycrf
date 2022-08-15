
def unigram_0(y_prev, y, X, i):
    return 'U[0]:%s' % (X[i][0])


def pos_unigram_0(y_prev, y, X, i):
    return 'POS_U[0]:%s' % (X[i][1])


def unigram_p1(y_prev, y, X, i):
    length = len(X)
    if i < length - 1:
        return 'U[+1]:%s' % (X[i+1][0])
    return ""


def unigram_p2(y_prev, y, X, i):
    length = len(X)
    if i < length - 2:
        return 'U[+2]:%s' % (X[i+2][0])
    return ""


def bigram_0(y_prev, y, X, i):
    length = len(X)
    if i < length - 1:
        return 'B[0]:%s %s' % (X[i][0], X[i+1][0])
    return ""


def pos_unigram_p1(y_prev, y, X, i):
    length = len(X)
    if i < length - 1:
        return 'POS_U[1]:%s' % (X[i+1][1])
    return ""


def pos_unigram_p2(y_prev, y, X, i):
    length = len(X)
    if i < length - 1:
        return 'POS_U[2]:%s' % (X[i+2][1])
    return ""


def pos_bigram_0(y_prev, y, X, i):
    length = len(X)
    if i < length - 1:
        return 'POS_B[0]:%s %s' % (X[i][1], X[i+1][1])
    return ""


def pos_bigram_p1(y_prev, y, X, i):
    length = len(X)
    if i < length - 2:
        return 'POS_B[1]:%s %s' % (X[i+1][1], X[i+2][1])
    return ""


def pos_trigram_0(y_prev, y, X, i):
    length = len(X)
    if i < length - 2:
        return 'POS_T[0]:%s %s %s' % (X[i][1], X[i+1][0], X[i+2][1])
    return ""


def unigram_m1(y_prev, y, X, i):
    if i > 0:
        return 'U[-1]:%s' % (X[i-1][0])
    return ""


def unigram_m2(y_prev, y, X, i):
    if i > 1:
        return 'U[-2]:%s' % (X[i-2][0])
    return ""


def bigram_m1(y_prev, y, X, i):
    if i > 0:
        return 'B[-1]:%s %s' % (X[i-1][0], X[i][0])
    return ""


def pos_unigram_m1(y_prev, y, X, i):
    if i > 0:
        return 'POS_U[-1]:%s' % (X[i-1][1])
    return ""


def pos_unigram_m2(y_prev, y, X, i):
    if i > 1:
        return 'POS_U[-2]:%s' % (X[i-2][1])
    return ""


def pos_bigram_m1(y_prev, y, X, i):
    if i > 0:
        return 'POS_B[-1]:%s %s' % (X[i-1][1], X[i][1])
    return ""


def pos_bigram_m2(y_prev, y, X, i):
    if i > 1:
        return 'POS_B[-2]:%s %s' % (X[i-2][1], X[i-1][1])
    return ""


def pos_trigram_m1(y_prev, y, X, i):
    length = len(X)
    if i > 0 and i < length - 1:
        return 'POS_T[-1]:%s %s %s' % (X[i-1][1], X[i][1], X[i+1][1])
    return ""


def pos_trigram_m2(y_prev, y, X, i):
    if i > 1:
        return 'POS_T[-2]:%s %s %s' % (X[i-2][1], X[i-1][1], X[i][1])
    return ""


feature_function_set = [
    unigram_0,
    pos_unigram_0,
    unigram_p1,
    bigram_0,
    pos_unigram_p1,
    pos_unigram_p2,
    pos_bigram_0,
    pos_bigram_p1,
    pos_trigram_0,
    unigram_m1,
    unigram_m2,
    bigram_m1,
    pos_unigram_m1,
    pos_unigram_m2,
    pos_bigram_m1,
    pos_bigram_m2,
    pos_trigram_m1,
    pos_trigram_m2
]