def unigram_func_factory(token, track_index, offset):
    def f(y_prev, y, x_bar, i):
        if x_bar[i+offset][track_index] == token:
            return 1
        else:
            return 0
    return f


def bigram_func_factory(bigram, track_index, offset):
    def f(y_prev, y, x_bar, i):
        length = len(x_bar)
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
        if offset == 0 and i < length and \
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

