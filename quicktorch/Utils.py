
import re
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot

class Utils:

    # --------------------------------------------------------------------
    def padding(array, length):

        res = array[:]

        for it in range(len(res)):

            if len(res[it]) < length:
                missing = length - len(res[it])
                res[it] = [0 for it in range(missing)] + res[it]
            if len(res[it]) > length:
                res[it] = res[it][:length]

        return res

    # --------------------------------------------------------------------
    def index(line, word_idx, idx_word, counter):

        try:
            words = line.split(" ")
        except:
            print("[ERROR]", line)
            return False, counter
        sentence = []
        for word in words:
            word = re.sub(r'[^\w\s]', "", word.lower())
            if word not in word_idx:
                word_idx[word] = counter
                idx_word[counter] = word
                counter += 1
            sentence.append(word_idx[word])
        return sentence, counter

    # --------------------------------------------------------------------
    def indexColumn(lines):

        word_idx = {"[nop]": 0}
        idx_word = {0: "[nop]"}
        counter = 1
        longest = 0
        x = []
        for line in lines:
            sentence, counter = Utils.index(line, word_idx, idx_word, counter)
            if not sentence: continue
            x.append(sentence)
            if len(sentence) > longest: longest = len(sentence)
        return x, word_idx, idx_word, counter, longest

    # -----------------------------------------------------------
    def plotly(xs, ys, names, types=[], filename="output.html"):
        """ Visualize network training performance

        Exports a HTML plotly file that allows exploration of
        network performance.
        """

        plots = []
        for it in range(len(xs)):
            plots.append(go.Scatter(x=xs[it], y=ys[it], name=names[it]))

        plot(plots, filename=filename, auto_open=False)
        

