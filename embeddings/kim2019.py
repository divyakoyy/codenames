import itertools
import numpy as np

import gensim.models.keyedvectors as word2vec
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from scipy.spatial.distance import cosine as cos_dist


class Kim2019(object):
    def __init__(self, w2v_file_path='data/GoogleNews-vectors-negative300.bin'):
        super().__init__()
        self.word_vectors = word2vec.KeyedVectors.load_word2vec_format(w2v_file_path, binary=True, unicode_errors='ignore')
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.lancaster_stemmer = LancasterStemmer()
        self.cm_cluelist = []  # candidate clues
        with open('data/cm_wordlist.txt') as infile:
            for line in infile:
                self.cm_cluelist.append(line.rstrip())
        self.word_dists = {}

    def get_weighted_nn(self, word):
        nn_w_dists = {}
        if word in self.word_vectors:
            all_vectors = (self.word_vectors,)
            for clue in self.cm_cluelist:
                b_dist = cos_dist(self.concatenate(clue, all_vectors), self.concatenate(word, all_vectors))
                nn_w_dists[clue] = b_dist
        else:
            print(word, "not in word_vectors")
        self.word_dists[word] = nn_w_dists
        return {k: 1-v for k,v in nn_w_dists.items()}

    # def update_word_dists(self, red_words, blue_words):
    #     to_remove = set(self.red_word_dists) - set(red_words)
    #     for word in to_remove:
    #         del self.red_word_dists[word]
    #     to_remove = set(self.blue_word_dists) - set(blue_words)
    #     for word in to_remove:
    #         del self.blue_word_dists[word]

    def get_clue(self, blue_words, red_words, chosen_blue_words):
        # print("BLUE:\t", blue_words)

        bests = {}

        # NOTE: we don't actually want to delete red/blue words from the dists
        # self.update_word_dists(red_words, blue_words)

        # for clue_num in range(1, 3 + 1):
        #     best_per_dist = np.inf
        #     best_per = ''
        #     best_blue_word = ''
        #     for blue_word in list(itertools.combinations(blue_words, clue_num)):
        best_clue = ''
        best_blue_dist = np.inf
        for clue in self.cm_cluelist:
            if not self.arr_not_in_word(clue, blue_words.union(red_words)):
                continue

            red_dist = np.inf
            for red_word in red_words:
                if clue in self.word_dists[red_word]:
                    if self.word_dists[red_word][clue] < red_dist:
                        red_dist = self.word_dists[red_word][clue]
            worst_blue_dist = 0
            for blue in chosen_blue_words:
                if clue in self.word_dists[blue]:
                    dist = self.word_dists[blue][clue]
                else:
                    dist = np.inf
                if dist > worst_blue_dist:
                    worst_blue_dist = dist


            if worst_blue_dist < best_blue_dist and worst_blue_dist < red_dist:
                best_blue_dist = worst_blue_dist
                best_clue = clue
                # print(worst_blue_dist,chosen_blue_words,clue)
        chosen_num = len(chosen_blue_words)
        bests[chosen_num] = (chosen_blue_words, best_clue, best_blue_dist)

        # for each clue (best for that clue_num)
        # find the worst distance and best distance between chosen words and clue
        # but they only use the worst distance, which was already calculated above
        # print("BESTS: ", bests)
        li = []
        pi = []
        chosen_clue_info = bests[chosen_num]
        for clue_num, clue_info in bests.items():
            best_blue_words, best_clue, worst_blue_dist = clue_info
            # NOTE: This is repeating what they already calculated above
            # and doesn't even use the values?
            # chosen_clue_info and chosen_num are the only variables that matter

            # worst_blue_dist = -np.inf
            # best_blue_dist = np.inf
            # worst_blue = ''
            # for blue in best_blue_words:
            #     dist = cos_dist(self.concatenate(blue, all_vectors), self.concatenate(best_clue, all_vectors))
            #     if dist > worst_blue_dist:
            #         worst_blue = blue
            #         worst_blue_dist = dist
            #     if dist < best_blue_dist:
            #         best_blue_dist = dist
            # NOTE: this only works because dicts guarantee insertion order in py>=3.7
            # otherwise we'd be choosing at random among clues that pass the threshold
            # instead of picking the one with the highest clue_num
            if worst_blue_dist != -np.inf:
                # print(worst_blue_dist, chosen_clue_info, chosen_num)
                chosen_clue_info = clue_info
                chosen_num = clue_num

            # li.append((worst / best, best_blue_words, worst_blue, best_clue,
            #            best_blue_dist, best_blue_dist ** len(best_blue_words)))
            # only li[0][3] is ever used, and only if chosen_clue_info[2] === np.inf
            li.append(best_clue)

        if chosen_clue_info[2] == np.inf:
            # NOTE: why is the tuple necessary...?
            # chosen_clue_info = ('', li[0][3], 0)
            chosen_clue_info = ('', li[0], np.inf)
            chosen_num = 1
        # print("LI: ", li)
        # print("The clue is: ", li[0][3])
        print('chosen_clue_info is:', chosen_clue_info)
        # return in array styled: ["clue", number]
        # return chosen_clue_info, chosen_num  # [li[0][3], 1]
        return chosen_clue_info[1], chosen_clue_info[2]

    def arr_not_in_word(self, word, arr):
        if word in arr:
            return False
        lemm = self.wordnet_lemmatizer.lemmatize(word)
        lancas = self.lancaster_stemmer.stem(word)
        for i in arr:
            if i == lemm or i == lancas:
                return False
            if i.find(word) != -1:
                return False
            if word.find(i) != -1:
                return False
        return True

    def combine(self, words, wordvecs):
        factor = 1.0 / float(len(words))
        new_word = self.concatenate(words[0], wordvecs) * factor
        for word in words[1:]:
            new_word += self.concatenate(word, wordvecs) * factor
        return new_word

    def concatenate(self, word, wordvecs):
        concatenated = wordvecs[0][word]
        for vec in wordvecs[1:]:
            concatenated = np.hstack((concatenated, vec[word]))
        return concatenated
