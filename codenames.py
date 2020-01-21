import random
import pprint
import itertools
import heapq
import sys
sys.path.insert(0, '../')
import process_hsm
import knn
import numpy as np
import statistics
from annoy import AnnoyIndex
import operator
from multisense import utils

class Game(object):

    def __init__(self,
                 ann_graph_path=None,
                 num_emb_batches = 22,
                 emb_type = 'custom',
                 emb_size = 200):
        """
        Variables:
        <ann_graph_path>: path to AnnoyIndex graph (Approximate Neares Neighbors)
        see: https://github.com/spotify/annoy
        <num_emb_batches>: number of batches of word embeddings
        <emb_type>: 'hsm', 'glove', 'multisense'
        <emb_size>: embedding dimension size
        """
        self.red_words = set()
        self.blue_words = set()
        self.weighted_nn = dict()
        # maps codewords' (from 'codewords.txt') line index to word
        self.idx_to_word = dict()
        self.num_emb_batches = num_emb_batches

        self.emb_size = emb_size

        ### HSM ###
        #self.word_senses_map = process_hsm.load_hsm_senses('../data/vocab.mat', '../data/wordreps.mat')
        #self.embeddings, self.hsm_idx_to_word, self.word_to_indices = process_hsm.map_to_array(self.word_senses_map)

        ### glove ###
        # glove_idx_to_word is a map of the index on the annoy graph to the word (added to the graph in the same order that they are taken from the pretrained txt files
        #self.embeddings, self.glove_idx_to_word = utils.get_glove_emb_from_txt(data_path = 'data/glove.6B.50d.txt', emb_size=self.emb_size)


        #self.graph = self.build_graph(emb_type=emb_type, embeddings=self.embeddings, num_trees = 100, metric = 'angular')
        #self.graph.save('glove.ann')

        self.graph = AnnoyIndex(self.emb_size)
        self.graph.load('../window5_lneighbor5e-2.ann')
        print("Built Annoy Graph")

        self._build_game()

        #self.vocab_map, self.inv_map = knn.build_id2word(vocab_map_fn='../vocab_dict.txt')


    def _build_game(self):
        self._generate_board()
        # for now randomly generate knn
        words = self.blue_words.union(self.red_words)
        #print(words)
        for word in words:
            # e.g. for word = "spoon",   weighted_nns[word] = {'fork':30, 'knife':25}
            #self.weighted_nn[word] = self.get_fake_knn(word)
            self.weighted_nn[word] = self.get_hsm_knn(word)
            #self.weighted_nn[word] = self.get_glove_knn(word)

    def _generate_board(self):
        # for now let's just set 5 red words and 5 blue words

        with open('codewords.txt') as file:
            for i, line in enumerate(file):
                word = line.strip().lower()
                self.idx_to_word[i] = word

        rand_idxs = random.sample(range(0, len(self.idx_to_word.keys())), 10)

        self.red_words = set([self.idx_to_word[idx] for idx in rand_idxs[:5]])
        self.blue_words = set([self.idx_to_word[idx] for idx in rand_idxs[5:]])

        self.red_words= set(["dog", "cat", "road", "star", "planet"])
        self.blue_words = set(["carpet", "bracelet", "book", "window", "lamp"])
        #self.blue_words = set( ["remote", "computer", "phone", "glass", "pillow"])




    def build_graph(self, emb_type = 'custom', embeddings = None, num_trees = 50, metric = 'angular'):
        if emb_type == 'hsm' or emb_type == 'glove':
            tree = knn.build_tree(self.num_emb_batches, input_type = emb_type, num_trees=num_trees, emb_size=self.emb_size,
                                  embeddings = embeddings, metric =metric)
        else:
            tree = knn.build_tree(self.num_emb_batches, num_trees = num_trees, emb_size = self.emb_size,
                              emb_dir = 'test_set_embeddings', metric = metric)
        return tree


    def get_clue(self, n, penalty):
        # where blue words are our team's words and red words are the other team's words
        # potential clue candidates are the intersection of weighted_nns[word] for each word in blue_words
        # we need to repeat this for the (|blue_words| C n) possible words we can give a clue for

        pq = []
        for word_set in itertools.combinations(self.blue_words, n):
            highest_clue, score = self.get_highest_clue(word_set, penalty)
            heapq.heappush(pq, (score, highest_clue))

        return heapq.heappop(pq)

    def get_highest_clue(self, chosen_words, penalty=1.0):

        potential_clues = set()
        for word in chosen_words:
            nns = self.weighted_nn[word]
            potential_clues.update(nns)

        highest_scoring_clue = None
        highest_score = float('-inf')

        for clue in potential_clues:
            blue_word_counts = [self.weighted_nn[blue_word][clue] for blue_word in self.blue_words if clue in self.weighted_nn[blue_word]]
            red_word_counts = [self.weighted_nn[red_word][clue] for red_word in self.red_words if clue in self.weighted_nn[red_word]]
            score = sum(blue_word_counts) - penalty * sum(red_word_counts)
            if score > highest_score:
                highest_scoring_clue = clue
                highest_score = score
        return highest_scoring_clue, highest_score


    def choose_words(self, n, clue, remaining_words):
        # given a clue word, choose the n words from remaining_words that most relates to the clue

        pq = []
        for word in remaining_words:
            score = self.get_score(clue, word)
            heapq.heappush(pq, (-1*score, word))

        ret = []
        for i in range(n):
            ret.append(heapq.heappop(pq))
        return ret



    def get_score(self, clue, word):
        '''
        :param clue: string
        :param possible_words: n-tuple of strings
        :return: score = sum(weighted_nn[possible_word][clue] for possible_word in possible_words)
        '''
        return self.weighted_nn[word][clue] if clue in self.weighted_nn[word] else 0

if __name__=='__main__':
    for i in range(5):
        game = Game()
        print("")
        print ("TRIAL ", str(i), ":")
        print("RED WORDS: ", list(game.red_words))

        print("BLUE WORDS: ", list(game.blue_words))

        print("NEAREST NEIGHBORS:")
        pprint.pprint(game.weighted_nn)


        score, clue = game.get_clue(2, 1)
        print("")
        print("CLUE CHOSEN: ", clue)

        print("WORDS CHOSEN FOR CLUE: ", game.choose_words(2, clue, game.blue_words.union(game.red_words)))

    #print(game.get_hsm_knn('star', 10))
