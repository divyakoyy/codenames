import heapq
import itertools
import os
import random
import re
import string
import sys
import urllib
import string
import argparse
import gzip
import pickle
import math
from datetime import datetime
import csv

# Gensim
from gensim.corpora import Dictionary
import gensim.downloader as api

# nltk
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer

import networkx as nx
import numpy as np
import requests
from tqdm import tqdm

# Embeddings
from embeddings.babelnet import Babelnet
from embeddings.word2vec import Word2Vec
from embeddings.glove import Glove
from embeddings.fasttext import FastText
from embeddings.bert import Bert
from embeddings.kim2019 import Kim2019

from utils import get_dict2vec_score

sys.path.insert(0, "../")

stopwords = [
    'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than', 'get', 'put',
]

idf_lower_bound = 0.0006
default_single_word_label_scores = (1, 1.1, 1.1, 1.2)

"""
Configuration for running the game
"""
class CodenamesConfiguration(object):
    def __init__(
        self,
        verbose=False,
        visualize=False,
        split_multi_word=True,
        disable_verb_split=True,
        debug_file=None,
        length_exp_scaling=None,
        use_heuristics=True,
        single_word_label_scores=default_single_word_label_scores,
        use_kim_scoring_function=False,
        babelnet_api_key=None,
    ):
        self.verbose = verbose
        self.visualize = visualize
        self.split_multi_word = split_multi_word
        self.disable_verb_split = disable_verb_split
        self.debug_file = debug_file
        self.length_exp_scaling = length_exp_scaling
        self.use_heuristics = use_heuristics
        self.single_word_label_scores = tuple(single_word_label_scores)
        self.use_kim_scoring_function = use_kim_scoring_function
        self.babelnet_api_key = babelnet_api_key

    def description(self):
        return (
            "<verbose: " + str(self.verbose) +
            ",visualize: " + str(self.visualize) +
            ",split multi-word clues: " + str(self.split_multi_word) +
            ",disable verb split: " + str(self.disable_verb_split) +
            ",length exp scaling: " + str(self.length_exp_scaling) +
            ",use heuristics: " + str(self.use_heuristics) +
            ",use kim scoring function: " + str(self.use_kim_scoring_function) +
            ">"
        )

class Codenames(object):

    def __init__(
        self,
        embedding_type="custom",
        configuration=None
    ):
        """
        :param embedding_type: e.g.'word2vec', 'glove', 'fasttext', 'babelnet'
        :param embedding: an embedding object that codenames will use to play

        """
        # Intialize variables
        if configuration != None:
            self.configuration = configuration
        else:
            self.configuration = CodenamesConfiguration()
        print("Codenames Configuration: ", self.configuration.__dict__)
        with open('data/word_to_dict2vec_embeddings', 'rb') as word_to_dict2vec_embeddings_file:
            self.word_to_dict2vec_embeddings = pickle.load(word_to_dict2vec_embeddings_file)
        self.embedding_type = embedding_type
        self.embedding = self._get_embedding_from_type(embedding_type)
        self.weighted_nn = dict()

        self.num_docs, self.word_to_df = self._load_document_frequencies()  # dictionary of word to document frequency

        # Used to get word stems
        self.stemmer = PorterStemmer()

    """
    Codenames game setup
    """

    def _get_embedding_from_type(self, embedding_type):
        """
        :param embedding_type: 'babelnet', 'word2vec', glove', 'fasttext'
        returns the embedding object that will be used to play

        """
        print("Building game for ", embedding_type, "...")
        if embedding_type == 'babelnet':
            return Babelnet(self.configuration)
        elif embedding_type == 'word2vec':
            return Word2Vec(self.configuration)
        elif embedding_type == 'glove':
            return Glove(self.configuration)
        elif embedding_type == 'fasttext':
            return FastText(self.configuration)
        elif embedding_type == 'bert':
            return Bert(self.configuration)
        elif embedding_type == 'kim2019':
            return Kim2019(self.configuration, self.word_to_dict2vec_embeddings)
        else:
            print("Valid embedding types are babelnet, word2vec, glove, fasttext, and bert")

        return None

    def _build_game(self, red=None, blue=None, save_path=None):
        """
        :param red: optional list of strings of opponent team's words
        :param blue: optional list of strings of our team's words
        :param save_path: optional directory path to save data between games
        :return: None
        """
        self._generate_board_words(red, blue)
        self.save_path = save_path
        self.weighted_nn = dict()

        words = self.blue_words.union(self.red_words)
        for word in words:
            self.weighted_nn[word] = self.embedding.get_weighted_nn(word)

        self._write_to_debug_file(["\n", "Building game with configuration:", self.configuration.description(), "\n\tBLUE words: ", " ".join(self.blue_words), "RED words:", " ".join(self.red_words), "\n"])

    def _generate_board_words(self, red=None, blue=None):
        """
        :param red: optional list of strings of opponent team's words
        :param blue: optional list of strings of our team's words
        :return: None
        """
        idx_to_word = dict()

        with open("data/codewords.txt") as file:
            for i, line in enumerate(file):
                word = line.strip().lower()
                idx_to_word[i] = word

        rand_idxs = random.sample(range(0, len(idx_to_word.keys())), 10)

        self.red_words = set([idx_to_word[idx] for idx in rand_idxs[:5]])
        self.blue_words = set([idx_to_word[idx] for idx in rand_idxs[5:]])

        if red is not None:
            self.red_words = set(red)
        if blue is not None:
            self.blue_words = set(blue)

    def _load_document_frequencies(self):
        """
        Sets up a dictionary from words to their document frequency
        """
        if (os.path.exists("data/word_to_df.pkl")) and (os.path.exists("data/text8_num_documents.txt")):
            with open('data/word_to_df.pkl', 'rb') as f:
                word_to_df = pickle.load(f)
            with open('data/text8_num_documents.txt', 'rb') as f:
                for line in f:
                    num_docs = int(line.strip())
                    break
        else:
            dataset = api.load("text8")
            dct = Dictionary(dataset)
            id_to_doc_freqs = dct.dfs
            num_docs = dct.num_docs
            word_to_df = {dct[id]: id_to_doc_freqs[id]
                          for id in id_to_doc_freqs}

        return num_docs, word_to_df


    def _write_to_debug_file(self, lst):
        if self.configuration.debug_file:
            with open(self.configuration.debug_file, 'a') as f:
                f.write(" ".join([str(x) for x in lst]))

    '''
    Codenames game methods
    '''

    def get_clue(self, n, penalty):
        # where blue words are our team's words and red words are the other team's words
        # potential clue candidates are the intersection of weighted_nns[word] for each word in blue_words
        # we need to repeat this for the (|blue_words| C n) possible words we can give a clue for

        pq = []
        for word_set in itertools.combinations(self.blue_words, n):
            highest_clues, score = self.get_highest_clue(
                word_set, penalty)
            # min heap, so push negative score
            heapq.heappush(pq, (-1 * score, highest_clues, word_set))

        # sliced_labels = self.get_cached_labels_from_synset(clue)
        # main_sense, _senses = self.get_cached_labels_from_synset_v5(clue)

        best_clues = []
        best_board_words_for_clue = []
        best_scores = []
        count = 0

        while pq:
            score, clues, word_set = heapq.heappop(pq)

            if count >= 5:
                break

            if self.configuration.visualize and callable(getattr(self.embedding, "get_intersecting_graphs", None)):
                for clue in clues:
                    self.embedding.get_intersecting_graphs(
                        word_set,
                        clue,
                        split_multi_word=self.configuration.split_multi_word,
                    )

            best_clues.append(clues)
            best_scores.append(score)
            best_board_words_for_clue.append(word_set)

            count += 1

        return best_scores, best_clues, best_board_words_for_clue

    def is_valid_clue(self, clue):
        # no need to remove red/blue words from potential_clues elsewhere
        # since we check for validity here
        for board_word in self.red_words.union(self.blue_words):
            # Check if clue or board_word are substring of each other, or if they share the same word stem
            if (clue in board_word or board_word in clue or self.stemmer.stem(clue) == self.stemmer.stem(board_word) or not clue.isalpha()):
                return False
        return True

    def get_highest_clue(self, chosen_words, penalty=1.0):

        if self.embedding_type == 'kim2019':
            chosen_clue, dist = self.embedding.get_clue(
                self.blue_words, self.red_words, chosen_words)
            # return the angular similarity
            return [chosen_clue], 1 - dist

        potential_clues = set()
        for word in chosen_words:
            nns = self.weighted_nn[word]
            potential_clues.update(nns)

        highest_scoring_clues = []
        highest_score = float("-inf")

        for clue in potential_clues:
            # don't consider clues which are a substring of any board words
            if not self.is_valid_clue(clue):
                continue
            blue_word_counts = []
            for blue_word in chosen_words:
                if clue in self.weighted_nn[blue_word]:
                    blue_word_counts.append(self.weighted_nn[blue_word][clue])
                else:
                    blue_word_counts.append(self.embedding.get_word_similarity(blue_word, clue))

            heuristic_score = 0

            self._write_to_debug_file([
                "\n", clue, "score breakdown for", " ".join(chosen_words),
                "\n\tblue words score:", round(sum(blue_word_counts),3),
            ])

            if self.configuration.use_heuristics is True:
                # the larger the idf is, the more uncommon the word
                idf = (1.0/self.word_to_df[clue]) if clue in self.word_to_df else 1.0

                # prune out super common words (e.g. "get", "go")
                if (clue in stopwords or idf < idf_lower_bound):
                    idf = 1.0
                dict2vec_weight = self.embedding.dict2vec_embedding_weight()
                dict2vec_score = dict2vec_weight*get_dict2vec_score(chosen_words, clue, self.red_words)

                heuristic_score = dict2vec_score + (-2*idf)
                self._write_to_debug_file([" IDF:", round(-2*idf,3), "dict2vec score:", round(dict2vec_score,3)])

            # Give embedding methods the opportunity to rescale the score using their own heuristics
            embedding_score = self.embedding.rescale_score(chosen_words, clue, self.red_words)

            if (self.configuration.use_kim_scoring_function):
                score = min(blue_word_counts) + heuristic_score
            else:
                score = sum(blue_word_counts) + embedding_score + heuristic_score

            if score > highest_score:
                highest_scoring_clues = [clue]
                highest_score = score
            elif score == highest_score:
                highest_scoring_clues.append(clue)

        return highest_scoring_clues, highest_score

    def choose_words(self, n, clue, remaining_words):
        # given a clue word, choose the n words from remaining_words that most relates to the clue

        pq = []

        for word in remaining_words:
            score = self.get_score(clue, word)
            # min heap, so push negative score
            heapq.heappush(pq, (-1 * score, word))

        ret = []
        for i in range(n):
            ret.append(heapq.heappop(pq))
        return ret

    def get_score(self, clue, word):
        """
        :param clue: string
        :param possible_words: n-tuple of strings
        :return: score = sum(weighted_nn[possible_word][clue] for possible_word in possible_words)
        """
        if clue in self.weighted_nn[word]:
            return self.weighted_nn[word][clue]
        else:
            try:
                return self.embedding.get_word_similarity(word, clue)
            except KeyError:
                return -1000


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('embeddings', nargs='+',
                        help='an embedding method to use when playing codenames')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='print out verbose information'),
    parser.add_argument('--visualize', dest='visualize', action='store_true',
                        help='visualize the choice of clues with graphs')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='Write score breakdown to a file. You can specify what file is used with --debug-file, or one will be created for you')
    parser.add_argument('--no-heuristics', dest='no_heuristics', action='store_true',
                        help='Remove heuristics such as IDF and dict2vec')
    parser.add_argument('--debug-file', dest='debug_file', default=None,
                        help='Write score breakdown to debug file')
    parser.add_argument('--num-trials', type=int, dest='num_trials', default=1,
                        help='number of trials of the game to run')
    parser.add_argument('--split-multi-word', dest='split_multi_word', default=True)
    parser.add_argument('--disable-verb-split', dest='disable_verb_split', default=True)
    parser.add_argument('--kim-scoring-function', dest='use_kim_scoring_function', action='store_true',
                        help='use the kim 2019 et. al. scoring function'),
    parser.add_argument('--length-exp-scaling', type=int, dest='length_exp_scaling', default=None,
                        help='Rescale lengths using exponent')
    parser.add_argument('--single-word-label-scores', type=float, nargs=4, dest='single_word_label_scores',
                        default=default_single_word_label_scores,
                        help='main_single, main_multi, other_single, other_multi scores')
    parser.add_argument('--babelnet-api-key', type=str, dest='babelnet_api_key', default=None)
    args = parser.parse_args()


    words = [
        'vacuum', 'whip', 'moon', 'school', 'tube', 'lab', 'key', 'table', 'lead', 'crown',
        'bomb', 'bug', 'pipe', 'roulette','australia', 'play', 'cloak', 'piano', 'beijing', 'bison',
        'boot', 'cap', 'car','change', 'circle', 'cliff', 'conductor', 'cricket', 'death', 'diamond',
        'figure', 'gas', 'germany', 'india', 'jupiter', 'kid', 'king', 'lemon', 'litter', 'nut',
        'phoenix', 'racket', 'row', 'scientist', 'shark', 'stream', 'swing', 'unicorn', 'witch', 'worm',
        'pistol', 'saturn', 'rock', 'superhero', 'mug', 'fighter', 'embassy', 'cell', 'state', 'beach',
        'capital', 'post', 'cast', 'soul', 'tower', 'green', 'plot', 'string', 'kangaroo', 'lawyer', 'fire',
        'robot', 'mammoth', 'hole', 'spider', 'bill', 'ivory', 'giant', 'bar', 'ray', 'drill', 'staff',
        'greece', 'press','pitch', 'nurse', 'contract', 'water', 'watch', 'amazon','spell', 'kiwi', 'ghost',
        'cold', 'doctor', 'port', 'bark','foot', 'luck', 'nail', 'ice', 'needle', 'disease', 'comic', 'pool',
        'field', 'star', 'cycle', 'shadow', 'fan', 'compound', 'heart', 'flute','millionaire', 'pyramid', 'africa',
        'robin', 'chest', 'casino','fish', 'oil', 'alps', 'brush', 'march', 'mint','dance', 'snowman', 'torch',
        'round', 'wake', 'satellite','calf', 'head', 'ground', 'club', 'ruler', 'tie','parachute', 'board',
        'paste', 'lock', 'knight', 'pit', 'fork', 'egypt', 'whale', 'scale', 'knife', 'plate','scorpion', 'bottle',
        'boom', 'bolt', 'fall', 'draft', 'hotel', 'game', 'mount', 'train', 'air', 'turkey', 'root', 'charge',
        'space', 'cat', 'olive', 'mouse', 'ham', 'washer', 'pound', 'fly', 'server','shop', 'engine', 'himalayas',
        'box', 'antarctica', 'shoe', 'tap', 'cross', 'rose', 'belt', 'thumb', 'gold', 'point', 'opera', 'pirate',
        'tag', 'olympus', 'cotton', 'glove', 'sink', 'carrot', 'jack', 'suit', 'glass', 'spot', 'straw', 'well',
        'pan', 'octopus', 'smuggler', 'grass', 'dwarf', 'hood', 'duck', 'jet', 'mercury',
    ]

    red_words = []
    blue_words = []

    for _ in range(0, args.num_trials):
        random.shuffle(words)
        red_words.append(words[:10])
        blue_words.append(words[10:20])

    amt_file_path = 'amt_102620_all_kim_scoring_fx.csv'
    amt_key_file_path = 'amt_102620_all_kim_scoring_fx_key.csv'
    # Setup CSVs
    if not os.path.exists(amt_file_path):
        with open(amt_file_path, 'w'): pass

    with open(amt_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header_row = ['embedding_name', 'clue'] + ["word" + str(x) for x in range(0,20)]
        writer.writerow(header_row)

    if not os.path.exists(amt_key_file_path):
        with open(amt_key_file_path, 'w'): pass

    with open(amt_key_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header_row = ['embedding_name', 'configuration','clue', 'word0ForClue', 'word1ForClue'] + ["blueWord" + str(x) for x in range(0,10)] + ["redWord" + str(x) for x in range(0,10)]
        writer.writerow(header_row)

    embedding_trial_to_clues = dict()

    shuffled_embeddings = args.embeddings
    random.shuffle(shuffled_embeddings)

    for embedding_type in shuffled_embeddings:
        embedding_trial_number = 0
        debug_file_path = None
        if args.debug is True or args.debug_file != None:
            debug_file_path = (embedding_type + "-" + datetime.now().strftime("%m-%d-%Y-%H.%M.%S") + ".txt") if args.debug_file == None else args.debug_file
            # Create directory to put debug files if it doesn't exist
            if not os.path.exists('debug_output'):
                os.makedirs('debug_output')
            debug_file_path = os.path.join('debug_output', debug_file_path)
            print("Writing debug output to", debug_file_path)

        configuration = CodenamesConfiguration(
            verbose=args.verbose,
            visualize=args.visualize,
            split_multi_word=args.split_multi_word,
            disable_verb_split=args.disable_verb_split,
            debug_file=debug_file_path,
            length_exp_scaling=args.length_exp_scaling,
            use_heuristics=(not args.no_heuristics),
            single_word_label_scores=args.single_word_label_scores,
            use_kim_scoring_function=args.use_kim_scoring_function,
            babelnet_api_key=args.babelnet_api_key,
        )

        game = Codenames(
            configuration=configuration,
            embedding_type=embedding_type,
        )

        for i, (red, blue) in enumerate(zip(red_words, blue_words)):

            game._build_game(red=red, blue=blue,
                             save_path="tmp_babelnet_" + str(i))
            if game.configuration.verbose:
                print("NEAREST NEIGHBORS:")
                for word, clues in game.weighted_nn.items():
                    print(word)
                    print(sorted(clues, key=lambda k: clues[k], reverse=True)[:5])

            best_scores, best_clues, best_board_words_for_clue = game.get_clue(2, 1)

            print("==================================================================================================================")
            print("TRIAL", str(i+1))
            print("RED WORDS: ", list(game.red_words))
            print("BLUE WORDS: ", list(game.blue_words))
            print("BEST CLUES: ")
            for score, clues, board_words in zip(best_scores, best_clues, best_board_words_for_clue):
                print()
                print("Clue(s):", ", ".join(clues), "|| Intended board words:", board_words, "|| Score:", str(round(score,3)))

            # Write to CSV
            heuristic_string = "WithHeuristics" if configuration.use_heuristics else "WithoutHeuristics"
            kim_scoring_fx_string = "KimFx" if configuration.use_kim_scoring_function else "WithoutKimFx"
            embedding_with_trial_number = embedding_type +  heuristic_string + kim_scoring_fx_string + "Trial" + str(embedding_trial_number)

            # Check if this clue has already been chosen
            embedding_number = embedding_type + str(embedding_trial_number)
            clue = best_clues[0][0]
            is_duplicate_clue = embedding_number in embedding_trial_to_clues and clue in embedding_trial_to_clues[embedding_number]
            if (is_duplicate_clue is False):
                if embedding_number not in embedding_trial_to_clues:
                    embedding_trial_to_clues[embedding_number] = set()
                embedding_trial_to_clues[embedding_number].add(clue)

                with open(amt_file_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([embedding_with_trial_number, clue] + list(game.blue_words.union(game.red_words)))

            with open(amt_key_file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([embedding_with_trial_number, str(configuration.__dict__), clue, best_board_words_for_clue[0][0], best_board_words_for_clue[0][1]] + list(game.blue_words) + list(game.red_words))

            embedding_trial_number += 1

            # Draw graphs for all words
            # all_words = red + blue
            # for word in all_words:
            #     game.draw_graph(game.graphs[word], word+"_all", get_labels=True)
