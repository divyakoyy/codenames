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
from datetime import datetime

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

sys.path.insert(0, "../")

# TODO Anna: Remove this?
blacklist = set([
    "s00081546n",  # term
    "s00021547n",  # concept
    "s00026969n",  # dictionary entry
    "s00058442n",  # object
    "s00020461n",  # semasiology
    "s00045800n",  # idea
    "s00050906n",  # lexicon
    "s00061984n",  # philosophy
    "s00081546n"   # word
])

stopwords = [
    'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than', 'get', 'put',
    'class', 'family'
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
    ):
        self.verbose = verbose
        self.visualize = visualize
        self.split_multi_word = split_multi_word
        self.disable_verb_split = disable_verb_split
        self.debug_file = debug_file
        self.length_exp_scaling = length_exp_scaling
        self.use_heuristics = use_heuristics
        self.single_word_label_scores = tuple(single_word_label_scores)

    def description(self):
        return "<verbose: "+str(self.verbose)+",visualize: "+str(self.visualize)+",split multi-word clues: "+str(self.split_multi_word)+",disable verb split: "+str(self.disable_verb_split)+",length exp scaling: "+str(self.length_exp_scaling)+",use heuristics: "+str(self.use_heuristics)+">"

class Codenames(object):

    def __init__(
        self,
        embedding_type="custom",
        configuration=None
    ):
        """
        # TODO: Clean up this documenation
        :param ann_graph_path: path to AnnoyIndex graph (Approximate Neares Neighbors)
        see: https://github.com/spotify/annoy
        :param num_emb_batches: number of batches of word embeddings
        :param embedding_type: e.g.'word2vec', 'glove', 'fasttext', 'babelnet'
        :param embedding: an embedding object that codenames will use to play

        """
        # Intialize variables
        if configuration != None:
            self.configuration = configuration
        else:
            self.configuration = CodenamesConfiguration()
        print(self.configuration.__dict__)
        self.embedding_type = embedding_type #TODO: remove this after the custom domain choosing from get_highest_clue is out
        self.embedding = self._get_embedding_from_type(embedding_type)
        self.weighted_nn = dict()

        self.word_to_df = self._get_df()  # dictionary of word to document frequency
        self.dict2vec_embeddings_file = 'data/word_to_dict2vec_embeddings'
        self.word_to_dict2vec_embeddings = self._get_dict2vec()

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

    def _get_df(self):
        """
        Sets up a dictionary from words to their document frequency
        """
        if (os.path.exists("data/word_to_df.pkl")):
            with open('data/word_to_df.pkl', 'rb') as f:
                word_to_df = pickle.load(f)
        else:
            dataset = api.load("text8")
            dct = Dictionary(dataset)
            id_to_doc_freqs = dct.dfs
            word_to_df = {dct[id]: id_to_doc_freqs[id]
                          for id in id_to_doc_freqs}

        return word_to_df

    def _get_dict2vec(self):
        input_file = open(self.dict2vec_embeddings_file,'rb')
        word_to_dict2vec_embeddings = pickle.load(input_file)
        return word_to_dict2vec_embeddings

    def _get_dict2vec_score(self, chosen_words, potential_clue, red_words):
        dict2vec_similarities = []
        red_dict2vec_similarities = []

        if potential_clue not in self.word_to_dict2vec_embeddings:
            if self.configuration.verbose:
                print("Potential clue word ", potential_clue, "not in dict2vec model")
            return 0.0

        potential_clue_embedding = self.word_to_dict2vec_embeddings[potential_clue]
        # TODO: change this to cosine distance
        for chosen_word in chosen_words:
            if chosen_word in self.word_to_dict2vec_embeddings:
                chosen_word_embedding = self.word_to_dict2vec_embeddings[chosen_word]
                euclidean_distance = np.linalg.norm(chosen_word_embedding-potential_clue_embedding)
                dict2vec_similarities.append(euclidean_distance)

        for red_word in red_words:
            if red_word in self.word_to_dict2vec_embeddings:
                red_word_embedding = self.word_to_dict2vec_embeddings[red_word]
                red_euclidean_distance = np.linalg.norm(red_word_embedding-potential_clue_embedding)
                red_dict2vec_similarities.append(red_euclidean_distance)
        #TODO: is average the best way to do this
        return 1/(sum(dict2vec_similarities)/len(dict2vec_similarities)) - 1/(min(red_dict2vec_similarities))

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

    def rescale_domain_score(self, score):
        if score < 0:
            return score * -1 / 2.5
        else:
            return score

    def is_valid_clue(self, clue):
        for board_word in self.red_words.union(self.blue_words):
            # Check if clue or board_word are substring of each other, or if they share the same word stem
            if (clue in board_word or board_word in clue or self.stemmer.stem(clue) == self.stemmer.stem(board_word)):
                return False
        return True

    def get_highest_clue(self, chosen_words, penalty=1.0, domain_threshold=0.45, domain_gap=0.3):

        potential_clues = set()
        for word in chosen_words:
            nns = self.weighted_nn[word]
            potential_clues.update(nns)

        # TODO : Instead of this override behavior, add domains to nn_w_dist
        # try to get a domain clue
        if self.embedding_type =='babelnet':
            domains = {}
            for word in set(chosen_words).union(self.red_words):
                for domain, score in self.embedding.nn_to_domain_label[word].items():
                    score = self.rescale_domain_score(score)
                    if score > domain_threshold:
                        if domain not in domains:
                            domains[domain] = dict()
                        domains[domain][word] = score
            domain_clues = []
            for domain, word_scores in domains.items():
                if all([word in word_scores for word in chosen_words]):
                    # get min word_set domain score
                    min_chosen_words_score = min(
                        [word_scores[word] for word in chosen_words])
                    # get max red_words domain score
                    red_words_domain_scores = [
                        word_scores[word] for word in self.red_words.intersection(word_scores.keys())]
                    if len(red_words_domain_scores) == 0:
                        max_red_words_score = 0
                    else:
                        max_red_words_score = max(
                            red_words_domain_scores) + domain_gap

                    if min_chosen_words_score > max_red_words_score:
                        domain_clues.append(domain)
            if len(domain_clues) >= 1:
                return domain_clues, 1  # TODO: return different score?

        potential_clues = potential_clues - \
            self.blue_words.union(self.red_words)

        highest_scoring_clues = []
        highest_score = float("-inf")

        for clue in potential_clues:
            # don't consider clues which are a substring of any board words
            if not self.is_valid_clue(clue):
                continue
            blue_word_counts = []
            red_word_counts = []
            for blue_word in chosen_words:
                if clue in self.weighted_nn[blue_word]:
                    blue_word_counts.append(self.weighted_nn[blue_word][clue])
                else:
                    blue_word_counts.append(-1.0)
            for red_word in self.red_words:
                if clue in self.weighted_nn[red_word]:
                    red_word_counts.append(self.weighted_nn[red_word][clue])

            heuristic_score = 0

            self._write_to_debug_file(["\n", clue, "score breakdown for", " ".join(chosen_words), "\n\tblue words score:", round(sum(blue_word_counts),3), " red words penalty:", round((penalty *sum(red_word_counts)),3)])

            if self.configuration.use_heuristics is True:
                # the larger the idf is, the more uncommon the word
                idf = (1.0/self.word_to_df[clue]) if clue in self.word_to_df else 1.0

                # prune out super common words (e.g. "get", "go")
                if (clue in stopwords or idf < idf_lower_bound):
                    idf = 1.0

                dict2vec_score = 10*self._get_dict2vec_score(chosen_words, clue, self.red_words)

                heuristic_score = dict2vec_score + (-2*idf)
                self._write_to_debug_file([" IDF:", round(-2*idf,3), "dict2vec score:", round(dict2vec_score,3)])

            # Give embedding methods the opportunity to rescale the score using their own heuristics
            embedding_score = self.embedding.rescale_score(chosen_words, clue, self.red_words)

            score = sum(blue_word_counts) - (penalty *sum(red_word_counts)) + embedding_score + heuristic_score

            if score > highest_score:
                highest_scoring_clues = [clue]
                highest_score = score
            elif score == highest_score:
                highest_scoring_clues.append(clue)

        return highest_scoring_clues, highest_score

    def choose_words(self, n, clue, remaining_words, domain_threshold=0.45):
        # given a clue word, choose the n words from remaining_words that most relates to the clue

        pq = []

        # TODO : Instead of this override behavior, add domains to nn_w_dist
        # try to get a domain clue
        if self.embedding_type =='babelnet':
            domain_words = []
            for word in remaining_words:
                if clue in self.embedding.nn_to_domain_label[word]:
                    score = self.rescale_domain_score(
                        self.embedding.nn_to_domain_label[word][clue])
                    if score > domain_threshold:
                        domain_words.append((word, score))
            if len(domain_words) >= n:
                # This is a domain clue, choose domain words
                return domain_words

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
        return self.weighted_nn[word][clue] if clue in self.weighted_nn[word] else -1000


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
    parser.add_argument('--length-exp-scaling', type=int, dest='length_exp_scaling', default=None,
                        help='Rescale lengths using exponent')
    parser.add_argument('--single-word-label-scores', type=float, nargs=4, dest='single_word_label_scores',
                        default=default_single_word_label_scores,
                        help='main_single, main_multi, other_single, other_multi scores')
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

    for useHeuristicOverride in [True, False]:
        for embedding_type in args.embeddings:
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
                use_heuristics=useHeuristicOverride,
                single_word_label_scores=args.single_word_label_scores,
            )

            game = Codenames(
                configuration=configuration,
                embedding_type=embedding_type,
            )

            for i, (red, blue) in enumerate(zip(red_words, blue_words)):

                game._build_game(red=red, blue=blue,
                                 save_path="tmp_babelnet_" + str(i))
                # TODO: Download version without using aliases. They may be too confusing
                if game.configuration.verbose:
                    print("NEAREST NEIGHBORS:")
                    for word, clues in game.weighted_nn.items():
                        print(word)
                        print(sorted(clues, key=lambda k: clues[k], reverse=True)[:5])

                best_scores, best_clues, best_board_words_for_clue = game.get_clue(2, 1)

                print("===================================================================================================")
                print("TRIAL", str(i+1))
                print("RED WORDS: ", list(game.red_words))
                print("BLUE WORDS: ", list(game.blue_words))
                print("BEST CLUES: ")
                for score, clues, board_words in zip(best_scores, best_clues, best_board_words_for_clue):
                    print()
                    print(clues, str(round(score,3)), board_words)
                    for clue in clues:
                        print(
                            "WORDS CHOSEN FOR CLUE: ",
                            game.choose_words(
                                2, clue, game.blue_words.union(game.red_words)),
                        )


                # Draw graphs for all words
                # all_words = red + blue
                # for word in all_words:
                #     game.draw_graph(game.graphs[word], word+"_all", get_labels=True)
