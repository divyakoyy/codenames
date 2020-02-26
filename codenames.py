import heapq
import itertools
import operator
import pprint
import random
import sys

import numpy as np
from annoy import AnnoyIndex
from nltk.corpus import wordnet as wn

from tqdm import tqdm

sys.path.insert(0, "../")


class Game(object):
    def __init__(
        self, ann_graph_path=None, num_emb_batches=22, emb_type="custom", emb_size=200
    ):
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

        # pre-process
        # data from: https://github.com/uhh-lt/path2vec#pre-trained-models-and-datasets
        # self.get_path2vec_emb_from_txt(data_path='data/jcn-semcor_embeddings.vec')
        self.lemma_nns = self.get_wordnet_nns()

        self._build_game()

# GAME SET-UP

    def _build_game(self, red=None, blue=None):
        self._generate_board(red, blue)
        # for now randomly generate knn
        words = self.blue_words.union(self.red_words)
        # print(words)
        for word in words:
            # e.g. for word = "spoon",   weighted_nns[word] = {'fork':30, 'knife':25}
            # self.weighted_nn[word] = self.get_fake_knn(word)
            # self.weighted_nn[word] = self.get_path2vec_knn(word)
            self.weighted_nn[word] = self.get_wordnet_knn(word)

    def add_lemmas(self, d, ss, hyper, n):
        for lemma_name in ss.lemma_names():
            if lemma_name not in d:
                d[lemma_name] = {}
            for neighbor in ss.lemmas() + hyper.lemmas():
                if neighbor not in d[lemma_name]:
                    d[lemma_name][neighbor] = float("inf")
                d[lemma_name][neighbor] = min(d[lemma_name][neighbor], n)

    def get_wordnet_nns(self):
        d_lemmas = {}
        for ss in tqdm(wn.all_synsets(pos="n")):
            self.add_lemmas(d_lemmas, ss, ss, 0)
            # get the transitive closure of all hypernyms of a synset
            # hypernyms = categories of
            for i, hyper in enumerate(ss.closure(lambda s: s.hypernyms())):
                self.add_lemmas(d_lemmas, ss, hyper, i + 1)

            # also write transitive closure for all instances of a synset
            # hyponyms = types of
            for instance in ss.instance_hyponyms():
                for i, hyper in enumerate(
                    instance.closure(lambda s: s.instance_hypernyms())
                ):
                    self.add_lemmas(d_lemmas, instance, hyper, i + 1)
                    for j, h in enumerate(hyper.closure(lambda s: s.hypernyms())):
                        self.add_lemmas(d_lemmas, instance, h, i + 1 + j + 1)
        return d_lemmas

    def get_wordnet_knn(self, word):
        if word not in self.lemma_nns:
            return {}
        return {
            k.name(): 1.0 / (v + 1)
            for k, v in self.lemma_nns[word].items()
            if k.name() != word
        }

    def get_path2vec_emb_from_txt(self, data_path):
        # map lemmas to synsets
        lemma_synsets = dict()
        for ss in tqdm(wn.all_synsets(pos="n")):
            for lemma_name in ss.lemma_names():
                if lemma_name not in lemma_synsets:
                    lemma_synsets[lemma_name] = set()
                lemma_synsets[lemma_name].add(ss)
        self.lemma_synsets = lemma_synsets

        synset_to_idx = {}
        idx_to_synset = {}
        with open(data_path, "r") as f:
            line = next(f)
            vocab_size, emb_size = line.split(" ")
            emb_size = int(emb_size)
            tree = AnnoyIndex(emb_size, metric="angular")
            for i, line in enumerate(f):
                parts = line.split(" ")
                synset_str = parts[0]
                emb_vector = np.array(parts[1:], dtype=float)
                if len(emb_vector) != emb_size:
                    print("unexpected emb vector size:", len(emb_vector))
                    continue
                synset_to_idx[synset_str] = i
                idx_to_synset[i] = synset_str
                tree.add_item(i, emb_vector)
        tree.build(100)
        self.graph = tree
        self.synset_to_idx = synset_to_idx
        self.idx_to_synset = idx_to_synset

    def get_path2vec_knn(self, word, nums_nns=250):
        if word not in self.lemma_synsets:
            return {}

        # get synset nns
        synsets = self.lemma_synsets[word]
        nn_w_dists = dict()
        for synset in synsets:
            id = self.synset_to_idx[synset.name()]
            nn_indices = set(self.graph.get_nns_by_item(id, nums_nns))
            nn_words = []
            for nn_id in nn_indices:
                ss = self.idx_to_synset[nn_id]
                # map synsets to lemmas
                try:
                    for lemma in wn.synset(ss).lemma_names():
                        if lemma not in nn_w_dists:
                            nn_w_dists[lemma] = self.graph.get_distance(id, nn_id)
                        nn_w_dists[lemma] = min(
                            self.graph.get_distance(id, nn_id), nn_w_dists[lemma]
                        )
                except ValueError:
                    print(ss, "not a valid synset")
        # return dict[nn] = score
        # we store multiple lemmas with same score,
        # because in the future we can downweight
        # lemmas that are closer to enemy words
        return nn_w_dists

    def _generate_board(self, red=None, blue=None):
        # for now let's just set 5 red words and 5 blue words

        with open("data/codewords.txt") as file:
            for i, line in enumerate(file):
                word = line.strip().lower()
                self.idx_to_word[i] = word

        rand_idxs = random.sample(range(0, len(self.idx_to_word.keys())), 10)

        self.red_words = set([self.idx_to_word[idx] for idx in rand_idxs[:5]])
        self.blue_words = set([self.idx_to_word[idx] for idx in rand_idxs[5:]])

        if red is not None:
            self.red_words = set(red)
        if blue is not None:
            self.blue_words = set(blue)
        # self.blue_words = set(["carpet", "bracelet", "book", "window", "lamp"])
        # self.red_words = set( ["remote", "computer", "phone", "glass", "pillow"])
        # self.blue_words= set(["dog", "cat", "road", "star", "planet"])

    def build_graph(
        self, emb_type="custom", embeddings=None, num_trees=50, metric="angular"
    ):
        if emb_type == "hsm" or emb_type == "glove":
            tree = knn.build_tree(
                self.num_emb_batches,
                input_type=emb_type,
                num_trees=num_trees,
                emb_size=self.emb_size,
                embeddings=embeddings,
                metric=metric,
            )
        else:
            tree = knn.build_tree(
                self.num_emb_batches,
                num_trees=num_trees,
                emb_size=self.emb_size,
                emb_dir="test_set_embeddings",
                metric=metric,
            )
        return tree

    def get_clue(self, n, penalty):
        # where blue words are our team's words and red words are the other team's words
        # potential clue candidates are the intersection of weighted_nns[word] for each word in blue_words
        # we need to repeat this for the (|blue_words| C n) possible words we can give a clue for

        pq = []
        for word_set in itertools.combinations(self.blue_words, n):
            highest_clue, score = self.get_highest_clue(word_set, penalty)
            # min heap, so push negative score
            heapq.heappush(pq, (-1*score, highest_clue))

        return heapq.heappop(pq)

    def get_highest_clue(self, chosen_words, penalty=1.0):

        potential_clues = set()
        for word in chosen_words:
            nns = self.weighted_nn[word]
            potential_clues.update(nns)
        potential_clues = potential_clues - self.blue_words.union(self.red_words)

        highest_scoring_clue = None
        highest_score = float("-inf")

        for clue in potential_clues:
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
            score = sum(blue_word_counts) - (penalty * sum(red_word_counts))
            if score > highest_score:
                highest_scoring_clue = clue
                # print("highest_scoring_clue:", highest_scoring_clue, score)
                highest_score = score
        return highest_scoring_clue, highest_score

    def choose_words(self, n, clue, remaining_words):
        # given a clue word, choose the n words from remaining_words that most relates to the clue

        pq = []
        for word in remaining_words:
            score = self.get_score(clue, word)
            # min heap, so push negative score
            heapq.heappush(pq, (-1*score, word))

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
    game = Game()
    # Use None to randomize the game, or pass in fixed lists
    red_words = [
        None, None, None, None, None
        # ["board", "web", "wave", "platypus", "mine"],
        # ["conductor", "alps", "jack", "date", "europe"],
        # ["cricket", "pirate", "day", "platypus", "pants"],
        # ["plane", "loch ness", "tooth", "nurse", "laser"],
        # ["match", "hawk", "life", "knife", "africa"],
    ]
    blue_words = [
        None, None, None, None, None
        # ["racket", "bug", "crown", "australia", "pipe"],
        # ["scuba diver", "play", "roulette", "table", "cloak"],
        # ["buffalo", "diamond", "kid", "witch", "swing"],
        # ["gas", "circle", "king", "unicorn", "cliff"],
        # ["lemon", "death", "conductor", "litter", "car"],
    ]

    for i, (red, blue) in enumerate(zip(red_words, blue_words)):

        game._build_game(red, blue)
        print("")
        print("TRIAL ", str(i), ":")
        print("RED WORDS: ", list(game.red_words))
        print("BLUE WORDS: ", list(game.blue_words))
        # print("NEAREST NEIGHBORS:")
        # for word, clues in game.weighted_nn.items():
        #     print(word)
        #     print(sorted(clues, key=lambda k: clues[k])[:5])

        score, clue = game.get_clue(2, 1)
        print("")
        print("CLUE CHOSEN: ", clue)

        print(
            "WORDS CHOSEN FOR CLUE: ",
            game.choose_words(2, clue, game.blue_words.union(game.red_words)),
        )

