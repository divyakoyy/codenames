import itertools
import heapq
import sys
import os
sys.path.insert(0, '../')
import numpy as np
from annoy import AnnoyIndex
import random
from nltk.corpus import wordnet as wn
import networkx as nx
import matplotlib as mp
mp.interactive(True)

class Game(object):
    def __init__(self):
        self.red_words = set()
        self.blue_words = set()
        self.weighted_nn = dict()
        self.idx_to_word = dict()

        self._generate_board('data/codewords.txt')

    def _generate_board(self, vocab_file):
        with open(vocab_file) as file:
            for i, line in enumerate(file):
                word = line.strip().lower()
                self.idx_to_word[i] = word

        rand_idxs = random.sample(range(0, len(self.idx_to_word.keys())), 10)
        self.red_words = set([self.idx_to_word[idx] for idx in rand_idxs[:5]])
        self.blue_words = set([self.idx_to_word[idx] for idx in rand_idxs[5:]])

        # self.red_words = {"room", "gold", "marathon", "star", "planet"}
        # self.blue_words = {"american", "government", "office", "election", "earth"}

    def build_game(self, embedding):
        words = self.blue_words.union(self.red_words)
        for word in words:
            # e.g. for word = "spoon",   weighted_nns[word] = {'fork':30, 'knife':25}
            self.weighted_nn[word] = embedding.get_multisense_knn(word)
            # print(word)
            # print(sorted(self.weighted_nn[word].items(), key=operator.itemgetter(1)))

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
            blue_word_counts = [self.weighted_nn[blue_word][clue] for blue_word in self.blue_words
                                if clue in self.weighted_nn[blue_word]]
            red_word_counts = [self.weighted_nn[red_word][clue] for red_word in self.red_words
                               if clue in self.weighted_nn[red_word]]
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
            # min heap, so push negative score
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

class MultisenseEmbedding(object):
    def __init__(
            self,
            base_dir,
            vocab_dir,
            testing_data_dir,
            num_emb_batches=22,
            emb_size=200):

        # maps codewords' (from 'codewords.txt') line index to word

        self.idx_to_word = dict()
        self.num_emb_batches = num_emb_batches
        self.emb_size = emb_size
        self.graph = AnnoyIndex(self.emb_size)

        self._load_ann_graph(os.path.join(base_dir, testing_data_dir + "custom.ann"))
        self.word_to_indices, self.indices_to_word = self._load_graph_indices(
            os.path.join(base_dir, vocab_dir),
            os.path.join(base_dir, testing_data_dir)
        )

    def _load_ann_graph(self, ann_path):
        try:
            self.graph.load(ann_path)
            print("Built Multisense Annoy Graph")
        except IOError:
            print("Construct Annoy graph in knn.py before trying to look at nn")

    def _load_graph_indices(self, vocab_dir, testing_data_dir):
        inv_vocab_dict_path = os.path.join(vocab_dir, "inv_vocab_dict_trimmed.npy")
        inv_vocab_dict = np.load(inv_vocab_dict_path, encoding='bytes').item()

        word_to_indices = dict()
        index_to_word = dict()
        run = 0
        index = 0
        while run < 1000:
            if run % 200 == 0:
                print ("Run: " + str(run))
            try:
                input_path = os.path.join(testing_data_dir, "input_" + str(run) + "_A.npy")
                input_file_a = np.load(input_path, encoding='bytes')
                for x in range(len(input_file_a)):
                    cur_words = [inv_vocab_dict[word_idx]['word'] for word_idx in input_file_a[x][0]]
                    center_word = cur_words[2].lower()
                    if center_word not in word_to_indices:
                        word_to_indices[center_word] = [index]
                    else:
                        word_to_indices[center_word].append(index)
                    index_to_word[index] = center_word
                    index += 1

                input_path = os.path.join(testing_data_dir, "input_" + str(run) + "_B.npy")
                input_file_b = np.load(input_path, encoding='bytes')
                for x in range(len(input_file_b)):
                    cur_words = [inv_vocab_dict[word_idx]['word'] for word_idx in input_file_b[x][0]]
                    center_word = cur_words[2].lower()
                    if center_word not in word_to_indices:
                        word_to_indices[center_word] = [index]
                    else:
                        word_to_indices[center_word].append(index)
                    index_to_word[index] = center_word
                    index += 1
            except BaseException as e:
                print(e)
                break
            run += 1
        return word_to_indices, index_to_word

    def get_multisense_knn(self, word, nums_nns = 250):
        if word not in self.word_to_indices:
            return {}

        nn_w_dists = dict()

        for id in self.word_to_indices[word]:
            nn_indices = set(self.graph.get_nns_by_item(id, nums_nns))
            nn_words = []
            for nn_id in nn_indices:
                nn_word = self.indices_to_word[nn_id]
                if nn_word == word:
                    continue
                if nn_word not in nn_w_dists:
                    nn_w_dists[nn_word] = []
                nn_w_dists[nn_word].append(self.graph.get_distance(id, nn_id))

        nn_w_scores = dict()
        for unique_neighbor_word in nn_w_dists:
            freq_nn = len(nn_w_dists[unique_neighbor_word])
            if freq_nn > 10:
                #take the top 5% closest distances
                closest_points = sorted(nn_w_dists[unique_neighbor_word])[:int(freq_nn*0.10)]
                nn_w_scores[unique_neighbor_word] = sum(closest_points)/len(closest_points)
        return nn_w_scores


class BertEmbedding(object):
    def __init__(
            self,
            base_dir,
            idx_to_word_dir,
            emb_size=768):

        self.emb_size = emb_size
        self.graph = AnnoyIndex(self.emb_size)

        self._load_ann_graph(os.path.join(base_dir, "bert_wordnet.ann"))
        self.word_to_indices, self.idx_to_word = self._load_graph_indices(
            os.path.join(base_dir, idx_to_word_dir))

    def _load_ann_graph(self, ann_path='bert/bert_wordnet.ann'):
        try:
            self.graph.load(ann_path)
            print("Built Bert Annoy Graph")
        except IOError:
            print("Construct Annoy graph in knn.py before trying to look at nn")

    def _load_graph_indices(self, idx_to_word_dir):
        idx_to_word = np.load(idx_to_word_dir).item()
        words_to_indices = dict()

        for idx in idx_to_word:
            word = idx_to_word[idx]
            if word not in words_to_indices:
                words_to_indices[word] = []
            words_to_indices[word].append(idx)

        return words_to_indices, idx_to_word

    def get_multisense_knn(self, word, nums_nns = 250):
        if word not in self.word_to_indices:
            return {}

        nn_w_dists = dict()

        for id in self.word_to_indices[word]:
            nn_indices = set(self.graph.get_nns_by_item(id, nums_nns))
            for nn_id in nn_indices:
                nn_word = self.idx_to_word[nn_id]
                if nn_word == word:
                    continue
                if nn_word not in nn_w_dists:
                    nn_w_dists[nn_word] = []
                nn_w_dists[nn_word].append(self.graph.get_distance(id, nn_id))
        #print("word: ", word)
        #pprint.pprint(nn_w_dists)
        nn_w_scores = dict()
        for unique_neighbor_word in nn_w_dists:
            freq_nn = len(nn_w_dists[unique_neighbor_word])
            if freq_nn > 10:
                #print("unique neighbor word: ", unique_neighbor_word)
                #pprint.pprint(nn_w_dists[unique_neighbor_word])
                #take the top 5% closest distances
                closest_points = sorted(nn_w_dists[unique_neighbor_word])[:int(freq_nn*0.10)]
                # score is the average of the distances
                nn_w_scores[unique_neighbor_word] = sum(closest_points)/len(closest_points)

        print("word: ", word, "nn w scores", nn_w_scores)
        return nn_w_scores

def closure_graph(synset):
    seen = set()
    graph = nx.DiGraph()

    def recurse(s):
        if not s in seen:
            seen.add(s)
            graph.add_node(s.name)
            string = str(s.name)
            for s1 in wn.synsets(string.split(".")[0]):
                print("s1", s1)
                graph.add_node(s1.name)
                graph.add_edge(s.name, s1.name)
                recurse(s1)

    recurse(synset)
    return graph

class WordNetEmbedding(object):
    def __init__(self):
        pass

    def get_multisense_knn(self, word):
        seen = set()
        graph = nx.DiGraph()
        word = "spoon"

        def recurse(s, count):
            s_name = str(s.name())
            print("s",s_name)
            # TODO: prevent multi words
            if count < 3 and not s in seen:
                seen.add(s)
                graph.add_node(s.name)

                for s1 in wn.synsets(s_name):
                    graph.add_node(s1.name)
                    graph.add_edge(s.name, s1.name)
                    recurse(s1, count+1)


        nn_w_scores = dict()
        print("Word", word)
        graph.add_node(word)
        for syn in wn.synsets(word):

            if not syn in seen:
                seen.add(syn)
                graph.add_node(syn.name)
                graph.add_edge(word, syn.name)
                for hypernym in syn.hypernyms():
                    graph.add_node(syn.name)
                    graph.add_edge(syn.name, hypernym.name)

                for hyponym in syn.hyponyms():
                    graph.add_node(syn.name)
                    graph.add_edge(syn.name, hyponym.name)


            print(syn, syn.name(), syn.definition())
            for lemma in syn.lemmas():
                if lemma.name() == word:
                    continue
                print("    Lemma:", lemma.name())

                graph.add_node(lemma)
                graph.add_edge(word, lemma)
                recurse(lemma,0)
                #nn_w_scores[lemma.name()] = w1.wup_similarity(syn)
                nn_w_scores[lemma.name()] = 1

        pos = nx.spring_layout(graph, k=0.3*1/np.sqrt(20), iterations=20)
        nx.draw(graph, with_labels=True, pos=pos)
        mp.pyplot.show(block=True)

        print("NN W SCORES\n", nn_w_scores)
        return nn_w_scores


if __name__=='__main__':
    # base_dir = "/usr/xtmp/ays7/"
    # vocab_dir = "wikipedia_tagged_25000/"
    # testing_data_dir = "wikipedia_tagged_25000/0201_model_nca-0/test_embeddings/"
    # embedding = MultisenseEmbedding(base_dir, vocab_dir, testing_data_dir)

    # base_dir = "/Users/divyakoyyalagunta/projects/multisense/codenames/bert/"
    # idx_to_word_dir = "annoy_tree_index_to_word_bert_wordnet.npy"
    # embedding = BertEmbedding(base_dir=base_dir, idx_to_word_dir=idx_to_word_dir)

    embedding = WordNetEmbedding()

    game = Game()
    game.build_game(embedding)
    print("")
    print("RED WORDS: ", list(game.red_words))
    print("GREEN WORDS: ", list(game.blue_words))
    # print("NEAREST NEIGHBORS:")
    # pprint.pprint(game.weighted_nn)

    score, clue = game.get_clue(2, 1)
    print("")
    print("CLUE CHOSEN: ", clue)

    print("WORDS CHOSEN FOR CLUE: ", game.choose_words(2, clue, game.blue_words.union(game.red_words)))
