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

from nltk.corpus import wordnet as wn

import networkx as nx
import numpy as np
import requests
from tqdm import tqdm

# Embeddings
from babelnet import Babelnet

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
    "s00081546n"  # word
])

"""
Configuration for running the game
"""


class CodenamesConfiguration(object):
    def __init__(
        self, verbose=False, visualize=False, split_multi_word=True, disable_verb_split=True
    ):
        self.verbose = verbose
        self.visualize = visualize
        self.split_multi_word = split_multi_word
        self.disable_verb_split = disable_verb_split


class Codenames(object):

    def __init__(
        self, ann_graph_path=None, num_emb_batches=22, embedding_type="custom", emb_size=200, file_dir=None,
        configuration=None
    ):
        """
        # TODO: Clean up this documenation
        :param ann_graph_path: path to AnnoyIndex graph (Approximate Neares Neighbors)
        see: https://github.com/spotify/annoy
        :param num_emb_batches: number of batches of word embeddings
        :param embedding_type: 'hsm', 'glove', 'multisense'
        :param embedding: an embedding object that codenames will use to play

        """
        # Intialize variables
        if configuration != None:
            self.configuration = configuration
        else:
            self.configuration = CodenamesConfiguration()
        self.embedding = self._get_embedding_from_type(embedding_type)
        self.weighted_nn = dict()
        self.num_emb_batches = num_emb_batches
        self.emb_size = emb_size
        self.file_dir = file_dir
        # pre-process

        # data from: https://github.com/uhh-lt/path2vec#pre-trained-models-and-datasets
        # self.get_path2vec_emb_from_txt(data_path='data/jcn-semcor_embeddings.vec')
        # self.lemma_nns = self.get_wordnet_nns()

        # self.categories_g = self.get_wibitaxonomy_categories_graph()
        # self.pages_g = self.get_wibitaxonomy_pages_graph()

        # self.graph = self.get_wikidata_graph()

        # self.graph = self.build_graph(emb_type=embedding_type, embeddings=self.embeddings, num_trees = 100, metric = 'angular')

        # self.graph.save('glove.ann')

        # self.graph = AnnoyIndex(self.emb_size)
        # self.graph.load('../window5_lneighbor5e-2.ann')
        # print("Built Annoy Graph")

        # self.model = gensim.models.KeyedVectors.load_word2vec_format('/Users/divyakoyyalagunta/Desktop/Research/word2vec_google_news/GoogleNews-vectors-negative300.bin', binary=True)

    """
    Codenames game setup
    """

    def _get_embedding_from_type(self, embedding_type):
        """
        :param embedding_type: 'hsm', 'glove', 'multisense'
        returns the embedding object that will be used to play

        """
        if embedding_type == 'babelnet':
            return Babelnet(self.configuration)

        return None

    def _build_game(self, red=None, blue=None, save_path=None):
        """
        :param red: optional list of strings of opponent team's words
        :param blue: optional list of strings of our team's words
        :param save_path: optional directory path to save data between games
        :return: None
        """
        self._generate_board_words(red, blue)
        if self.configuration.verbose:
            print("red words:", red)
            print("blue words:", blue)
        self.sess = requests.Session()
        self.wikipedia_url = "https://en.wikipedia.org/w/api.php"
        self.save_path = save_path
        self.weighted_nn = dict()

        words = self.blue_words.union(self.red_words)
        for word in words:
            # e.g. for word = "spoon",   weighted_nns[word] = {'fork':30, 'knife':25}
            # self.weighted_nn[word] = self.get_fake_knn(word)
            # self.weighted_nn[word] = self.get_hsm_knn(word)
            # self.weighted_nn[word] = self.get_path2vec_knn(word)
            # self.weighted_nn[word] = self.get_wordnet_knn(word)
            # self.weighted_nn[word] = self.get_glove_knn(word)
            # self.weighted_nn[word] = self.get_wikidata_knn(word)
            # self.weighted_nn[word] = self.get_wibitaxonomy(word, pages=True, categories=True)
            # self.weighted_nn[word] = self.get_word2vec_knn(word)
            # self.weighted_nn[word] = self.get_babelnet(word)
            self.weighted_nn[word] = self.embedding.get_weighted_nn(word)

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

            if self.configuration.visualize:  # and count == 0:
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

    def get_highest_clue(self, chosen_words, penalty=1.0, domain_threshold=0.45, domain_gap=0.3):

        potential_clues = set()
        for word in chosen_words:
            nns = self.weighted_nn[word]
            potential_clues.update(nns)

        # TODO : Instead of this override behavior, add domains to nn_w_dist
        # try to get a domain clue
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

            # Give embedding methods the opportunity to rescale the score using their own heuristics
            embedding_score = self.embedding.rescale_score(
                chosen_words, clue, self.red_words)

            score = sum(blue_word_counts) - (penalty *
                                             sum(red_word_counts)) + embedding_score
            # if score >= highest_score and self.verbose:
            #     print(clue, score, ">= highest_scoring_clue")
            if score > highest_score:
                highest_scoring_clues = [clue]
                highest_score = score
            elif score == highest_score:
                highest_scoring_clues.append(clue)

        return highest_scoring_clues, highest_score

    def choose_words(self, n, clue, remaining_words, domain_threshold=0.45):
        # given a clue word, choose the n words from remaining_words that most relates to the clue

        pq = []

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

    """
    Other embedding methods
    """

    # TODO Anna: Do we need all these prior to v5 methods anymore?
    def get_random_n_labels(self, labels, n, delimiter=" "):
        if len(labels) > 0:
            rand_indices = list(range(len(labels)))
            random.shuffle(rand_indices)
            return delimiter.join([labels[i] for i in rand_indices[:n]])
        return None

    def get_cached_labels_from_synset(self, synset, delimiter=" "):
        if synset not in self.synset_to_labels:
            labels = self.get_labels_from_synset(synset)
            self.write_synset_labels(synset, labels)
            filtered_labels = [label for label in labels if len(
                label.split("_")) == 1 or label.split("_")[1][0] == '(']
            sliced_labels = self.get_random_n_labels(
                filtered_labels, 3, delimiter) or synset
            self.synset_to_labels[synset] = sliced_labels
        else:
            sliced_labels = self.synset_to_labels[synset]
        return sliced_labels

    def load_synset_labels(self):
        if not os.path.exists(self.synset_labels_file):
            return {}
        synset_to_labels = {}
        with open(self.synset_labels_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 1:
                    # no labels found for this synset
                    continue
                synset, labels = parts[0], parts[1:]
                filtered_labels = [label for label in labels if len(
                    label.split("_")) == 1 or label.split("_")[1][0] == '(']
                sliced_labels = self.get_random_n_labels(
                    filtered_labels, 3) or synset
                synset_to_labels[synset] = sliced_labels
        return synset_to_labels

    def write_synset_labels(self, synset, labels):
        with open(self.synset_labels_file, "a") as f:
            f.write("\t".join([synset] + labels) + "\n")

    def get_labels_from_synset(self, synset):
        url = "https://babelnet.org/sparql/"
        queryString = """
        SELECT ?label WHERE {{
            <http://babelnet.org/rdf/s{synset}> a skos:Concept .
            OPTIONAL {{
                <http://babelnet.org/rdf/s{synset}> lemon:isReferenceOf ?sense .
                ?entry lemon:sense ?sense .
                ?entry lemon:language "EN" .
                ?entry rdfs:label ?label
            }}
        }}
        """.format(
            synset=synset.lstrip("bn:")
        )
        query = queryString.replace(" ", "+")
        fmt = urllib.parse.quote(
            "application/sparql-results+json".encode("UTF-8"), safe=""
        )
        params = {
            "query": query,
            "format": fmt,
            "key": "e3b6a00a-c035-4430-8d71-661cdf3d5837",
        }
        payload_str = "&".join("%s=%s" % (k, v) for k, v in params.items())

        res = requests.get("?".join([url, payload_str]))
        if "label" not in res.json()["results"]["bindings"][0]:
            return []
        labels = [r["label"]["value"]
                  for r in res.json()["results"]["bindings"]]
        return labels

    def get_babelnet_results(self, word, i):
        url = "https://babelnet.org/sparql/"
        queryString = """
        SELECT DISTINCT ?synset ?broader ?label (COUNT(?narrower) AS ?count) WHERE {{
            ?synset skos:broader{{{i}}} ?broader .
            ?synset skos:narrower ?narrower .
            ?broader lemon:isReferenceOf ?sense .
            ?entry lemon:sense ?sense .
            ?entry lemon:language "EN" .
            ?entry rdfs:label ?label .
            {{
                SELECT DISTINCT ?synset WHERE {{
                    ?entries a lemon:LexicalEntry .
                    ?entries lemon:sense ?sense .
                    ?sense lemon:reference ?synset .
                    ?entries rdfs:label "{word}"@en
                }} LIMIT 3
            }}
        }}
        """.format(
            i=i, word=word
        )
        query = queryString.replace(" ", "+")
        fmt = urllib.parse.quote(
            "application/sparql-results+json".encode("UTF-8"), safe=""
        )
        params = {
            "query": query,
            "format": fmt,
            "key": "e3b6a00a-c035-4430-8d71-661cdf3d5837",
        }
        payload_str = "&".join("%s=%s" % (k, v) for k, v in params.items())
        try:
            res = requests.get("?".join([url, payload_str]))
            return [
                (
                    r["synset"]["value"].split("/")[-1],
                    r["broader"]["value"].split("/")[-1],
                    r["label"]["value"],
                    r["count"]["value"],
                    i,
                )
                for r in res.json()["results"]["bindings"]
            ]
        except Exception as e:
            print(word, i)
            print(res.status_code, res.text)
            raise e

    def get_babelnet(self, word, depth=3):
        l = []
        nn = {}
        hyponym_count = {}
        assert self.save_path is not None
        with open(self.save_path, "a") as f:
            for i in range(1, depth+1):
                l += self.get_babelnet_results(word.lower(), i)
                l += self.get_babelnet_results(word.capitalize(), i)
            for (synset, broader, label, count, i) in l:
                f.write(
                    "\t".join([word, synset, broader, label, str(i)]) + "\n")
                if len(label.split("_")) > 1:
                    continue
                if label not in nn:
                    nn[label] = i
                    hyponym_count[label] = 0
                nn[label] = min(i, nn[label])
                hyponym_count[label] += int(count)

        for label in hyponym_count:
            if hyponym_count[label] > 100:
                del nn[label]
        return {k: 1.0 / (v + 1) for k, v in nn.items() if k != word}


### WORDNET ###


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
                    if self.verbose:
                        print("unexpected emb vector size:", len(emb_vector))
                    continue
                synset_to_idx[synset_str] = i
                idx_to_synset[i] = synset_str
                tree.add_item(i, emb_vector)
        tree.build(100)
        self.graph = tree
        self.synset_to_idx = synset_to_idx
        self.idx_to_synset = idx_to_synset

    def get_wibitaxonomy_categories_graph(self):
        file_dir = "data/wibi-ver2.0/taxonomies/"
        categories_file = file_dir + "WiBi.categorytaxonomy.ver1.0.txt"
        return nx.read_adjlist(
            categories_file, delimiter="\t", create_using=nx.DiGraph()
        )

    def get_wibitaxonomy_pages_graph(self):
        file_dir = "data/wibi-ver2.0/taxonomies/"
        pages_file = file_dir + "WiBi.pagetaxonomy.ver2.0.txt"
        return nx.read_adjlist(pages_file, delimiter="\t", create_using=nx.DiGraph())

    def get_wibitaxonomy(self, word, pages, categories):
        nn_w_dists = {}
        if pages:
            req_params = {
                "action": "opensearch",
                "namespace": "0",
                "search": word,
                "limit": "5",
                "format": "json",
            }
            req = self.sess.get(url=self.wikipedia_url, params=req_params)
            req_data = req.json()
            search_results = req_data[1]
            for w in search_results:
                try:
                    lengths = nx.single_source_shortest_path_length(
                        self.pages_g, source=w, cutoff=10
                    )
                    for neighbor, length in lengths.items():
                        if neighbor not in nn_w_dists:
                            nn_w_dists[neighbor] = length
                        else:
                            if self.verbose:
                                print(neighbor, 'length:', length,
                                      'prev length:', nn_w_dists[neighbor])
                        nn_w_dists[neighbor] = min(
                            length, nn_w_dists[neighbor])
                except NodeNotFound:
                    # if self.verbose:
                    #     print(w, "not in pages_g")
                    pass
        if categories:
            req_params = {
                "action": "opensearch",
                "namespace": "0",
                "search": "Category:" + word,
                "limit": "3",
                "format": "json",
            }
            req = self.sess.get(url=self.wikipedia_url, params=req_params)
            req_data = req.json()
            search_results = req_data[1]

            for w_untrimmed in search_results:
                w = w_untrimmed.split(":")[1]
                try:
                    lengths = nx.single_source_shortest_path_length(
                        self.categories_g, source=w, cutoff=10
                    )
                    for neighbor, length in lengths.items():
                        if neighbor not in nn_w_dists:
                            nn_w_dists[neighbor] = length
                        else:
                            if self.verbose:
                                print(neighbor, 'length:', length,
                                      'prev length:', nn_w_dists[neighbor])
                        nn_w_dists[neighbor] = min(
                            length, nn_w_dists[neighbor])
                except NodeNotFound:
                    # if self.verbose:
                    #     print(w, "not in categories_g")
                    pass
        return {k: 1.0 / (v + 1) for k, v in nn_w_dists.items() if k != word}

    def get_wikidata_graph(self):
        file_dir = "data/"
        source_id_names_file = file_dir + "daiquery-2020-02-25T23_38_13-08_00.tsv"
        target_id_names_file = file_dir + "daiquery-2020-02-25T23_54_03-08_00.tsv"
        edges_file = file_dir + "daiquery-2020-02-25T23_04_31-08_00.csv"
        self.source_name_id_map = {}
        self.wiki_id_name_map = {}
        with open(source_id_names_file, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                wiki_id, array_str = line.strip().split("\t")
                if len(array_str) <= 8:
                    if self.verbose:
                        print("array_str:", array_str)
                    continue
                # array_str[4:-4]
                source_names = re.sub(r"[\"\[\]]", "", array_str).split(",")
                for name in source_names:
                    if name not in self.source_name_id_map:
                        self.source_name_id_map[name] = set()
                    if wiki_id not in self.wiki_id_name_map:
                        self.wiki_id_name_map[wiki_id] = set()
                    self.source_name_id_map[name].add(wiki_id)
                    self.wiki_id_name_map[wiki_id].add(name)
        with open(target_id_names_file, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                wiki_id, name = line.strip().split("\t")
                if wiki_id not in self.wiki_id_name_map:
                    self.wiki_id_name_map[wiki_id] = set()
                self.wiki_id_name_map[wiki_id].add(name)

        return nx.read_adjlist(edges_file, delimiter=",", create_using=nx.DiGraph())

    def get_wikidata_knn(self, word):
        if word not in self.source_name_id_map:
            return {}
        wiki_ids = self.source_name_id_map[word]

        nn_w_dists = {}
        for wiki_id in wiki_ids:
            try:
                lengths = nx.single_source_shortest_path_length(
                    self.graph, source=wiki_id, cutoff=10
                )
            except NodeNotFound:
                if self.verbose:
                    print(wiki_id, "not in G")
                continue
            for node in lengths:
                names = self.wiki_id_name_map[str(node)]
                for name in names:
                    if name not in nn_w_dists:
                        nn_w_dists[name] = lengths[node]
                    nn_w_dists[name] = min(lengths[node], nn_w_dists[name])
        return {k: 1.0 / (v + 1) for k, v in nn_w_dists.items() if k != word}

    def get_wordnet_nns(self):
        d_lemmas = {}
        for ss in tqdm(wn.all_synsets(pos="n")):
            self.add_lemmas(d_lemmas, ss, ss, 0)
            # get the transitive closure of all hypernyms of a synset
            for i, hyper in enumerate(ss.closure(lambda s: s.hypernyms())):
                self.add_lemmas(d_lemmas, ss, hyper, i + 1)

            # also write transitive closure for all instances of a synset
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
                            nn_w_dists[lemma] = self.graph.get_distance(
                                id, nn_id)
                        nn_w_dists[lemma] = min(
                            self.graph.get_distance(
                                id, nn_id), nn_w_dists[lemma]
                        )
                except ValueError:
                    if self.verbose:
                        print(ss, "not a valid synset")
        # return dict[nn] = score
        # we store multiple lemmas with same score,
        # because in the future we can downweight
        # lemmas that are closer to enemy words
        return nn_w_dists

    def get_word2vec_knn(self, clue_word):
        nn_w_dists = {}
        limit = 5

        def recurse_word2vec(word, curr_limit):
            if curr_limit >= limit or word not in self.model.vocab:
                return
            neighbors = [x[0] for x in self.model.most_similar(word)]
            for neighbor in neighbors:
                if (self.model.vocab[neighbor].count < 2 or len(neighbor.split("_")) > 1):
                    continue
                dist = self.model.similarity(neighbor, clue_word)
                neighbor = neighbor.lower()
                if neighbor not in nn_w_dists:
                    nn_w_dists[neighbor] = dist
                    recurse_word2vec(neighbor, curr_limit + 1)
                nn_w_dists[neighbor] = min(dist, nn_w_dists[neighbor])

        recurse_word2vec(clue_word, 0)

        # if self.verbose:
        #     print(clue_word, nn_w_dists)

        return {k: 1.0 / (v + 1) for k, v in nn_w_dists.items() if k != clue_word}

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('embedding', type=str,
                        help='an embedding method to use when playing codenames')
    parser.add_argument('--verbose', dest='verbose', default=False,
                        help='print out verbose information')
    parser.add_argument('--visualize', dest='visualize', default=False,
                        help='visualize the choice of clues with graphs')
    args = parser.parse_args()

    words = ['vacuum', 'whip', 'moon', 'school', 'tube', 'lab', 'key', 'table', 'lead', 'crown', 'bomb', 'bug', 'pipe', 'roulette',
             'australia', 'play', 'cloak', 'piano']
    random.shuffle(words)

    # ["bug", "crown", "australia", "pipe",
    # "bomb", "play", "roulette", "table", "cloak"],
    # ["key", "piano", "lab", "school", "lead",  # ],
    # "whip", "tube", "vacuum", "lab", "moon"],

    # Use None to randomize the game, or pass in fixed lists
    red_words = [
         ['moon', 'play', 'vacuum', 'school', 'cloak', 'piano', 'table', 'lab', 'key', 'tube']
,
        #['crown', 'bomb', 'bug', 'pipe', 'roulette', 'australia', 'play', 'cloak', 'table']
    ]

    blue_words = [
        ['whip', 'roulette', 'australia', 'lead', 'bug', 'crown', 'bomb', 'pipe'],
        #['vacuum', 'whip', 'moon', 'school', 'tube', 'lab', 'key', 'piano', 'lead'],
    ]

    configuration = CodenamesConfiguration(
        verbose=args.verbose, visualize=args.visualize)
    game = Codenames(
        configuration=configuration,
        embedding_type=args.embedding,
    )

    for i, (red, blue) in enumerate(zip(red_words, blue_words)):

        game._build_game(red=red, blue=blue,
                         save_path="tmp_babelnet_" + str(i))
        print("")
        print("TRIAL", str(i), ":")
        print("RED WORDS: ", list(game.red_words))
        print("BLUE WORDS: ", list(game.blue_words))
        # TODO: Download version without using aliases. They may be too confusing
        if game.configuration.verbose:
            print("NEAREST NEIGHBORS:")
            for word, clues in game.weighted_nn.items():
                print(word)
                print(sorted(clues, key=lambda k: clues[k], reverse=True)[:5])

        best_scores, best_clues, best_board_words_for_clue = game.get_clue(2, 1)
        print("===================================================================================================")
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
