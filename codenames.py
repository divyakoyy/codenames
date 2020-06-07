import gzip
import heapq
import itertools
import os
import random
import re
import requests
import sys
import urllib

import numpy as np
# from annoy import AnnoyIndex
import networkx as nx
from networkx.exception import NodeNotFound
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
from nltk.corpus import wordnet as wn

from tqdm import tqdm
import matplotlib.pyplot as plt


sys.path.insert(0, "../")

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

babelnet_relationships_limits = {
    "HYPERNYM" : float("inf"),
    # "OTHER" : 15,
    # "MERONYM": 15,
    # "HYPONYM": 15,
}

class Game(object):
    def __init__(
        self, ann_graph_path=None, num_emb_batches=22, emb_type="custom", emb_size=200, file_dir=None,
        verbose=False, visualize=False, synset_labels_file=None,
        synset_main_sense_file=None, synset_senses_file=None, synset_glosses_file=None, synset_domains_file=None
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
        self.nn_synsets = dict()
        self.graphs = dict()
        # maps codewords' (from 'codewords.txt') line index to word
        self.idx_to_word = dict()
        self.num_emb_batches = num_emb_batches

        self.emb_size = emb_size
        self.verbose = verbose
        self.visualize = visualize
        self.file_dir = file_dir
        # self.synset_labels_file = synset_labels_file
        # self.synset_to_labels = self.load_synset_labels()
        self.synset_main_sense_file = synset_main_sense_file
        self.synset_senses_file = synset_senses_file
        self.synset_glosses_file = synset_glosses_file
        self.synset_domains_file = synset_domains_file
        (
            self.synset_to_main_sense,
            self.synset_to_senses,
            self.synset_to_domains
        ) = self.load_synset_labels_v5()

        # pre-process

        # data from: https://github.com/uhh-lt/path2vec#pre-trained-models-and-datasets
        # self.get_path2vec_emb_from_txt(data_path='data/jcn-semcor_embeddings.vec')
        # self.lemma_nns = self.get_wordnet_nns()

        # self.categories_g = self.get_wibitaxonomy_categories_graph()
        # self.pages_g = self.get_wibitaxonomy_pages_graph()

        # self.graph = self.get_wikidata_graph()

        # self.graph = self.build_graph(emb_type=emb_type, embeddings=self.embeddings, num_trees = 100, metric = 'angular')

        # self.graph.save('glove.ann')

        # self.graph = AnnoyIndex(self.emb_size)
        # self.graph.load('../window5_lneighbor5e-2.ann')
        # print("Built Annoy Graph")

        # self.model = gensim.models.KeyedVectors.load_word2vec_format('/Users/divyakoyyalagunta/Desktop/Research/word2vec_google_news/GoogleNews-vectors-negative300.bin', binary=True)

# GAME SET-UP

    def _build_game(self, red=None, blue=None, save_path=None):
        self._generate_board(red, blue)
        words = self.blue_words.union(self.red_words)
        if self.verbose:
            print("blue and red words:", words)
        self.sess = requests.Session()
        self.wikipedia_url = "https://en.wikipedia.org/w/api.php"
        self.save_path = save_path
        # reset weighted_nn between trials
        self.weighted_nn = dict()
        self.nn_synsets = dict()
        self.domains_nn = dict()
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
            (
                self.weighted_nn[word],
                self.nn_synsets[word],
                self.domains_nn[word],
                self.graphs[word]
            ) = self.get_babelnet_v5(word)
        print(self.domains_nn)

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
            filtered_labels = [label for label in labels if len(label.split("_")) == 1 or label.split("_")[1][0] == '(']
            sliced_labels = self.get_random_n_labels(filtered_labels, 3, delimiter) or synset
            self.synset_to_labels[synset] = sliced_labels
        else:
            sliced_labels = self.synset_to_labels[synset]
        return sliced_labels

    def load_synset_labels(self):
        if not os.path.exists(self.synset_labels_file):
            return {}
        synset_to_labels = {}
        with open(self.synset_labels_file, 'r') as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 1:
                    # no labels found for this synset
                    continue
                synset, labels = parts[0], parts[1:]
                filtered_labels = [label for label in labels if len(label.split("_")) == 1 or label.split("_")[1][0] == '(']
                sliced_labels = self.get_random_n_labels(filtered_labels, 3) or synset
                synset_to_labels[synset] = sliced_labels
        return synset_to_labels

    def write_synset_labels(self, synset, labels):
        with open(self.synset_labels_file, 'a') as f:
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
        """.format(synset=synset.lstrip('bn:'))
        query = queryString.replace(" ", "+")
        fmt = urllib.parse.quote("application/sparql-results+json".encode('UTF-8'), safe="")
        params = {
            "query": query,
            "format": fmt,
            "key": "e3b6a00a-c035-4430-8d71-661cdf3d5837",
        }
        payload_str = "&".join("%s=%s" % (k,v) for k,v in params.items())

        res = requests.get('?'.join([url, payload_str]))
        if 'label' not in res.json()['results']['bindings'][0]:
            return []
        labels = [
            r['label']['value']
            for r in res.json()['results']['bindings']
        ]
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
        """.format(i=i, word=word)
        query = queryString.replace(" ", "+")
        fmt = urllib.parse.quote("application/sparql-results+json".encode('UTF-8'), safe="")
        params = {
            "query": query,
            "format": fmt,
            "key": "e3b6a00a-c035-4430-8d71-661cdf3d5837"
        }
        payload_str = "&".join("%s=%s" % (k,v) for k,v in params.items())
        try:
            res = requests.get('?'.join([url, payload_str]))
            return [
                (
                    r['synset']['value'].split('/')[-1],
                    r['broader']['value'].split('/')[-1],
                    r['label']['value'],
                    r['count']['value'],
                    i
                )
                for r in res.json()['results']['bindings']
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
        with open(self.save_path , "a") as f:
            for i in range(1, depth+1):
                l += self.get_babelnet_results(word.lower(), i)
                l += self.get_babelnet_results(word.capitalize(), i)
            #print(word, len(l))
            for (synset, broader, label, count, i) in l:
                f.write("\t".join([word, synset, broader, label, str(i)]) + "\n")
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

    def get_cached_labels_from_synset_v5(self, synset, get_domains=False):
        """This actually gets the main_sense but also writes all senses/glosses"""
        # TODO: remove second OR
        if synset not in self.synset_to_main_sense or (
            get_domains and synset not in self.synset_to_domains):
            print("getting query", synset)
            assert False
            labels_json = self.get_labels_from_synset_v5_json(synset)
            self.write_synset_labels_v5(synset, labels_json)

            # filtered_labels = [label for label in labels if len(label.split("_")) == 1 or label.split("_")[1][0] == '(']
            # self.synset_to_labels[synset] = self.get_random_n_labels(filtered_labels, 3) or synset
        # sliced_labels = self.synset_to_labels[synset]
        main_sense = self.synset_to_main_sense[synset]
        senses = self.synset_to_senses[synset]
        domains = self.synset_to_domains[synset] if get_domains else {}
        return main_sense, senses, domains

    def load_synset_labels_v5(self):
        """Load synset_to_main_sense"""
        synset_to_main_sense = {}
        synset_to_senses = {}
        synset_to_domains = {}
        if os.path.exists(self.synset_main_sense_file):
            with open(self.synset_main_sense_file, 'r') as f:
                for line in f:
                    parts = line.strip().split("\t")
                    synset, main_sense = parts[0], parts[1]
                    synset_to_main_sense[synset] = main_sense
                    synset_to_senses[synset] = set()
                    # synset_to_domains[synset] = dict()
                    # TODO: uncomment and remove initialization below
        if os.path.exists(self.synset_senses_file):
            with open(self.synset_senses_file, 'r') as f:
                for line in f:
                    parts = line.strip().split("\t")
                    assert len(parts) == 5
                    synset, full_lemma, simple_lemma, source, pos = parts
                    if source == 'WIKIRED':
                        continue
                    synset_to_senses[synset].add(simple_lemma)
        if os.path.exists(self.synset_domains_file):
            with open(self.synset_domains_file, 'r') as f:
                for line in f:
                    parts = line.strip().split("\t")
                    assert len(parts) == 3
                    synset, domain, score = parts
                    if synset not in synset_to_domains:
                        synset_to_domains[synset] = dict()
                    if domain == 'NONE':
                        continue
                    score == float(score)
                    if domain in synset_to_domains[synset]:
                        synset_to_domains[synset][domain] = max(synset_to_domains[synset][domain], score)
                    else:
                        synset_to_domains[synset][domain] = score

        return synset_to_main_sense, synset_to_senses, synset_to_domains

    def write_synset_labels_v5(self, synset, json):
        """Write to synset_main_sense_file, synset_senses_file, and synset_glosses_file"""
        with open(self.synset_main_sense_file, 'a') as f:
            if 'mainSense' not in json:
                print("no main sense for", synset)
                main_sense = synset
            else:
                main_sense = json['mainSense']
            f.write('\t'.join([synset, main_sense]) + "\n")
            self.synset_to_main_sense[synset] = main_sense

        with open(self.synset_senses_file, 'a') as f:
            self.synset_to_senses[synset] = set()
            if 'senses' in json:
                for sense in json['senses']:
                    properties = sense['properties']
                    line = [
                        synset,
                        properties['fullLemma'],
                        properties['simpleLemma'],
                        properties['source'],
                        properties['pos'],
                    ]
                    f.write('\t'.join(line) + "\n")
                    self.synset_to_senses[synset].add(properties['simpleLemma'])

        with open(self.synset_glosses_file, 'a') as f:
            if 'glosses' in json:
                for gloss in json['glosses']:
                    line = [
                        synset,
                        gloss['source'],
                        gloss['gloss'],
                    ]
                    f.write('\t'.join(line) + "\n")

        with open(self.synset_domains_file, 'a') as f:
            self.synset_to_domains[synset] = dict()
            if 'domains' in json:
                for domain, score in json['domains'].items():
                    if domain in self.synset_to_domains[synset]:
                        self.synset_to_domains[synset][domain] = max(
                            self.synset_to_domains[synset][domain], score
                        )
                    else:
                        self.synset_to_domains[synset][domain] = score
                    f.write('\t'.join([synset, domain, str(score)]) + '\n')
            if 'domains' not in json or len(json['domains']) == 0:
                f.write('\t'.join([synset, 'NONE', '-100']) + '\n')


    def get_labels_from_synset_v5_json(self, synset):
        url = 'https://babelnet.io/v5/getSynset'
        params = {
            'id': synset,
            'key': 'e3b6a00a-c035-4430-8d71-661cdf3d5837'
        }
        headers = {'Accept-Encoding': 'gzip'}
        res = requests.get(url=url, params=params, headers=headers)
        return res.json()

    def parse_lemma_v5(self, lemma):
        lemma_parsed = lemma.split('#')[0]
        parts = lemma_parsed.split('_')
        single_word = len(parts) == 1 or parts[1].startswith('(')
        return parts[0], single_word

    def get_single_word_label_v5(self, lemma, senses):
        """"""
        parsed_lemma, single_word = self.parse_lemma_v5(lemma)
        if single_word:
            return parsed_lemma

        for sense in senses:
            parsed_lemma, single_word = self.parse_lemma_v5(sense)
            if single_word:
                return parsed_lemma
        # didn't find a single word label
        return lemma.split('#')[0]

    def get_babelnet_v5(self, word):

        count_by_relation_group = {key:0 for key in babelnet_relationships_limits.keys()}
        def should_add_relationship(relationship):
            return relationship in babelnet_relationships_limits.keys() and \
                   count_by_relation_group[relationship] < babelnet_relationships_limits[relationship]

        G = nx.DiGraph()
        with gzip.open(self.file_dir + word + '.gz', 'r') as f:
            for line in f:
                source, target, language, short_name, relation_group, is_automatic, _idx = line.decode("utf-8").strip().split('\t')

                if should_add_relationship(relation_group) and is_automatic == 'False':
                    G.add_edge(source, target)
                    count_by_relation_group[relation_group] += 1

        nn_w_dists = {}
        nn_w_synsets = {}
        nn_w_domains = {}
        with open(self.file_dir + word + '_synsets', 'r') as f:
            for line in f:
                synset = line.strip()
                try:
                    lengths = nx.single_source_shortest_path_length(
                        G, source=synset, cutoff=10
                    )
                except NodeNotFound as e:
                    print(e)
                    continue
                for neighbor, length in lengths.items():
                    neighbor_main_sense, neighbor_senses, _ = self.get_cached_labels_from_synset_v5(neighbor, get_domains=False)
                    single_word_label = self.get_single_word_label_v5(neighbor_main_sense, neighbor_senses)
                    if single_word_label not in nn_w_dists:
                        nn_w_dists[single_word_label] = length
                        nn_w_synsets[single_word_label] = neighbor
                    else:
                        if nn_w_dists[single_word_label] > length:
                            nn_w_dists[single_word_label] = length
                            nn_w_synsets[single_word_label] = neighbor
                main_sense, sense, domains = self.get_cached_labels_from_synset_v5(synset, get_domains=True)
                for domain, score in domains.items():
                    nn_w_domains[domain] = float(score)
        return (
            {k: 1.0 / (v + 1) for k, v in nn_w_dists.items() if k != word},
            nn_w_synsets,
            nn_w_domains,
            G
        )


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
        categories_file = file_dir + 'WiBi.categorytaxonomy.ver1.0.txt'
        return nx.read_adjlist(categories_file, delimiter='\t', create_using=nx.DiGraph())

    def get_wibitaxonomy_pages_graph(self):
        file_dir = "data/wibi-ver2.0/taxonomies/"
        pages_file = file_dir + 'WiBi.pagetaxonomy.ver2.0.txt'
        return nx.read_adjlist(pages_file, delimiter='\t', create_using=nx.DiGraph())

    def get_wibitaxonomy(self, word, pages, categories):
        nn_w_dists = {}
        if pages:
            req_params = {
                "action": "opensearch",
                "namespace": "0",
                "search": word,
                "limit": "5",
                "format": "json"
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
                                print(neighbor, 'length:', length, 'prev length:', nn_w_dists[neighbor])
                        nn_w_dists[neighbor] = min(length, nn_w_dists[neighbor])
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
                "format": "json"
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
                                print(neighbor, 'length:', length, 'prev length:', nn_w_dists[neighbor])
                        nn_w_dists[neighbor] = min(length, nn_w_dists[neighbor])
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
                            nn_w_dists[lemma] = self.graph.get_distance(id, nn_id)
                        nn_w_dists[lemma] = min(
                            self.graph.get_distance(id, nn_id), nn_w_dists[lemma]
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
            if (curr_limit >= limit or word not in self.model.vocab):
                return
            neighbors = [x[0] for x in self.model.most_similar(word)]
            for neighbor in neighbors:
                if (self.model.vocab[neighbor].count < 2 or  len(neighbor.split("_")) > 1):
                    continue
                dist = self.model.similarity(neighbor, clue_word)
                neighbor = neighbor.lower()
                if neighbor not in nn_w_dists:
                    nn_w_dists[neighbor] = dist
                    recurse_word2vec(neighbor, curr_limit+1)
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


### CODENAMES FUNCTIONS ###

    def get_intersecting_graphs(self, word_set, clue):
        # clue is the word that intersects for this word_set
        # for example if we have the word_set (beijing, phoenix) and clue city,
        # this method will produce a directed graph that shows the path of intersection
        clue_graph = nx.DiGraph()
        word_set_graphs = [nx.DiGraph() for _ in word_set]

        if all([clue in self.domains_nn[word] for word in word_set]):
            # this clue came frome 2 domains, don't plot
            return

        for word in word_set:
            # Create graph that shows the intersection from the clue to the word_set
            if clue in self.nn_synsets[word]:
                clue_synset = self.nn_synsets[word][clue] # synset of the clue
            else:
                if self.verbose:
                    print("Clue ", clue, "not found for word", word)
                continue
            word_graph = self.graphs[word]
            shortest_path = []
            shortest_path_length = float("inf")
            with open(self.file_dir + word + '_synsets', 'r') as f:
                for line in f:
                    synset = line.strip()
                    try:
                        path = nx.shortest_path(word_graph, synset, clue_synset)
                        if len(path) < shortest_path_length:
                            shortest_path_length = len(path)
                            shortest_path = path
                    except:
                        if self.verbose:
                            print("No path between", synset, clue, clue_synset)

                # path goes from source (word in word_set) to target (clue)
                shortest_path_labels = []
                for synset in shortest_path:
                    # TODO: add graph for domains?
                    main_sense, senses, _ = self.get_cached_labels_from_synset_v5(synset, get_domains=False)
                    single_word_label = self.get_single_word_label_v5(main_sense, senses)
                    shortest_path_labels.append(single_word_label)

                print ("shortest path from", word, "to clue", clue, clue_synset, ":", shortest_path_labels)

                formatted_labels = [label.replace(' ', '\n') for label in shortest_path_labels]
                formatted_labels.reverse()
                nx.add_path(clue_graph, formatted_labels)



        self.draw_graph(clue_graph, ('_').join([clue] + [word for word in word_set]))


    def draw_graph(self, graph, graph_name, get_labels=False):
        write_dot(graph, 'test.dot')
        pos = graphviz_layout(graph, prog='dot')

        # if we need to get labels for our graph (because the node text is synset ids)
        # we will create a dictionary { node : label } to pass into our graphing options
        nodes_to_labels = dict()
        current_labels = nx.draw_networkx_labels(graph, pos=pos)

        if get_labels:
            for synset_id in current_labels:
                main_sense, senses, domains = self.get_cached_labels_from_synset_v5(synset_id, get_domains=True)
                nodes_to_labels[synset_id] = main_sense
        else:
            nodes_to_labels = { label : label for label in current_labels}

        plt.figure()
        options = {
            'node_color': 'white',
            'line_color': 'black',
            'linewidths': 0,
            'width': 0.5,
            'pos': pos,
            'with_labels': True,
            'font_color': 'black',
            'font_size': 4,
            'labels': nodes_to_labels,
        }
        nx.draw(graph, **options)

        if not os.path.exists('intersection_graphs'):
            os.makedirs('intersection_graphs')
        filename = 'intersection_graphs/' + graph_name + '.png'
        # margins
        plot_margin = 0.35
        x0, x1, y0, y1 = plt.axis()
        plt.axis((x0 - plot_margin,
                  x1 + plot_margin,
                  y0 - plot_margin,
                  y1 + plot_margin))
        plt.savefig(filename, dpi=300)
        plt.close()

    def get_clue(self, n, penalty):
        # where blue words are our team's words and red words are the other team's words
        # potential clue candidates are the intersection of weighted_nns[word] for each word in blue_words
        # we need to repeat this for the (|blue_words| C n) possible words we can give a clue for

        pq = []
        for word_set in itertools.combinations(self.blue_words, n):
            highest_clues, score = self.get_highest_clue(word_set, penalty)
            # min heap, so push negative score
            heapq.heappush(pq, (-1 * score, highest_clues, word_set))

        # sliced_labels = self.get_cached_labels_from_synset(clue)
        # main_sense, _senses = self.get_cached_labels_from_synset_v5(clue)

        best_clues = []
        best_scores = []
        count = 0

        while pq:
            score, clues, word_set = heapq.heappop(pq)

            if count >= 5:
                break

            if self.visualize:  # and count == 0:
                for clue in clues:
                    self.get_intersecting_graphs(word_set, clue)

            best_clues.append(clues)
            best_scores.append(score)

            count += 1

        return best_scores, best_clues

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

        # try to get a domain clue
        domains = {}
        for word in set(chosen_words).union(self.red_words):
            for domain, score in self.domains_nn[word].items():
                score = self.rescale_domain_score(score)
                if score > domain_threshold:
                    if domain not in domains:
                        domains[domain] = dict()
                    domains[domain][word] = score
        domain_clues = []
        for domain, word_scores in domains.items():
            if all([word in word_scores for word in chosen_words]):
                # get min word_set domain score
                min_chosen_words_score = min([word_scores[word] for word in chosen_words])
                # get max red_words domain score
                red_words_domain_scores = [word_scores[word] for word in self.red_words.intersection(word_scores.keys())]
                if len(red_words_domain_scores) == 0:
                    max_red_words_score = 0
                else:
                    max_red_words_score = max(red_words_domain_scores) + domain_gap

                if min_chosen_words_score > max_red_words_score:
                    domain_clues.append(domain)
        if len(domain_clues) >= 1:
            return domain_clues, 1  # TODO: return different score?

        potential_clues = potential_clues - self.blue_words.union(self.red_words)

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
            score = sum(blue_word_counts) - (penalty * sum(red_word_counts))
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
            if clue in self.domains_nn[word]:
                score = self.rescale_domain_score(self.domains_nn[word][clue])
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
    file_dir = 'babelnet_v6/'
    # synset_labels_file = 'synset_to_labels.txt'
    synset_main_sense_file = 'synset_to_main_sense.txt'
    synset_senses_file = 'synset_to_senses.txt'
    synset_glosses_file = 'synset_to_glosses.txt'
    synset_domains_file = 'synset_to_domains.txt'
    game = Game(
        verbose=False,
        visualize=True,
        file_dir=file_dir,
        # synset_labels_file=file_dir+synset_labels_file,
        synset_main_sense_file=file_dir+synset_main_sense_file,
        synset_senses_file=file_dir+synset_senses_file,
        synset_glosses_file=file_dir+synset_glosses_file,
        synset_domains_file=file_dir+synset_domains_file,
    )
    # Use None to randomize the game, or pass in fixed lists
    red_words = [
        # None,
        # None,
        # None,
        # None,
        # None,
        # ['ray', 'mammoth', 'ivory', 'racket', 'bug'],
        # ['laser', 'pan', 'stock', 'box', 'game'],
        # ['bomb', 'car', 'moscow', 'pipe', 'hand'],
        # ['penguin', 'stick', 'racket', 'scale', 'ivory'],
        # ['horseshoe', 'amazon', 'thumb', 'spider', 'lion'],
        # ["racket",
        ["bug", "crown", "australia", "pipe", #],
        "bomb", "play", "roulette", "table", "cloak"],
        # ["gas", "circle", "unicorn", "king", "cliff"],
        # ["conductor", "diamond", "kid", "witch", "swing"],
        # ["death", "litter", "car", "lemon", "conductor"],
        # ["board", "web", "wave", "platypus", "mine"],
        # ["conductor", "alps", "jack", "date", "europe"],
        # ["cricket", "pirate", "day", "platypus", "pants"],
        # ["plane", "loch ness", "tooth", "nurse", "laser"],
        # ["match", "hawk", "life", "knife", "africa"],
    ]
    blue_words = [
        # None,
        # None,
        # None,
        # None,
        # None,
        # ["racket", "bug", "crown", "australia", "pipe"],
        # ["scuba diver", "play", "roulette", "table", "cloak"],
        # ["buffalo", "diamond", "kid", "witch", "swing"],
        # ["gas", "circle", "king", "unicorn", "cliff"],
        # ["lemon", "death", "conductor", "litter", "car"],
        ['key', 'piano', 'lab', 'school', 'lead', #],
        'whip', 'tube', 'vacuum', 'lab', 'moon'],
        # ['change', 'litter', 'scientist', 'worm', 'row'],
        # ['boot', 'figure', 'cricket', 'ball', 'nut'],
        # ['bear', 'figure', 'swing', 'shark', 'stream'],
        # ["jupiter", "moon"], #"pipe", "racket", "bug"],
        # ["phoenix", "beijing"], #"play", "table", "cloak"],
        # ["bear", "bison"], #"diamond", "witch", "swing"],
        # ["cap", "boot"], #"circle", "unicorn", "cliff"],
        # ["india", "germany"], #"death", "litter", "car"],
    ]

    for i, (red, blue) in enumerate(zip(red_words, blue_words)):

        game._build_game(red=red, blue=blue, save_path="tmp_babelnet_"+str(i))
        print("")
        print("TRIAL ", str(i), ":")
        print("RED WORDS: ", list(game.red_words))
        print("BLUE WORDS: ", list(game.blue_words))
        # TODO: Download version without using aliases. They may be too confusing
        if game.verbose:
            print("NEAREST NEIGHBORS:")
            for word, clues in game.weighted_nn.items():
                print(word)
                print(sorted(clues, key=lambda k: clues[k], reverse=True)[:5])

        best_scores, best_clues = game.get_clue(2, 1)
        print("BEST CLUES: ")
        for score, clues in zip(best_scores, best_clues):
            print(score, clues)
            for clue in clues:
                print(
                    "WORDS CHOSEN FOR CLUE: ",
                    game.choose_words(2, clue, game.blue_words.union(game.red_words)),
                )

        all_words = red + blue
        for word in all_words:
            game.draw_graph(game.graphs[word], word+"_hypernyms", get_labels=True)
