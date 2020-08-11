# TODO Anna: Do we need all these prior to v5 methods anymore?

self.sess = requests.Session()
self.wikipedia_url = "https://en.wikipedia.org/w/api.php"  


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
            if self.configuration.verbose:
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