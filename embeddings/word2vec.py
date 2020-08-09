
from gensim.models import KeyedVectors


class Word2Vec(object):

    def __init__(self, configuration=None):


        # Initialize variables
        self.configuration = configuration
        self.graphs = dict() #TODO: visualizations for other embedding methods

        self.word2vec_model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)


    """
    Required codenames methods
    """

    def get_weighted_nn(self, word, n=500):
        nn_w_similarities = dict()
        limit = 1

        def recurse_word2vec(word, curr_limit):
            if curr_limit >= limit or word not in self.word2vec_model.vocab:
                return
            neighbors_and_similarities = self.word2vec_model.most_similar(word, topn=n)
            for neighbor, similarity in neighbors_and_similarities:
                if (self.word2vec_model.vocab[neighbor].count < 2 or len(neighbor.split("_")) > 1):
                    continue
                neighbor = neighbor.lower()
                if neighbor not in nn_w_similarities:
                    nn_w_similarities[neighbor] = similarity
                    recurse_word2vec(neighbor, curr_limit + 1)
                nn_w_similarities[neighbor] = max(similarity, nn_w_similarities[neighbor])

        recurse_word2vec(word, 0)

        return {k: v for k, v in nn_w_similarities.items() if k != word}

    def rescale_score(self, chosen_words, potential_clue, red_words):
        """
        :param chosen_words: potential board words we could apply this clue to
        :param clue: potential clue
        :param red_words: opponent's words
        returns: penalizes a potential_clue for being have high word2vec similarity with opponent's words
        """
        word2vec_similarities = []
        red_word2vec_similarities = []
        if potential_clue not in self.word2vec_model:
            if self.configuration.verbose:
                print("Potential clue word ", potential_clue, "not in Google news word2vec model")
            return 0.0

        for red_word in red_words:
            if red_word in self.word2vec_model:
                red_word2vec_similarities.append(self.word2vec_model.similarity(red_word, potential_clue))

        if self.configuration.debug_file:
            with open(self.configuration.debug_file, 'a') as f:
                f.write(" ".join([str(x) for x in [
                    " penalty for red words word2vec", round(-0.5*sum(red_word2vec_similarities)/len(red_word2vec_similarities),3), "\n"
                ]]))
        #TODO: is average the best way to do this
        return -0.5*sum(red_word2vec_similarities)/len(red_word2vec_similarities)
