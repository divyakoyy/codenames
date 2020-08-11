
from gensim.models import KeyedVectors


class Glove(object):

    def __init__(self, configuration=None):

        # Initialize variables
        self.configuration = configuration

        self.glove_model = KeyedVectors.load_word2vec_format('data/glove-wiki-gigaword-300.txt.gz')

    """
    Required codenames methods
    """

    def get_weighted_nn(self, word, n=500):
        nn_w_similarities = dict()

        if word not in self.glove_model.vocab:
            return
        neighbors_and_similarities = self.glove_model.most_similar(word, topn=n)
        for neighbor, similarity in neighbors_and_similarities:
            if len(neighbor.split("_")) > 1:
                continue
            neighbor = neighbor.lower()
            if neighbor not in nn_w_similarities:
                nn_w_similarities[neighbor] = similarity
            nn_w_similarities[neighbor] = max(similarity, nn_w_similarities[neighbor])

        return {k: v for k, v in nn_w_similarities.items() if k != word}

    def rescale_score(self, chosen_words, potential_clue, red_words):
        """
        :param chosen_words: potential board words we could apply this clue to
        :param clue: potential clue
        :param red_words: opponent's words
        returns: penalizes a potential_clue for being have high word2vec similarity with opponent's words
        """
        glove_similarities = []
        red_glove_similarities = []
        if potential_clue not in self.glove_model:
            if self.configuration.verbose:
                print("Potential clue word ", potential_clue, "not in Google news word2vec model")
            return 0.0

        for red_word in red_words:
            if red_word in self.glove_model:
                red_glove_similarities.append(self.glove_model.similarity(red_word, potential_clue))

        if self.configuration.debug_file:
            with open(self.configuration.debug_file, 'a') as f:
                f.write(" ".join([str(x) for x in [
                    " glove penalty for red words:", round(-0.5*sum(red_glove_similarities)/len(red_glove_similarities),3), "\n"
                ]]))
        #TODO: is average the best way to do this
        return -0.5*sum(red_glove_similarities)/len(red_glove_similarities)
