
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

		if word not in self.word2vec_model.vocab:
			return nn_w_similarities
		neighbors_and_similarities = self.word2vec_model.most_similar(word, topn=n)
		for neighbor, similarity in neighbors_and_similarities:
			if len(neighbor.split("_")) > 1 or len(neighbor.split("-")) > 1:
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
		max_red_similarity = float("-inf")
		if potential_clue not in self.word2vec_model:
			if self.configuration.verbose:
				print("Potential clue word ", potential_clue, "not in Google news word2vec model")
			return 0.0

		for red_word in red_words:
			if red_word in self.word2vec_model:
				similarity = self.word2vec_model.similarity(red_word, potential_clue)
				if similarity > max_red_similarity:
					max_red_similarity = similarity

		if self.configuration.debug_file:
			with open(self.configuration.debug_file, 'a') as f:
				f.write(" ".join([str(x) for x in [
					" word2vec penalty for red words:", max_red_similarity, "\n"
				]]))
		return -0.5*max_red_similarity

	def dict2vec_embedding_weight(self):
		return 2.0

	def get_word_similarity(self, word1, word2):
		try:
			return self.word2vec_model.similarity(word1, word2)
		except KeyError:
			return -1.0
