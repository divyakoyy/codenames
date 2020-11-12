
from annoy import AnnoyIndex
import numpy as np

class Bert(object):

	def __init__(self, configuration=None):


		# Initialize variables
		self.configuration = configuration
		self.graphs = dict() #TODO: visualizations for other embedding methods

		emb_size = 768 # bert_12_768_12, book_corpus_wiki_en_cased
		self.bert_annoy_tree = AnnoyIndex(emb_size, 'angular')
		self.bert_annoy_tree.load('data/annoy_tree_bert_emb_768_text8_small.ann')
		self.bert_annoy_tree_idx_to_word = np.load('data/annoy_tree_index_to_word_bert_emb_768_text8_small.npy', allow_pickle=True).item()
		self.bert_annoy_tree_word_to_idx = {v: k for k, v in self.bert_annoy_tree_idx_to_word.items()}

	"""
	Required codenames methods
	"""

	def get_weighted_nn(self, word, n=500):
		nn_w_similarities = dict()

		if word not in self.bert_annoy_tree_word_to_idx:
			return nn_w_similarities

		annoy_idx = self.bert_annoy_tree_word_to_idx[word]
		neigbors_and_distances = self.bert_annoy_tree.get_nns_by_item(annoy_idx, n, include_distances=True)

		for neighbor_annoy_idx, distance in zip(neigbors_and_distances[0], neigbors_and_distances[1]):
			neighbor_word = self.bert_annoy_tree_idx_to_word[neighbor_annoy_idx].lower()
			if len(neighbor_word.split("_")) > 1 or len(neighbor_word.split("-")) > 1:
				continue
			similarity = 1 - (distance ** 2 / 2)
			#print("Word:",word, "Neighbor:",neighbor_word, "Similarity:",similarity)
			if neighbor_word not in nn_w_similarities:
				nn_w_similarities[neighbor_word] = similarity
			nn_w_similarities[neighbor_word] = max(similarity, nn_w_similarities[neighbor_word])

		return {k: v for k, v in nn_w_similarities.items() if k != word}

	def rescale_score(self, chosen_words, potential_clue, red_words):
		"""
		:param chosen_words: potential board words we could apply this clue to
		:param potential_clue: potential clue
		:param red_words: opponent's words
		returns: penalizes a potential_clue for being have high bert similarity with opponent's words
		"""
		max_red_similarity = float("-inf")
		if potential_clue not in self.bert_annoy_tree_word_to_idx:
			if self.configuration.verbose:
				print("Potential clue word ", potential_clue, "not in bert model")
			return 0.0

		for red_word in red_words:
			if red_word in self.bert_annoy_tree_word_to_idx:
				similarity = self.get_word_similarity(red_word, potential_clue)
				if similarity > max_red_similarity:
					max_red_similarity = similarity

		if self.configuration.debug_file:
			with open(self.configuration.debug_file, 'a') as f:
				f.write(" ".join([str(x) for x in [
					" bert penalty for red words:", max_red_similarity, "\n"
				]]))
		return -0.5*max_red_similarity


	def dict2vec_embedding_weight(self):
		return 2.0

	def get_word_similarity(self, word1, word2):
		try:
			# cosine distance = sqrt(2(1-*cos(u, v)), as calculated from Annoy. see https://github.com/spotify/annoy for reference.
			angular_dist = self.bert_annoy_tree.get_distance(self.bert_annoy_tree_word_to_idx[word1], self.bert_annoy_tree_word_to_idx[word2])
			return 1 - (angular_dist**2 / 2)
		except KeyError:
			return -1.0
