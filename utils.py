import re
import os
import string
import pickle
from scipy.spatial import distance

punctuation = re.compile("[" + re.escape(string.punctuation) + "]")

word_to_dict2vec_embeddings = None

def remove_punctuation(word):
	return word.translate(str.maketrans('', '', string.punctuation))

def is_single_word(word):
	return len(neighbor.split("_")) == 1 and len(neighbor.split("-")) == 1


def get_dict2vec_score(chosen_words, potential_clue, red_words):
	global word_to_dict2vec_embeddings
	if word_to_dict2vec_embeddings == None:
		try:
			input_file = open('data/word_to_dict2vec_embeddings','rb')
		except IOError:
		  print("Error: data/word_to_dict2vec_embeddings does not exist.")
		  return 0.0
		word_to_dict2vec_embeddings = pickle.load(input_file)

	dict2vec_distances = []
	red_dict2vec_distances = []

	if potential_clue not in word_to_dict2vec_embeddings:
		return 0.0

	potential_clue_embedding = word_to_dict2vec_embeddings[potential_clue]
	for chosen_word in chosen_words:
		if chosen_word in word_to_dict2vec_embeddings:
			chosen_word_embedding = word_to_dict2vec_embeddings[chosen_word]
			cosine_distance = distance.cosine(chosen_word_embedding, potential_clue_embedding)
			dict2vec_distances.append(cosine_distance)

	for red_word in red_words:
		if red_word in word_to_dict2vec_embeddings:
			red_word_embedding = word_to_dict2vec_embeddings[red_word]
			red_dict2vec_distances.append(distance.cosine(red_word_embedding, potential_clue_embedding))
	avg_distance_chosen_words = sum(dict2vec_distances)/len(dict2vec_distances)
	avg_distance_red_words = sum(red_dict2vec_distances)/len(red_dict2vec_distances)
	return 0.5 * (1 - avg_distance_chosen_words) - 0.25 * (1 - avg_distance_red_words)

def get_dict2vec_dist(word1, word2):
	global word_to_dict2vec_embeddings
	if word_to_dict2vec_embeddings == None:
		print("reloading embedding")
		try:
			input_file = open('data/word_to_dict2vec_embeddings','rb')
		except IOError:
		  print("Error: data/word_to_dict2vec_embeddings does not exist.")

		word_to_dict2vec_embeddings = pickle.load(input_file)

	if word1 not in word_to_dict2vec_embeddings or word2 not in word_to_dict2vec_embeddings:
		return 1.0

	cosine_distance = distance.cosine(word_to_dict2vec_embeddings[word1], word_to_dict2vec_embeddings[word2])

	return cosine_distance
