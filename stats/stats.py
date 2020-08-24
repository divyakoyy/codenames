import os
import csv
from tabulate import tabulate
import pprint

def generate_stats(input_file_paths, amt_results_file_paths):
	assert(len(input_file_paths) == len(amt_results_file_paths))
	input_dict = dict()
	"""
	{ 'embedding_name' : { 'clue':, 'word0ForClue':, 'word1ForClue':, 'blueWords':[], }
	"""
	for i in range(0,len(input_file_paths)):
		input_file_path = input_file_paths[i]
		with open(input_file_path, 'r', newline='') as csvfile:
			reader = csv.DictReader(csvfile)
			for row in reader:
				blue_words = []
				for x in range(0,10):
					blue_word_column_name = 'blueWord' + str(x)
					blue_words.append(row[blue_word_column_name])
				# e.g. word2vecWithHeuristicsTrial1.2
				input_dict[row['embedding_name']+'.'+str(i)] = {
					'clue' : row['clue'],
					'word0ForClue' : row['word0ForClue'],
					'word1ForClue' : row['word1ForClue'],
					'blueWords' : blue_words,

			}

	pprint.pprint(input_dict.keys())

	# Generate stats
	embedding_keys = [
		'word2vecWithHeuristics', 
		'word2vecWithoutHeuristics',
		'gloveWithHeuristics', 
		'gloveWithoutHeuristics',
		'fasttextWithHeuristics', 
		'fasttextWithoutHeuristics',
		'bertWithHeuristics', 
		'bertWithoutHeuristics',
		'babelnetWithHeuristics', 
		'babelnetWithoutHeuristics',
	]
	results_dict = dict()
	for key in embedding_keys:
		results_dict[key] = { 'intendedWordPrecisionAt2': [], 'intendedWordRecallAt4': [], 'blueWordPrecisionAt2': [], 'blueWordPrecisionAt4': [] }

	for x in range(0,len(amt_results_file_paths)):
		amt_results_file_path = amt_results_file_paths[x]

		with open(amt_results_file_path, 'r', newline='') as csvfile:
			reader = csv.DictReader(csvfile)

			
			for row in reader:

				embedding_name_in_csv = row['Input.embedding_name']
				embedding_name = embedding_name_in_csv + '.'+str(x)

				expected_words = set([input_dict[embedding_name]['word0ForClue'], input_dict[embedding_name]['word1ForClue']])
				blue_words = set(input_dict[embedding_name]['blueWords'])

				answers = [row['Answer.rank1'], row['Answer.rank2'], row['Answer.rank3'], row['Answer.rank4']]
				embedding_key = embedding_name[:embedding_name.find('Trial')]
				print("Original embedding name", row['Input.embedding_name'], "Embedding name lookup in input", embedding_name, "Embedding key", embedding_key)
				print("EXPECTED WORDS", expected_words, "ANSWERS", answers, "BLUE WORDS", blue_words)

				# “The word intended”, precision at 2 = recall at 2 (# of intended words chosen in the first 2 ranks/2)
				num_intended_words_chosen_in_first_two_ranks = len(expected_words.intersection(set(answers[:2])))
				results_dict[embedding_key]['intendedWordPrecisionAt2'].append(num_intended_words_chosen_in_first_two_ranks/2.0)
				print(num_intended_words_chosen_in_first_two_ranks, results_dict[embedding_key]['intendedWordPrecisionAt2'])

				# “The word intended”, recall at 4 (# of intended words chosen in the first 4 ranks/2)
				num_intended_words_chosen_in_all_four_ranks = len(expected_words.intersection(set(answers)))
				results_dict[embedding_key]['intendedWordRecallAt4'].append(num_intended_words_chosen_in_all_four_ranks/2.0)

				# “Any blue word”, precision at 2 (# of blue words chosen in the first 2 ranks/2)
				num_blue_words_chosen_in_first_two_ranks = len(blue_words.intersection(set(answers[:2])))
				results_dict[embedding_key]['blueWordPrecisionAt2'].append(num_blue_words_chosen_in_first_two_ranks/2.0)

				# “Any blue word”, precision at 4  (# of blue words chosen in the first 4 ranks/# of ranks selected)
				num_blue_words_chosen_in_all_four_ranks = len(blue_words.intersection(set(answers)))
				num_ranks_selected = float(len([answer for answer in answers if answer != 'noMoreRelatedWords']))
				results_dict[embedding_key]['blueWordPrecisionAt4'].append(num_blue_words_chosen_in_all_four_ranks/num_ranks_selected)
				print()
	avg_stats = dict()
	stat_types = ['intendedWordPrecisionAt2', 'intendedWordRecallAt4', 'blueWordPrecisionAt2', 'blueWordPrecisionAt4']

	l = []
	
	for embedding_key in results_dict:
		avg_stats[embedding_key] = dict()
		for stat_metric in results_dict[embedding_key]:
			stats_list = results_dict[embedding_key][stat_metric]
			avg_stats[embedding_key][stat_metric] = sum(stats_list)/len(stats_list)

		row = [embedding_key] + [avg_stats[embedding_key][stat_type] for stat_type in stat_types]
		l.append(row)

	table = tabulate(l, headers=['embedding_algorithm'] + stat_types, tablefmt='orgtbl')

	print(table)

if __name__=='__main__':
	input_file_paths = ['../data/amt_key_0.csv', '../data/amt_key_1.csv']
	amt_results_file_paths = ['../data/Batch_291089_batch_results.csv', '../data/Batch_291092_batch_results.csv']

	# input_file_paths = ['../data/amt_test_0.csv']
	# amt_results_file_paths = ['../data/batch_results_test_0.csv']
	generate_stats(input_file_paths, amt_results_file_paths)
