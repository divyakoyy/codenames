import os
import csv
from tabulate import tabulate
import pprint
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
import statistics
import math

'''
Pre-processing. 
If a given trial (i.e. a unique set of board words with a clue) produces the
same clue for the embedding and embedding+DictionaryRelevance, we only provide that clue once to AMT.
Assuming that an Amazon turker will provide the same answer for the same board words + clue, we use the 
result of that HIT in both the embedding and embedding+DictionaryRelevance bucket. The following method 
pre-processes the amt_results csvs to make the statistics logic simpler.
'''
def name_for_with_trial_from_without_trial(embedding_name):
	# If the without trial does not exist, try the with trial since it has the same clue
	start_idx = embedding_name.find('Heuristics')
	return embedding_name[0:start_idx-3] + embedding_name[start_idx:]


def prefill_without_trials_using_with_trials(input_file_path, amt_results_file_path):
	input_keys_to_clue = {}
	print ("Processing", input_file_path, amt_results_file_path)
	with open(input_file_path, 'r', newline='') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:  
			input_keys_to_clue[row['embedding_name']] = row['clue']

	trial_name_to_row_dict = dict()
	with open(amt_results_file_path, 'r', newline='') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:  
			trial_name_to_row_dict[row['Input.embedding_name']] = list(row.values())

	missing_embedding_keys = set(input_keys_to_clue.keys()).difference(set(trial_name_to_row_dict.keys()))
	print(len(missing_embedding_keys), "missing keys in", amt_results_file_path, ":", missing_embedding_keys)

	seen_keys = list(trial_name_to_row_dict.keys())
	for missing_key in missing_embedding_keys:

		embedding_name = missing_key[0:missing_key.find("With")] # e.g. bert
		trial_number = missing_key[missing_key.find("Trial")+5:] # e.g. 5
		for trial_name in trial_name_to_row_dict:
			clue = trial_name_to_row_dict[trial_name][28] # 27 = Input.clue
			if trial_name.startswith(embedding_name) and trial_name.endswith(trial_number) and clue == input_keys_to_clue[missing_key] and missing_key not in seen_keys:
				print(trial_name, "maps to", missing_key)
				trial_name_to_row_dict[trial_name][27] = missing_key # 27 = Input.embedding_name
				seen_keys.append(missing_key)
		
				with open(amt_results_file_path, 'a', newline='') as csvfile:
					writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
					writer.writerow(trial_name_to_row_dict[trial_name])

		# mapped_key = name_for_with_trial_from_without_trial(missing_key)
		# # no AMT response for this trial
		# if (mapped_key not in embedding_name_to_row_dict):
		#   print(mapped_key,"not in", amt_results_file_path)
		#   continue
		# embedding_name_to_row_dict[mapped_key][27] = missing_key
		
		# with open(amt_results_file_path, 'a', newline='') as csvfile:
		#   writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		#   writer.writerow(embedding_name_to_row_dict[mapped_key])

'''
Statistics
'''

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
	representations = [
		'word2vec', 'glove', 'fasttext', 'kim2019', 'babelnet', 'bert',
	]

	renamed_embedding_keys = dict()
	embedding_keys = []
	for x in ["WithHeuristics", "WithoutHeuristics"]:
		for y in ["KimFx", "WithoutKimFx"]:
			for rep in representations:
				embedding_key = rep+x+y
				embedding_keys.append(rep+x+y)
				renamed_key = rep + ("+DictRelevance" if x == "WithHeuristics" else "") + ("+KimFx" if y == "KimFx" else "")
				renamed_embedding_keys[embedding_key] = renamed_key

	print(embedding_keys, renamed_embedding_keys)

	results_dict = dict()
	for key in embedding_keys:
		results_dict[key] = { 'intendedWordPrecisionAt2': [], 'intendedWordRecallAt4': [], 'blueWordPrecisionAt2': [], 'blueWordPrecisionAt4': [] }

	num_trials = 0
	for x in range(0,len(amt_results_file_paths)):
		amt_results_file_path = amt_results_file_paths[x]

		with open(amt_results_file_path, 'r', newline='') as csvfile:
			reader = csv.DictReader(csvfile)

			print("Getting stats for", amt_results_file_path)
			for row in reader:

				embedding_name_in_csv = row['Input.embedding_name']
				embedding_name = embedding_name_in_csv + '.'+str(x)

				expected_words = set([input_dict[embedding_name]['word0ForClue'], input_dict[embedding_name]['word1ForClue']])
				blue_words = set(input_dict[embedding_name]['blueWords'])

				answers = [row['Answer.rank1'], row['Answer.rank2'], row['Answer.rank3'], row['Answer.rank4']]
				embedding_key = embedding_name[:embedding_name.find('Trial')]
				#print("Original embedding name", row['Input.embedding_name'], "Embedding name lookup in input", embedding_name, "Embedding key", embedding_key)
				#print("EXPECTED WORDS", expected_words, "ANSWERS", answers, "BLUE WORDS", blue_words)

				# “The word intended”, precision at 2 = recall at 2 (# of intended words chosen in the first 2 ranks/2)
				num_intended_words_chosen_in_first_two_ranks = len(expected_words.intersection(set(answers[:2])))
				results_dict[embedding_key]['intendedWordPrecisionAt2'].append(num_intended_words_chosen_in_first_two_ranks/2.0)
				#print(num_intended_words_chosen_in_first_two_ranks, results_dict[embedding_key]['intendedWordPrecisionAt2'])

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
				num_trials += 1
	stat_types = ['intendedWordPrecisionAt2', 'intendedWordRecallAt4', 'blueWordPrecisionAt2', 'blueWordPrecisionAt4']

	l = []

	print("Number of Trials:", num_trials)

	results_dict = { renamed_embedding_keys[embedding] : results_dict[embedding] for embedding in results_dict}
	
	avg_stats = dict()

	for embedding_key in results_dict:
		trials = 0
		avg_stats[embedding_key] = dict()
		for stat_metric in results_dict[embedding_key]:
			stats_list = results_dict[embedding_key][stat_metric]
			avg_stats[embedding_key][stat_metric] = sum(stats_list)/len(stats_list)
			trials += len(stats_list)
		print(embedding_key, "trials", trials/4)

		row = [embedding_key] + [avg_stats[embedding_key][stat_type] for stat_type in stat_types]
		l.append(row)


	table = tabulate(l, headers=['embedding_algorithm'] + stat_types, tablefmt='orgtbl')

	print(table)

	keys = {'word2vec':None, 
		  'glove':None,
		  'fasttext':None,
		  'bert':None,
		  'babelnet':None,
		  'kim2019':None}

	# Wilcoxon signed rank test

	print("************ Our scoring fx ***************")

	for embedding_key in keys:
		for stat_metric in stat_types:
			# Compare representation vs. representation+DictRelevance (our scoring function)
			print(stat_metric, embedding_key, "vs.", embedding_key+"+DictRelevance")
			stat = stats.wilcoxon(results_dict[embedding_key][stat_metric], results_dict[embedding_key+"+DictRelevance"][stat_metric])
			print(stat)  
			
			# z, pval = z_score(results_dict[embedding_key+"+DictRelevance"][stat_metric], results_dict[embedding_key][stat_metric])
			# print("z score", z, "p", pval)
			# print()
		print("=========================================================\n")

	print("\n************ Kim scoring fx ***************")

	for embedding_key in keys:
		for stat_metric in stat_types:
			# Compare representation vs. representation+DictRelevance (our scoring function)
			print(stat_metric, embedding_key+"+KimFx", "vs.", embedding_key+"+DictRelevance+KimFx")
			stat = stats.wilcoxon(results_dict[embedding_key+"+KimFx"][stat_metric], results_dict[embedding_key+"+DictRelevance+KimFx"][stat_metric])
			print(stat)  

			# z, pval = z_score(results_dict[embedding_key+"+DictRelevance+KimFx"][stat_metric], results_dict[embedding_key+"+KimFx"][stat_metric])
			# print("z score", z, "p", pval)
			# print()
		print("=========================================================\n")


	population_std = {}
	# need to switch the loop order
	for stat_metric in stat_types:
		# treat all precision@2 and precision@2+DictRelevance (across all embeddings) as our populations
		all_a1 = [] 
		all_a2 = []
		for embedding_key in keys:
			all_a1 += results_dict[embedding_key][stat_metric]
			all_a2 += results_dict[embedding_key+"+DictRelevance"][stat_metric]
		# should this be statistics.pstdev
		population_std[stat_metric] = statistics.stdev([a_i - b_i for a_i, b_i in zip(all_a1, all_a2)])

		for embedding_key in keys:
			z, p = z_score(
				results_dict[embedding_key][stat_metric],
				results_dict[embedding_key+"+DictRelevance"][stat_metric],
				population_std[stat_metric],
			)
			print(embedding_key, stat_metric, "z:", z, "p:", p)

	return avg_stats

def z_score(a1, a2, population_std):
	# assuming a1 and a2 are paired
	assert len(a1) == len(a2)
	differences = [a_i - b_i for a_i, b_i in zip(a1, a2)]# shape: (num_samples)
	mean_difference = sum(differences)/len(a1)
	z = mean_difference / (population_std / math.sqrt(len(a1)))
	pval = 2*(1 - stats.norm.cdf(abs(z)))
	return z, pval

# def z_score(a1, a2):
#     population_std_1 = statistics.stdev(a1)
#     population_std_2 = statistics.pstdev(a2)
#     sample_mean_1 = sum(a1)/len(a1)
#     sample_mean_2 = sum(a2)/len(a2)
#     population_size = len(a1) + len(a2)
#     population_mean_1 = sum(a1)/population_size
#     population_mean_2 = sum(a2)/population_size

#     z_score_denom = math.sqrt((population_std_1**2/len(a1)) + (population_std_2**2/len(a2)))
#     z_score_numerator = (sample_mean_1 - sample_mean_2) - (population_mean_1 - population_mean_2)

#     z = z_score_numerator/z_score_denom
#     pval = 2*(1 - stats.norm.cdf(abs(z)))

#     return z, pval


'''
Plotting
'''
def plot(avg_stats, kimfx=False):

	x = []
	y = []
	labels = []
	colors = []

	# for embedding in avg_stats:
	#     embedding_label = embedding.replace('+DictRelevance', '')
	#     labels.append(embedding_label)
	#     x.append(avg_stats[embedding]['intendedWordPrecisionAt2'])
	#     y.append(avg_stats[embedding]['intendedWordRecallAt4'])
	#     if 'DictRelevance' in embedding:
	#         colors.append('blue')
	#     else:
	#         colors.append('green')

	# Chart for our scoring function with and without dictRelevance
	for embedding in avg_stats:
		if "+KimFx" in embedding:
			embedding_label = embedding.replace('+DictRelevance', '')
			labels.append(embedding_label)
			x.append(avg_stats[embedding]['intendedWordPrecisionAt2'])
			y.append(avg_stats[embedding]['intendedWordRecallAt4'])
			if 'DictRelevance' in embedding:
				colors.append('blue')
			else:
				colors.append('green')

	fig, ax = plt.subplots()
	scatter = ax.scatter(x, y, c=colors, alpha=0.5, s=8)

	for i, txt in enumerate(labels):
		if txt == 'babelnet' and colors[i] == 'blue':
			offset = (4,-6)
		elif txt == 'word2vec' and colors[i] == 'blue':
			offset = (4,-6)
		else:
			offset = (4,-2)
		ax.annotate(txt, 
					(x[i], y[i]), 
					fontsize=6.5, 
					textcoords="offset points", # how to position the text
					xytext=offset, 
					ha='left')

	#fig.suptitle('Intended Word Precision at 2 vs. Recall at 4', fontsize=12)
	fig.show()
	plt.xlabel('Precision at 2', fontsize=10)
	plt.ylabel('Recall at 4', fontsize=10)
	#produce a legend with the unique colors from the scatter
	legend_elements = [Line2D([0], [0], marker='o', color='w', label='WordRepresentation',
						markerfacecolor='g', markersize=7, alpha=0.5),
					   Line2D([0], [0], marker='o', color='w', label='WordRepresentation+DictionaryRelevance',
						markerfacecolor='b', markersize=7, alpha=0.5),]
	ax.legend(handles=legend_elements, loc='lower right')
	fig.savefig('precison_recall.png')

if __name__=='__main__':
	# Input keys
	# key_input_file_paths = ['../data/amt_0825_batch0_key.csv', '../data/amt_0825_batch1_key.csv', '../data/amt_0826_batch0_key.csv', '../data/amt_0826_batch1_key.csv', '../data/amt_0826_batch2_key.csv', '../data/amt_091020_kim2019_batch0_key.csv', '../data/amt_092220_bertavgemb_batch0_key.csv']
	# key_input_file_paths = ['../data/amt_092320_all_batch0_key.csv']
	# key_input_file_paths = ['../data/amt_102620_all_kim_scoring_fx_key.csv'] 
	key_input_file_paths = ['../data/amt_official_111220_all_key.csv']

	# Results from AMT
	# amt_results_file_paths = ['../data/amt_0825_batch0_results.csv', '../data/amt_0825_batch1_results.csv', '../data/amt_0826_batch0_results.csv', '../data/amt_0826_batch1_results.csv', '../data/amt_0826_batch2_results.csv', '../data/amt_091020_kim2019_batch0_results.csv', '../data/amt_092220_bertAvg_results.csv']
	# amt_results_file_paths = ['../data/amt_official_results_092320_all_batch0.csv']
	# amt_results_file_paths = ['../data/amt_102620_all_kim_scoring_fx_results.csv']
	amt_results_file_paths = ['../data/amt_official_111220_all_results.csv']

	# Pre-processing
	for key_input_file_path, amt_results_file_path in zip(key_input_file_paths, amt_results_file_paths):
		prefill_without_trials_using_with_trials(key_input_file_path, amt_results_file_path)

	# Statistics calculation
	avg_stats = generate_stats(key_input_file_paths, amt_results_file_paths)

	# Graphing
	plot(avg_stats)

	stats_csv_file_path = "amt_official_111220_statistics.csv"
	if not os.path.exists(stats_csv_file_path):
		with open(stats_csv_file_path, 'w'):
			pass

	with open(stats_csv_file_path, 'a', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		writer.writerow(["embedding", "intendedWordPrecisionAt2", "intendedWordRecallAt4", "blueWordPrecisionAt2", "blueWordPrecisionAt4" ])
		for embedding in avg_stats:
			writer.writerow([embedding, avg_stats[embedding]['intendedWordPrecisionAt2'], avg_stats[embedding]['intendedWordRecallAt4'], avg_stats[embedding]['blueWordPrecisionAt2'], avg_stats[embedding]['blueWordPrecisionAt4']])

