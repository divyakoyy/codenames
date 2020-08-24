import os
import csv


def name_for_with_trial_from_without_trial(embedding_name):
	# If the without trial does not exist, try the with trial since it has the same clue
	start_idx = embedding_name.find('Heuristics')
	return embedding_name[0:start_idx-3] + embedding_name[start_idx:]

def prefill_without_trials_using_with_trials(input_file_path, amt_results_file_path):
	input_keys = []
	with open(input_file_path, 'r', newline='') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:	
			input_keys.append(row['embedding_name'])

	embedding_name_to_row_dict = dict()
	with open(amt_results_file_path, 'r', newline='') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:	
			embedding_name_to_row_dict[row['Input.embedding_name']] = list(row.values())

	missing_embedding_keys = set(input_keys).difference(set(embedding_name_to_row_dict.keys()))
	print(len(missing_embedding_keys), "missing keys in", amt_results_file_path, ":", missing_embedding_keys)

	for missing_key in missing_embedding_keys:
		mapped_key = name_for_with_trial_from_without_trial(missing_key)
		# no AMT response for this trial
		if (mapped_key not in embedding_name_to_row_dict): 
			print(mapped_key,"not in AMT results")
			continue
		embedding_name_to_row_dict[mapped_key][27] = missing_key
		
		with open(amt_results_file_path, 'a', newline='') as csvfile:
			writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			writer.writerow(embedding_name_to_row_dict[mapped_key])

if __name__ == '__main__':
	input_file_path = '../data/amt_key_1.csv'
	amt_results_file_path = '../data/Batch_291092_batch_results.csv'

	prefill_without_trials_using_with_trials(input_file_path, amt_results_file_path)