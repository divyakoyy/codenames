import os
import csv


def name_for_with_trial_from_without_trial(embedding_name):
	# If the without trial does not exist, try the with trial since it has the same clue
	start_idx = embedding_name.find('Heuristics')
	return embedding_name[0:start_idx-3] + embedding_name[start_idx:]

def prefill_without_trials_using_with_trials(input_file_path, amt_results_file_path):
	input_keys = []
	print ("Processing", input_file_path, amt_results_file_path)
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
			print(mapped_key,"not in", amt_results_file_path)
			continue
		embedding_name_to_row_dict[mapped_key][27] = missing_key
		
		with open(amt_results_file_path, 'a', newline='') as csvfile:
			writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			writer.writerow(embedding_name_to_row_dict[mapped_key])

if __name__ == '__main__':
	key_input_file_paths = ['../data/amt_0825_batch0_key.csv', '../data/amt_0825_batch1_key.csv', '../data/amt_0826_batch0_key.csv', '../data/amt_0826_batch1_key.csv', '../data/amt_0826_batch2_key.csv']
	amt_results_file_paths = ['../data/amt_0825_batch0_results.csv', '../data/amt_0825_batch1_results.csv', '../data/amt_0826_batch0_results.csv', '../data/amt_0826_batch1_results.csv', '../data/amt_0826_batch2_results.csv']

	for key_input_file_path, amt_results_file_path in zip(key_input_file_paths, amt_results_file_paths):
		prefill_without_trials_using_with_trials(key_input_file_path, amt_results_file_path)