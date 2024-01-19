def feature_filter(df: any, samplesIndex: list):
	feature_toFilter = []
	for feature_name, row  in df.iteritems():
		n_C = 0
		for i in samplesIndex:
			if row[i] == 1:
				n_C += 1
		if n_C < 7:
			feature_toFilter.append(feature_name)
	return feature_toFilter

def sample_filter(samples: list):
	filtered_sample = []
	for sample in samples:
		if sample[0] == 'C':
			filtered_sample.append(sample)
	return filtered_sample





	

	