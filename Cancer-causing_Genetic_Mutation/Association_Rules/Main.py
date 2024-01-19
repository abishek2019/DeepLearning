# Documented by: Abishek Lakandri

import pandas as pd
import data_filter
import find_set_features as find_set
import association_rule

def main():
	k = 3
	df = pd.read_csv('mutations.csv')
	samples =  [df.loc[i][df.columns[0]] for i in range(162)]
	df.drop('Samples', axis=1, inplace=True)

	# Considering only 'C' samples
	filtered_samples = data_filter.sample_filter(samples) 
	filtered_sampIndex = [samples.index(data) for data in filtered_samples]

	# Selecting features with >= 7 samples
	feature_toFilter = data_filter.feature_filter(df, filtered_sampIndex)
	for feature in feature_toFilter:
		df.drop(feature, axis=1, inplace=True)

	#finding pairs 
	pair_value = find_set.pairs(df, filtered_sampIndex)

	#c3, f3 for triplets
	c3 = find_set.c3Construct(pair_value)
	f3 = find_set.generateF3(c3, filtered_sampIndex, df)

	#c4, f4 and rules for quadruplets
	c4 = find_set.cNConstruct(f3, 3)
	f4 = find_set.generateFN(c4, filtered_sampIndex, df, 4)
	association_rule.findQuadRules(f4, filtered_sampIndex, df)

	# c5, f5 and rules for quintuplets
	c5 = find_set.cNConstruct(f4, 4)
	f5 = find_set.generateFN(c5, filtered_sampIndex, df, 5)
	print("\nThe set of quintuplets are:-")
	for elem in f5:
		print(f"({elem[0][:6]}, {elem[1][:6]}, {elem[2][:6]}, {elem[3][:6]}, {elem[4][:6]})")
	association_rule.findQuintRules(f5, filtered_sampIndex, df)

	c6 = find_set.cNConstruct(f5, 5)
	print(f'c6: {len(c6)}')

if __name__ == '__main__':
    main()   