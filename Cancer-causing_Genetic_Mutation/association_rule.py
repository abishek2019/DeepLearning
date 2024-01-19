from tabulate import tabulate

def findQuadRules(f4: list, samplesIndex: list, df: any):
	association_rules = []
	for each in f4:
		A = each[0]
		B = each[1]
		C = each[2]
		D = each[3]
		count = 0
		freq_BCD = 0
		freq_ACD = 0
		freq_ABD = 0
		freq_ABC = 0
		for i in samplesIndex:
			if df.loc[i][A] == 1 and df.loc[i][B] == 1 and df.loc[i][C] == 1\
			and df.loc[i][D] == 1:
				count += 1
			if df.loc[i][B] == 1 and df.loc[i][C] == 1 and df.loc[i][D] == 1:
				freq_BCD += 1
			if df.loc[i][A] == 1 and df.loc[i][C] == 1 and df.loc[i][D] == 1:
				freq_ACD += 1
			if df.loc[i][A] == 1 and df.loc[i][B] == 1 and df.loc[i][D] == 1:
				freq_ABD += 1
			if df.loc[i][A] == 1 and df.loc[i][B] == 1 and df.loc[i][C] == 1:
				freq_ABC += 1
		support =  count / 162
		conf_consq_D = count / freq_ABC
		conf_consq_C = count / freq_ABD
		conf_consq_B = count / freq_ACD
		conf_consq_A = count / freq_BCD

		# Generate rules
		association_rules.append({'Ante1': A, 'Ante2': B, 'Ante3': C, 'Cons': D,'sup': round(support * 100, 1),\
			'conf': round(conf_consq_D * 100, 1), 'S*C': round(support * conf_consq_D, 3)})

		association_rules.append({'Ante1': A, 'Ante2': B, 'Ante3': D, 'Cons': C, 'sup': round(support * 100, 1),\
			'conf': round(conf_consq_C * 100, 1), 'S*C': round(support * conf_consq_C, 3)})

		association_rules.append({'Ante1': A, 'Ante2': C, 'Ante3': D, 'Cons': B, 'sup': round(support * 100, 1),\
			'conf': round(conf_consq_B * 100, 1), 'S*C': round(support * conf_consq_B, 3)})

		association_rules.append({'Ante1': B, 'Ante2': C, 'Ante3': D, 'Cons': A, 'sup': round(support * 100, 1),\
			'conf': round(conf_consq_A * 100, 1), 'S*C': round(support * conf_consq_A, 3)})
	sortMergeFinalRules(association_rules, 'quadruplets')

def findQuintRules(f5, samplesIndex, df):
	association_rules = []
	for each in f5:
		A = each[0]
		B = each[1]
		C = each[2]
		D = each[3]
		E = each[4]
		count = 0
		freq_BCDE = 0
		freq_ACDE = 0
		freq_ABDE = 0
		freq_ABCE = 0
		freq_ABCD = 0
		for i in samplesIndex:
			if df.loc[i][A] == 1 and df.loc[i][B] == 1 and df.loc[i][C] == 1\
			and df.loc[i][D] and df.loc[i][E] == 1:
				count += 1
			if df.loc[i][B] == 1 and df.loc[i][C] == 1 and df.loc[i][D] == 1 and df.loc[i][E] == 1:
				freq_BCDE += 1
			if df.loc[i][A] == 1 and df.loc[i][C] == 1 and df.loc[i][D] == 1 and df.loc[i][E] == 1:
				freq_ACDE += 1
			if df.loc[i][A] == 1 and df.loc[i][B] == 1 and df.loc[i][D] == 1 and df.loc[i][E] == 1:
				freq_ABDE += 1
			if df.loc[i][A] == 1 and df.loc[i][B] == 1 and df.loc[i][C] == 1 and df.loc[i][E] == 1:
				freq_ABCE += 1
			if df.loc[i][A] == 1 and df.loc[i][B] == 1 and df.loc[i][C] == 1 and df.loc[i][D] == 1:
				freq_ABCD += 1
		support =  count / 162
		conf_consq_E = count / freq_ABCD
		conf_consq_D = count / freq_ABCE
		conf_consq_C = count / freq_ABDE
		conf_consq_B = count / freq_ACDE
		conf_consq_A = count / freq_BCDE

		# Generate rules
		association_rules.append({'Ante1': A, 'Ante2': B, 'Ante3': C, 'Ante4': D, 'Cons': E,\
		 'sup': round(support * 100, 1), 'conf': round(conf_consq_E * 100, 1),\
		 'S*C': round(support * conf_consq_E, 3)})

		association_rules.append({'Ante1': A, 'Ante2': B, 'Ante3': C, 'Ante4': E, 'Cons': D,\
		 'sup': round(support * 100, 1), 'conf': round(conf_consq_D * 100, 1),\
		 'S*C': round(support * conf_consq_D, 3)})

		association_rules.append({'Ante1': A, 'Ante2': B, 'Ante3': D, 'Ante4': E, 'Cons': C,\
		 'sup': round(support * 100, 1), 'conf': round(conf_consq_C * 100, 1),\
		 'S*C': round(support * conf_consq_C, 3)})

		association_rules.append({'Ante1': A, 'Ante2': C, 'Ante3': D, 'Ante4': E, 'Cons': B,\
		 'sup': round(support * 100, 1), 'conf': round(conf_consq_B * 100, 1),\
		 'S*C': round(support * conf_consq_B, 3)})

		association_rules.append({'Ante1': B, 'Ante2': C, 'Ante3': D, 'Ante4': E, 'Cons': A,\
		 'sup': round(support * 100, 1),'conf': round(conf_consq_A * 100, 1),\
		 'S*C': round(support * conf_consq_A, 3)})
	sortMergeFinalRules(association_rules, 'quintuplets')


def sortMergeFinalRules(association_rules: list, setType: str):	
		# Sorting the rules by values
		sorted_support = sorted(association_rules, key=lambda d: d['sup'], reverse=True) 
		sorted_confidence = sorted(association_rules, key=lambda d: d['conf'], reverse=True) 

		# Setting the minimum thresholds: support >= 1.9% and confidence > 80%
		threshold = {'sup': '>= 1.9%', 'conf': '> 80%'}
		filtered_support = [data for data in sorted_support if data['sup'] >= 1.9]
		filtered_confidence = [data for data in sorted_confidence if data['conf'] > 80]

		# Merging the two filtered rules
		final_rules = filtered_support
		for data in filtered_confidence:
			exists = False 
			for rule in final_rules:
				if setType == 'quadruplets':
					if data['Ante1'] == rule['Ante1'] and data['Ante2'] == rule['Ante2']\
					and data['Ante3'] == rule['Ante3'] and data['Cons'] == rule['Cons']:
						exists = True
						break
				if setType == 'quintuplets':
					if data['Ante1'] == rule['Ante1'] and data['Ante2'] == rule['Ante2']\
					and data['Ante3'] == rule['Ante3'] and data['Ante4'] == rule['Ante4']\
					and data['Cons'] == rule['Cons']:
						exists = True
						break
			if not exists:
				final_rules.append(data)
		sorted_SxC = sorted(final_rules, key=lambda d: d['S*C'], reverse=True) 
		print_sorted(sorted_SxC, 'SUPPORT * CONFIDENCE', setType, threshold)

# print the rules
def print_sorted(rules: list, text: str, string: str, threshold: dict):
	print(f'\n->Association rules for {string} after sorting by  {text}:-\n')
	if threshold:
		print(f"Selected threshold: Support {threshold['sup']}\tConfidence {threshold['conf']}\n")
	j = 0
	arr = []
	for elem in rules:
		j += 1
		if string == 'quadruplets':
			ante = f"{elem['Ante1'][:6]}, {elem['Ante2'][:6]} & {elem['Ante3'][:6]}"
		else:
			ante = f"{elem['Ante1'][:6]}, {elem['Ante2'][:6]}, {elem['Ante3'][:6]} & {elem['Ante4'][:6]}"
		arr.append([j, f"If {ante},", f"then {elem['Cons'][:6]}", f"{elem['sup']}%",\
			f"{elem['conf']}%", elem['S*C']])

	arr.insert(0, ['Index', 'If Antecedent', 'then Consequent', 'Support', 'Confidence', 'S*C'])
	print(tabulate(arr))
    	











