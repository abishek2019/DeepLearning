import itertools

def pairs(df: any, samplesIndex: list):
	pair_value = []
	for f1, row1 in df.iteritems():
		for f2, row2 in df.iteritems():
			if (f1 != f2):
				count = 0
				for i in samplesIndex:
					if row1[i] == 1 and row2[i] == 1:
						count += 1
				if (count >= 3):
						p = (f1, f2)
						# Check if the pair exists in f2
						add = True;
						for elem in pair_value:
							if sorted(p) == sorted(elem['pair']):
								add = False
								break
						if add:
							pair_value.append({'pair': p, 'count' : count})
	return pair_value

def c3Construct(pair_value: list):
	c3 = []
	for elem1 in pair_value:
		pair1 = elem1['pair']
		for elem2 in pair_value:
			pair2 = elem2['pair']
			if sorted(pair1) != sorted(pair2): 
				if set(pair1).intersection(set(pair2)):
					triple = tuple(set(pair1 + pair2))
					# Check if the triple exists in c3
					add = True;
					for elem in c3:	
						if sorted(triple) == sorted(elem):
							add = False
							break
					if add:
						c3.append(triple)
	return c3
	

def generateF3(c3: list, samplesIndex: list, df: any):
	elemtoPrune = []
	for elem in c3:
		subsets = list(itertools.combinations(elem, 2))
		for each in subsets:
			A = each[0]
			B = each[1]
			count = 0
			for i in samplesIndex:
				if df.loc[i][A] == 1 and df.loc[i][B] == 1:
					count += 1
			if count < 3:
				elemtoPrune.append(elem)
				break
	for elem in elemtoPrune:
		c3.remove(elem)	
	f3 = c3
	return f3

def cNConstruct(f3: list, n: int):
	c = []
	for elem1 in f3:
		set1 = elem1
		for elem2 in f3:
			set2 = elem2
			if sorted(set1) != sorted(set2): 
				if len(set(set1).intersection(set(set2))) == (n - 1):
					finalSet = tuple(set(set1 + set2))
					# Check if the finalSet exists in c
					add = True;
					for elem in c:	
						if sorted(finalSet) == sorted(elem):
							add = False
							break
					if add:
						c.append(finalSet)
	return c

def generateFN(cN: list, samplesIndex: list, df: any, n: int):
	n = n - 1
	elemtoPrune = []
	for elem in cN:
		subsets = list(itertools.combinations(elem, n))
		for each in subsets:
			count = 0
			A = each[0]
			B = each[1]
			C = each[2]
			if n == 4:
				D = each[3]
			for i in samplesIndex:
				if n == 3:
					if df.loc[i][A] == 1 and df.loc[i][B] == 1 and df.loc[i][C] == 1:
						count += 1
				if n == 4:
					if df.loc[i][A] == 1 and df.loc[i][B] == 1 and df.loc[i][C] == 1\
					and df.loc[i][D] == 1:
						count += 1
			if count < 3:
				elemtoPrune.append(elem)
				break
	for elem in elemtoPrune:
		cN.remove(elem)	
	fN = cN
	return fN

