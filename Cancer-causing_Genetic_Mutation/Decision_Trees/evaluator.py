def evaluator(final_list: list):
	final_c_list = final_list['final_c_list']
	final_nc_list = final_list['final_nc_list']
	positives_count = ['TP' if sample[0] == 'C' else 'FP' for sample in final_c_list]
	negatives_count = ['TN' if sample[0] == 'N' else 'FN' for sample in final_nc_list]
	tp = positives_count.count('TP')
	fp = positives_count.count('FP') 
	tn = negatives_count.count('TN')
	fn = negatives_count.count('FN')
	return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}