import pandas as pd
import randomizer 
import find_entropy
import decision_tree_classifier 
import metrics_calc 
import evaluator 
import print_class 
import sample_divider 
import print_class 


def main():
	k = 3
	df = pd.read_csv('mutations.csv')
	samples =  [df.loc[i][df.columns[0]] for i in range(162)]
	df.drop('Samples', axis=1, inplace=True)
	k_samples = randomizer.randomize(samples, k)
	best_features_list = []
	metrics_list = []
	for i in range(k):
		f = find_entropy.best_features(k_samples['training_list'][i], samples, df, 'f')
		f_subsamples = sample_divider.divide_to_subsamples(k_samples['training_list'][i], samples, f['feature'], df)
		a = find_entropy.best_features(f_subsamples['has_feature'], samples, df, 'a')
		b = find_entropy.best_features(f_subsamples['has_no_feature'], samples, df, 'b')
		test_set_classified = decision_tree_classifier.decision_tree_classify(k_samples['test_list'][i], samples,\
		f['feature'], a['feature'], b['feature'], df)
		basic_metrics = evaluator.evaluator(test_set_classified)
		metrics = metrics_calc.find_metrices(basic_metrics)
		metrics_list.append(metrics)
		best_features_list.append([f, a, b]) 
	averages = find_averages(metrics_list, k)
	print_class.print_function(best_features_list, metrics_list, averages, k)

def find_averages(metrics_list: list, k: int):
    total_acc = total_sens = total_spec = total_prec = total_mr = total_fdr = total_fomr =0
    for i in range(k):
        metrices = metrics_list[i]
        total_acc += metrices['accuracy']
        total_sens += metrices['sensitivity']
        total_spec += metrices['specificity']
        total_prec += metrices['precision']
        total_mr += metrices['miss_rate']
        total_fdr += metrices['fdr']
        total_fomr += metrices['fomr']
    return {'avg_accuracy': round(total_acc / k, 2), 'avg_sensitivity': round(total_sens / k, 2),
    'avg_specificity': round(total_spec / k, 2), 'avg_precision': round(total_prec / k, 2), 
    'avg_miss_rate': round(total_mr / k, 2), 'avg_fdr': round(total_fdr / k, 2), 'avg_fomr': round(total_fomr / k, 2)}

if __name__ == '__main__':
    main()   