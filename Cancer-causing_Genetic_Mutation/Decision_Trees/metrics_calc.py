def find_metrices(basic_metrics: dict):
    tp = basic_metrics['tp']
    fp = basic_metrics['fp']
    tn = basic_metrics['tn']
    fn = basic_metrics['fn']
    handle_0 = 0.000000001
    accuracy = (tp + tn) / 54
    sensitivity = tp / (tp + fn + handle_0)
    specificity = tn / (tn + fp + handle_0)
    precision = tp / (tp + fp + handle_0)
    miss_rate = fn / (fn + tp + handle_0)
    fdr = fp / (fp + tp + handle_0)
    fomr = fn / (fn + tn + handle_0)
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'accuracy': round(accuracy * 100, 2),
    'sensitivity': round(sensitivity * 100, 2),'specificity': round(specificity * 100, 2), 
    'precision': round(precision * 100, 2), 'miss_rate': round(miss_rate * 100, 2), 
    'fdr': round(fdr * 100, 2), 'fomr': round(fomr * 100, 2)}