import sample_divider

def decision_tree_classify(test_list: list, samples: list, f: str, a: str, b: str, df: any):
    final_c_list = []
    final_nc_list = []
    f_classify = sample_divider.divide_to_subsamples(test_list, samples, f, df)
    a_classify = sample_divider.divide_to_subsamples(f_classify['has_feature'], samples, a, df)
    b_classify = sample_divider.divide_to_subsamples(f_classify['has_no_feature'], samples, b, df)
    a1 = find_actualClass(a_classify['has_feature'])
    a2 = find_actualClass(a_classify['has_no_feature'])
    b1 = find_actualClass(b_classify['has_feature'])
    b2 = find_actualClass(b_classify['has_no_feature'])
    a1_data = {'data': a_classify['has_feature'], 'repr': a1}
    a2_data = {'data': a_classify['has_no_feature'], 'repr': a2}
    b1_data = {'data': b_classify['has_feature'], 'repr': b1}
    b2_data = {'data': b_classify['has_no_feature'], 'repr': b2}
    process_list = [a1_data, a2_data, b1_data, b2_data]
    for data in process_list:
        if data['repr'] == 'cancer':
            final_c_list += data['data']
        else: 
            final_nc_list += data['data']
    return {'final_c_list': final_c_list, 'final_nc_list': final_nc_list}

def find_actualClass(sample_list):
    c = 0
    nc = 0
    actual_class = ''
    for sample in sample_list:
        if sample[0] == 'C':
            c += 1
        else:
            nc += 1
    if c > nc:
        actual_class = 'cancer'
    else:
        actual_class = 'n-cancer'
    return actual_class