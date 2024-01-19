def divide_to_subsamples(data_list: list, samples: list, best_feature: str, df: any):
    has_feature = []
    has_no_feature = []
    index_of_data_list = [samples.index(data) for data in data_list]
    for i in range(len(data_list)):
        if df.loc[index_of_data_list[i]][best_feature] == 1:
            has_feature.append(data_list[i])
        else:
            has_no_feature.append(data_list[i])
    return {'has_feature': has_feature, 'has_no_feature': has_no_feature}