import math
from tabulate import tabulate

def best_features(training_list: list, samples: list, df: any, string: str):
    gain_list = []
    handle_0 = 0.000000001
    index_of_trn_list = [samples.index(data) for data in training_list]
    n_t = len(training_list)
    n_C =0
    n_NC = 0
    for i in range(n_t):
        if training_list[i][0] == 'C':
            n_C += 1
        else:
            n_NC += 1
    p_Ct = n_C / (n_t + handle_0)         
    p_NCt = n_NC / (n_t + handle_0) 
    h_T = -(p_Ct * math.log(p_Ct + handle_0, 2) + p_NCt * math.log(p_NCt + handle_0, 2))

    # find gain for each feature
    for feature_name, row  in df.iteritems():
        n_l =0
        n_r = 0
        n_Cl = 0
        n_NCl = 0
        n_Cr = 0
        n_NCr = 0
        # loop through each sample
        for i in range(n_t):
            n = row[index_of_trn_list[i]]
            if n == 1:
                n_l += 1
                if training_list[i][0] == 'C':
                    n_Cl += 1
                else: 
                    n_NCl += 1  
            else:
                n_r += 1
                if training_list[i][0] == 'C':
                    n_Cr += 1
                else: 
                    n_NCr += 1
        p_l = n_l / (n_t + handle_0)
        p_r = n_r / (n_t + handle_0)
        p_Cl = n_Cl / (n_l + handle_0)
        p_NCl = n_NCl / (n_l + handle_0)
        p_Cr = n_Cr / (n_r + handle_0)
        p_NCr = n_NCr / (n_r + handle_0)
        h_Tl = -(p_Cl * math.log(p_Cl + handle_0, 2) + p_NCl * math.log(p_NCl + handle_0, 2))
        h_Tr = -(p_Cr * math.log(p_Cr + handle_0, 2) + p_NCr * math.log(p_NCr + handle_0, 2))
        h_ST = p_l * h_Tl + p_r * h_Tr
        gain  = h_T - h_ST
        gain_list.append({'n_l': n_l, 'n_r': n_r, 'n_Cl': n_Cl, 'n_NCl': n_NCl, 'n_Cr': n_Cr, 'n_NCr': n_NCr,\
         'p_l': p_l,'p_r': p_r, 'h_ST': h_ST, 'h_T': h_T, 'gain': gain, 'feature': feature_name})
    index_of_top10 = sorted(range(len(gain_list)), key=lambda i: gain_list[i]['gain'], reverse=True)[:10]
    arr = []
    for i in index_of_top10:
        arr.append([gain_list[i]['feature'][:10], gain_list[i]['n_l'], gain_list[i]['n_r'], gain_list[i]['n_Cl'], gain_list[i]['n_NCl'],\
        gain_list[i]['n_Cr'], gain_list[i]['n_NCr'], round(gain_list[i]['p_l'], 5), round(gain_list[i]['p_r'], 5), \
        round(gain_list[i]['h_ST'], 5), round(gain_list[i]['h_T'], 5), gain_list[i]['gain']])
    arr.insert(0, ['Genetic Mutation', 'n(tL)', 'n(tR)', 'n(tL, C)', 'n(tL, NC)', 'n(tR, C)', 'n(tR, NC)', 'PL', \
        'PR', 'H(s, t)', 'H(t)', 'gain(s)'])
    if string == 'f':
        print(tabulate(arr))
    return {'feature': gain_list[index_of_top10[0]]['feature'], 'gain': gain_list[index_of_top10[0]]['gain']}