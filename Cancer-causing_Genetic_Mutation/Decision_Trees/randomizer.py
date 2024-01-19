import pandas as pd 
import random as rand

def randomize(all_list: list, k: int):
    all_samples = all_list
    test_list = []                            
    training_list = []
    for i in range(k):
        test = rand.sample(all_list, 54)
        test_list.append(test)
        training = [s for s in all_samples if s not in test]
        training_list.append(training)
        all_list = [s for s in all_list if s not in test]
    return {'test_list': test_list, 'training_list': training_list}