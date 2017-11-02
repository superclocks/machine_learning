import numpy as np
from copy import copy

def frequenctItemset(d, min_sup = 2, dic = 0):
    #if don't supply the dictionary, then loop over the whole data.
    if dic == 0:
        dic = []
        for sample in d:
            for item in sample:
                dic.append(set([item]))
        
    fre_item_set = []
    C1 = dic
    while(1):
        Ck = aprioriGen(C1)
        C_sup = countSupport(Ck, d)
        fre_item_set.append(C_sup)
        C1 = C_sup[0]
        tmp = dict()
        
    a=10
def countSupport(Ck, data, thresh = 1):
    item_sets = []
    sup_counts = []
    for ck in Ck:
        sup = 0
        for one in data:
            if(ck & one == ck):
                sup = sup + 1
        if(sup > thresh):
            item_sets.append(ck)
            sup_counts.append(sup)
    return [item_sets, sup_counts]

def aprioriGen(frequent_item_sets):
    item_set = []
    number = len(frequent_item_sets)
    for i in range(0, number):
        s1 = frequent_item_sets[i]
        len1 = len(s1)
        for j in range(i+1, number):
            s2 = frequent_item_sets[j]
            merge_sets = mergeSet(s1, s2)
            if(len1 + len1 -1 == len(merge_sets) - 1):
                if(hasFrequentSubSet(merge_sets, frequent_item_sets)):
                    item_set.append(merge_sets)
        
    return item_set
def isSubSet(sub_set, paraent_set):
    for ee in paraent_set:
            if (set(sub_set) & ee) == set(sub_set):
                return True
    return False
    
def hasFrequentSubSet(merge_set, paraent_set):
    set_list = list(merge_set)
    marker = True 
    for e in set_list:
        sub_set = copy(set_list)
        sub_set.remove(e)
        r = isSubSet(sub_set, paraent_set)
        marker = marker & r
    return marker
                
def mergeSet(s1, s2):
    return s1 | s2
    
def readData(path):
    data = []
    for line in open(path, 'r'):
        ele = line.split(' ')
        s = set()
        ele = ele[0: len(ele) - 1]
        for e in ele:
            s.add(e)
        data.append(s)
    return data

if __name__ == '__main__':
    data = readData('retail.dat')
    data = data[0: 10]
    frequenctItemset(data)