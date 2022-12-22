import numpy as np
import pandas as pd
import cooler
import subprocess
import os
import shutil

def extract_constraint_mats(resolution):
    '''
    if not os.path.exists("DataFull/Constraints"):
        subprocess.run("mkdir -p DataFull/Constraints", shell = True) # similar function as 'os.makedirs("DataFull/Constraints")'
    '''

    if os.path.exists("DataFull/Constraints"):
        shutil.rmtree("DataFull/Constraints")  # os.remove(filename); os.rmdir(directory) an empty directory; shutil.rmtree() remove a directory and its all contents
    os.makedirs("DataFull/Constraints")  # will recursively creat the directories, if the any of them are missing

    filepath = '../Datasets/Drosophila/GSM3820057_Cell1.10000.mcool'
    AllRes = cooler.fileops.list_coolers(filepath)
    # print(AllRes)

    res = resolution
    c = cooler.Cooler(filepath + '::resolutions/' + str(res))
    c1 = c.chroms()[:]  # c1 chromesize information in the list format
    print(c1.loc[0, 'name'], c1.index)
    # print('\n')

    for i in c1.index:
        print(i, c1.loc[i, 'name'])
        chro = c1.loc[i, 'name']  # chro is str
        # print(type(chro))
        c2 = c.matrix(balance = True, as_pixels = True, join = True).fetch(chro)
        # print(c2)
        c2 = c2[['start1', 'start2', 'balanced']]
        # print(c2)
        c2.fillna(0, inplace = True)
        if i == 6:
            pass
        else:
            c2.to_csv('DataFull/Constraints/chrom_' + str(i+1) + '_' + str(res) + '.txt', sep = '\t', index = False, header = False)

def tri_recursion(k):   #  function recursion, which means a defined function can call itself.
    if(k > 0):
        result = k + tri_recursion(k - 1)
        print(k, result)
    else:
        result = 10
    return result

def funct1(a, Lis = None):
    if Lis == None:
        Lis = []
    Lis.append(a)
    return Lis

def funct(a, Lis = []):
    Lis.append(a)
    return Lis

def parrot(voltage, state='a stiff', action='voom', type='Norwegian Blue'):
    print("\n-- This parrot wouldn't", action, end=' ')
    print("if you put", voltage, "volts through it.")
    print("-- Lovely plumage, the", type)
    print("-- It's", state, "!")

if __name__ == '__main__':
    # extract_constraint_mats(40000)

    for nn in list(range(1, 7)):
        if nn == 2:
            pass
        else:
            print(nn)
    # print([i for i in range(1, 7)])

    # below is function Recursion
    print("\n\nRecursion Example Results")
    tri_recursion(6)

    # comparation between below two instances Lis == None and Lis == []:
    for nn in range(1, 11):
        print(funct1(nn))

    for nn in range(1, 11):
        print(funct(nn))

    # Bellow information tells you how to pass the arguments to def function parameters
    # the positional arguments (i.e., non-Kwargs) must precede the Kwargs.

    parrot(1000)  # 1 positional argument
    parrot(voltage = 1000)  # 1 keyword argument
    parrot(voltage = 1000000, action = 'VOOOOOM')  # 2 keyword arguments
    parrot(action = 'VOOOOOM', voltage = 1000000)  # 2 keyword arguments
    parrot('a million', 'bereft of life', 'jump')  # 3 positional arguments in the default order
    parrot('a thousand', state = 'pushing up the daisies')  # 1 positional, 1 keyword