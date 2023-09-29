import numpy as np
import csv
import random
import linecache

random.seed()
seed = random.randrange(1000000)
random.seed(seed)

linesToSample = random.sample(range(11000000), 10000)
top = max(linesToSample)
x=0

writer = open('higgs2.csv', 'w')

with open("HIGGS.csv") as f:
    while True:
        if x%100000==0:
            print(x, flush=True)
        data = f.readline()
        if x in linesToSample:
            writer.write(data)
        if not data or x > top:
            break
        x+=1

writer.close()
