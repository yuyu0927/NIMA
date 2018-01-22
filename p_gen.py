#/bin/python3
import os
os.environ.setdefault('PATH', '')
import numpy as np
# s = np.random.normal(5,0.2,10000)
# s = np.rint(s)
# a=np.histogram(s,bins=np.arange(1,12),density=True)
# print(a)

def gen(mu,sigma):
    s = np.random.normal(mu,sigma,10000)
    s = np.rint(s)
    s_u = ''
    for i in np.histogram(s,bins=np.arange(1,12),density=True)[0]:
        s_u += str(i) + ' '
    return s_u
f_out = open('class.txt','w')
f_mean = open('mos_with_names.txt')
f_std = open('mos_std.txt')
means = f_mean.readlines()
stds = f_std.readlines()
for i in range(len(means)):
    mean = float(means[i].split(' ')[0])
    std = float(stds[i])
    name = means[i].split(' ')[1].lower()
    f_out.write(gen(mean,std)+name)
