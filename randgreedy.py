# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 15:53:37 2019

@author: Administrator
"""

from numpy.random import randint
import numpy as np
from numpy import array
from math import log,ceil
import time
from itertools import combinations
np.set_printoptions(precision=4,suppress=True)
import copy


def min_depth(m):
    if len(m)==1:
        return m[0]
    m.sort()
    while len(m)>2:
        d=max(m[0],m[1])+1
        m.pop(0);m.pop(0);
        i=0
        while i < (len(m)) and  m[i]<d :
            i+=1
        m.insert(i,d)
    return max(m[0],m[1])+1
def feasible(m,goal_depth):
    return min_depth(m)<=goal_depth
def xors(x,y):
    return (x|y)-(y&x)
def sum_(sw):
    r=set()
    for v in sw:
        r=xors(r,v[0])
    return r
def norm(x):
    s=0
    for a in x:
        s+=a^2
    return s
##生成n乘m的随机0,1矩阵
def gen_rand_mat(n,m):
    s=randint(0,2,m*n)
    s.resize(n,m)
    return s
def weight(x):
    s=0
    for y in x:
        s+=y
    return(s)
f=open(r'C:\Users\Administrator\Desktop\GHQ.txt',mode='r+')
f.seek(0,2)
#M=gen_rand_mat(5,5)
for asddd in range(25):
    M=np.array(
  
[[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
 [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
 [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
 [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
 [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
 [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
 [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
 [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
 [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    (n,m)=M.shape
    tagets=[]
    for j in range(n):
        s=[]
        for i in range(m):
            if M[j][i]==1:
                s+=[i]
        tagets+=[s]
    S=[[set([i]),0] for i in range(m)]
    goal_depth=[3 for i in range(n)]
    depth=[0 for i in range(m)]
    for i in range(m):
        S[i][1]=depth[i]
    weights=[weight(M[i]) for i in range(n)]
    print("原始矩阵:",M,"初始重量：",weights, sep='\n')
    f.write("第次测试: \n 原始矩阵 \n "+ str(M) +'\n 目标深度限制：\n'+str(goal_depth)+"\n 目标:\n"+str(tagets)+'\n')
    since=time.time() 
    while max(weights)>2:
        since1=time.time()
        m1=M.shape[1]
        print("阶段的矩阵列数：",m1) 
        #to_append=array([0 for i in range(n)]).reshape(n,1)
        conl=[i for i in range(m1)]
        value=list(combinations(conl,2))
        maxbenefit=[[],0]
        for x in value:
            tem=M[:,x[0]]*M[:,x[1]]
            benefit=0
            for i in range(n):
                if tem[i]==1:
                    dept=[max(S[x[0]][1],S[x[1]][1])+1]
                    conl1=[i for i in range(m1)]
                    conl1.remove(x[0])
                    conl1.remove(x[1])
                    for asd in conl1:
                        if M[i][asd]==1:
                            dept+=[S[asd][1]]
                    if feasible(dept,goal_depth[i]):
                        benefit+=1
            if benefit==maxbenefit[1]:
                maxbenefit[0]+=[x]
            if benefit>maxbenefit[1]:
                maxbenefit=[[x],benefit]
        max_benefit=maxbenefit[0]
        f.write('max_benefit'+str(max_benefit)+'\n'+'benefit'+str(maxbenefit[1])+'\n')
        print(('max_benefit'+str(max_benefit)+'\n'+'benefit'+str(maxbenefit[1])+'\n'))
        x=max_benefit[randint(0,len(max_benefit))]
        basis=[xors(S[x[0]][0],S[x[1]][0]),max(S[x[0]][1],S[x[1]][1])+1]
        S=S+[basis]
        f.write('newbasis'+str(basis)+'\n')
        print(('newbasis'+str(basis)+'\n'))
        #choose and update weights and addarray
        tem=M[:,x[0]]*M[:,x[1]]
        newcon=[]
        for i in range(n):
            if tem[i]==1:
                dept=[max(S[x[0]][1],S[x[1]][1])+1]
                conl1=[i for i in range(m1)]
                conl1.remove(x[0])
                conl1.remove(x[1])
                for asd in conl1:
                    if M[i][asd]==1:
                        dept+=[S[asd][1]]
                if feasible(dept,goal_depth[i]):
                    newcon=newcon+[i]
                    M[i][x[0]]=0;M[i][x[1]]=0
        add_array=(np.zeros((n,1)))
        for asdd in newcon:
            add_array[asdd][0]=1
        M=np.hstack((M,add_array))
        print('更新后的M',M)
        weights=[weight(M[i]) for i in range(n)]
        print('更新后的重量',weights)
        f.write('更新后的重量'+str(weights))
        timed=time.time()-since1
        print('本次选择耗时 {:.0f}m {:.0f}s'.format(timed//60 ,timed % 60))
        f.write('本次选择耗时 {:.0f}m {:.0f}s'.format(timed//60 ,timed % 60) +'\n')
    num=M.shape[1]-m
    for y in weights:
        if y==2:
            num+=1
    print('门',num)
    f.write('当前距离weights'+str(weights)+'\n')
    f.write('所有的节点'+str(S)+'\n')
    f.write('整个优化需要的总门数：'+str(num)+'\n')
    time_=time.time()-since
    print('整个算法的运行时间 {:.0f}m {:.0f}s'.format(time_ //60 ,time_ % 60) )
    f.write('本次算法的运行时间 {:.0f}m {:.0f}s'.format(time_ //60 ,time_ % 60) +'\n')
f.close()