# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 21:03:01 2019

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 08:44:41 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 10:11:45 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 20:52:32 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 19:14:37 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 21:28:01 2018

@author: liujian
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 10:36:12 2018
增加了second选项的随机
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
f=open(r'C:\Users\Administrator\Desktop\temp1.txt',mode='r+')
f.seek(0,2)
M=np.array([[1,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0],
[0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,1],
[0,0,1,0,0,0,1,0,0,1,0,0,1,1,0,0],
[0,0,0,1,0,0,0,1,0,0,1,1,0,1,0,0],
[1,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0],
[0,1,0,0,1,0,0,0,1,0,0,1,0,1,0,0],
[0,0,1,0,0,1,0,0,1,1,0,0,0,0,1,0],
[0,0,0,1,0,0,1,1,0,1,0,0,0,0,0,1],
[0,0,0,1,0,0,1,0,1,0,0,0,1,0,0,0],
[1,0,0,0,1,0,0,1,0,1,0,0,0,1,0,0],
[0,1,0,0,1,1,0,0,0,0,1,0,0,0,1,0],
[0,0,1,1,0,1,0,0,0,0,0,1,0,0,0,1],
[0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,1],
[1,0,0,1,0,1,0,0,0,1,0,0,1,0,0,0],
[1,1,0,0,0,0,1,0,0,0,1,0,0,1,0,0],
[0,1,0,0,0,0,0,1,0,0,0,1,0,0,1,1]]
            )
(n,m)=M.shape
S=[[set([i]),0] for i in range(m)]
Dist=[0 for i in range(n)]
for i in range(n):
    Dist[i]=weight(M[i])-1
goal_depth=[]
tagets=[]
for j in range(n):
    s=[]
    for i in range(m):
        if M[j][i]==1:
            s+=[i]
    tagets+=[set(s)]
biaoshun=[195,91,57,41,33,29,26,25,24,23,23]
since=time.time()    
while max(Dist)>1:
    if 1 in Dist:
        t=Dist.index(1)
        newbasis=tagets[t]
        print('newbasis',newbasis)
        f.write('newbasis'+str(newbasis))
        value=combinations(S,2)
        (x,y)=next(value)
        while xors(x[0],y[0])!=newbasis:
            (x,y)=next(value)
        Dist[t]=0
        distances=list(set(Dist))
        distances.sort()
        for j in distances:
            j=int(j)
            if j !=0 and j!=1:
                value=combinations(S,j-1)
                tempj=[]
                for asd in range(n):
                    if Dist[asd]==j:
                        tempj+=[asd]
                    
                while tempj!=[]:
                    try:
                        _ = next(value)
                        re=sum_(list(_))
                        pop=[]
                        for i in tempj:
                            if re==xors(tagets[i],newbasis):
#                                if feasible(dept,goal_depth[i]):
                                pop+=[i]
                                Dist[i]=j-1
                        for asd in pop:
                            tempj.remove(asd)
                    except StopIteration:
                        break
        
        S.append([newbasis,max(x[1],y[1])+1])
        print('distance',Dist)
        f.write('distance'+str(Dist))
        continue
    
    if biaozhun[max(Dist)-1]>=len(S):
    
    since1=time.time()
    maxbenefit=[[],0]
    valuet=list(combinations(S,2))
    value=list(combinations(valuet,2))
    for (x,y) in value:
        basis1=[xors(x[0][0],x[1][0]),max(x[0][1],x[1][1])+1]
        basis2=[xors(y[0][0],y[1][0]),max(y[0][1],y[1][1])+1]
        benefit=0
        tem=[i for i in range(n)]
        distances=list(set(Dist))
        distances.sort()
        for j in distances:##如果能减2，basis1和2都要参与
            if j ==3:#不可能减2
                tempj=[]
                for i in tem:
                    if Dist[i]==3:
                        tempj+=[i]
                pop=[]
                for i in tempj:
                    if tagets[i]==xors(basis2[0],basis1[0]):
                        benefit+=2
                        pop+=[i]
                for asd in pop:
                    tem.remove(asd)
            if j > 3:
                value2=combinations(S,j-3)
                tempj=[]
                for asd in tem:
                    if Dist[asd]==j:
                        tempj+=[asd]
                
                while tempj!=[]:
                    try:
                        _ = next(value2)
                        re=sum_(list(_))
                        pop=[]
                        for i in tempj:
                            if xors(re,tagets[i])==xors(basis1[0],basis2[0]):
#                                if feasible(dept,goal_depth[i]):
                                pop+=[i]
                                tem.remove(i)
                                benefit+=2
                        for asd in pop:
                            tempj.remove(asd)
                    except StopIteration:
                        break
        for j in distances:#减1
            if j ==2:
                tempj=[]
                for i in tem:
                    if Dist[i]==2:
                        tempj+=[i]
                pop=[]
                for i in tempj:
                    if tagets[i]==xors(basis2[0] , basis1[0]):
                        benefit+=1
                        pop+=[i]
                for asd in pop:
                    tem.remove(asd)
                    tempj.remove(asd)
                pop=[]
                for i in tempj:
                    for asd in S:
                        if tagets[i]==xors(asd[0] , basis1[0]) :
                           benefit+=1
                           pop+=[i]
                           break
                for asd in pop:
                    tem.remove(asd)
                    tempj.remove(asd)
                pop=[]
                for i in tempj:
                    for asd in S:
                        if tagets[i]==xors(asd[0] , basis2[0]) :
                           benefit+=1
                           pop+=[i]
                           break
                for asd in pop:
                    tem.remove(asd)
                    tempj.remove(asd)
            if j > 2:
                value2=combinations(S,j-1)
                tempj=[]
                for asd in tem:
                    if Dist[asd]==j:
                        tempj+=[asd]
                
                while tempj!=[]:
                    try:
                        _ = next(value2)
                        re=sum_(list(_))
                        pop=[]
                        for i in tempj:
                            if re==xors(tagets[i],basis1[0]) or re==xors(tagets[i],basis2[0]):
#                                if feasible(dept,goal_depth[i]):
                                pop+=[i]
                                tem.remove(i)
                                benefit+=1
                        for asd in pop:
                            tempj.remove(asd)
                    except StopIteration:
                        break
#                value2=combinations(S,j-2)
#                tempj=[]
#                for asd in tem:
#                    if Dist[asd]==j:
#                        tempj+=[asd]
#                
#                while tempj!=[]:
#                    try:
#                        _ = next(value2)
#                        re=sum_(list(_))
#                        pop=[]
#                        for i in tempj:
#                            if xors(tagets[i],re)==xors(basis2[0],basis1[0]) :
##                                if feasible(dept,goal_depth[i]):
#                                pop+[i]
#                                tem.remove(i)
#                                benefit+=1
#                        for asd in pop:
#                            tempj.remove(asd)
#                    except StopIteration:
#                        break
        if benefit==maxbenefit[1]:
            maxbenefit[0]+=[(x,y)]
        if benefit>maxbenefit[1]:
            maxbenefit[1]=benefit
            maxbenefit[0]=[(x,y)]
        
    
    for x in valuet:
        basis1=[xors(x[0][0],x[1][0]),max(x[0][1],x[1][1])+1]
        for t in S:
            y=(basis1,t)
            basis2=[xors(y[0][0],y[1][0]),max(y[0][1],y[1][1])+1]
            benefit=0
            tem=[i for i in range(n)]
            distances=list(set(Dist))
            distances.sort()
            for j in distances:##如果能减2，只需要basis2参与
                if j ==2:#
                    tempj=[]
                    for i in tem:
                        if Dist[i]==2:
                            tempj+=[i]
                    pop=[]
                    for i in tempj:
                        if tagets[i]==basis2[0]:
                            benefit+=2
                            pop+=[i]
                    for asd in pop:
                        tem.remove(asd)
                if j > 3:
                    value2=combinations(S,j-2)
                    tempj=[]
                    for asd in tem:
                        if Dist[asd]==j:
                            tempj+=[asd]
                    
                    while tempj!=[]:
                        try:
                            _ = next(value2)
                            re=sum_(list(_))
                            pop=[]
                            for i in tempj:
                                if xors(re,tagets[i])==basis2[0]:
    #                                if feasible(dept,goal_depth[i]):
                                    pop+=[i]
                                    tem.remove(i)
                                    benefit+=2
                            for asd in pop:
                                tempj.remove(asd)
                        except StopIteration:
                            break
            for j in distances:#只减1,basis1和basis2只能有一个参与

                if j >=2:
                    value2=combinations(S,j-1)
                    tempj=[]
                    for asd in tem:
                        if Dist[asd]==j:
                            tempj+=[asd]
                    
                    while tempj!=[]:
                        try:
                            _ = next(value2)
                            re=sum_(list(_))
                            pop=[]
                            for i in tempj:
                                if re==xors(tagets[i],basis1[0]) or re==xors(tagets[i],basis2[0]):
    #                                if feasible(dept,goal_depth[i]):
                                    pop+=[i]
                                    tem.remove(i)
                                    benefit+=1
                            for asd in pop:
                                tempj.remove(asd)
                        except StopIteration:
                            break
            if benefit==maxbenefit[1]:
                maxbenefit[0]+=[(x,y)]
            if benefit>maxbenefit[1]:
                maxbenefit[1]=benefit
                maxbenefit[0]=[(x,y)]
            
        
        
        
        
        
    print('maxbenefit',maxbenefit[0],maxbenefit[1])
    
    f.write('maxbenefit'+str(maxbenefit[0])+str(maxbenefit[1]))
    (x,y)=maxbenefit[0][randint(0,len(maxbenefit[0]))]
    basis1=[xors(x[0][0],x[1][0]),max(x[0][1],x[1][1])+1]
    basis2=[xors(y[0][0],y[1][0]),max(y[0][1],y[1][1])+1]
    print('basis1 and basis2',basis1,basis2)
    f.write('basis1 and basis2'+str(basis1)+str(basis2))
    tem=[i for i in range(n)]
    distances=list(set(Dist))
    distances.sort()
    for j in distances:
        if j ==2:
            tempj=[]
            for i in tem:
                if Dist[i]==2:
                    tempj+=[i]
            pop=[]
            for i in tempj:
                if tagets[i]==basis2[0]:
                    Dist[i]-=2
                    pop+=[i]
            for asd in pop:
                tem.remove(asd)
        if j > 2:
            value2=combinations(S+[basis1],j-2)
            tempj=[]
            for asd in tem:
                if Dist[asd]==j:
                    tempj+=[asd]
            
            while tempj!=[]:
                try:
                    _ = next(value2)
                    re=sum_(list(_))
                    pop=[]
                    for i in tempj:
                        if re==xors(tagets[i],basis2[0]):
#                                if feasible(dept,goal_depth[i]):
                            pop+=[i]
                            tem.remove(i)
                            Dist[i]-=2
                    for asd in pop:
                        tempj.remove(asd)
                except StopIteration:
                    break
    for j in distances:
#        if j ==2:
#            pop=[]
#            for i in tem:
#                if tagets[i]==basis2[0] or tagets[i]==basis1[0]:
#                    Dist[i]-=1
#                    pop+=[i]
#            for asd in pop:
#                tem.remove(asd)
        if j >= 2:
            value2=combinations(S+[basis1,basis2],j-1)
            tempj=[]
            for asd in tem:
                if Dist[asd]==j:
                    tempj+=[asd]
            
            while tempj!=[]:
                try:
                    _ = next(value2)
                    re=sum_(list(_))
                    pop=[]
                    for i in tempj:
                        if xors(re,basis1[0])==tagets[i] or xors(re,basis2[0])==tagets[i]:
#                                if feasible(dept,goal_depth[i]):
                            pop+=[i]
                            tem.remove(i)
                            Dist[i]-=1
                    for asd in pop:
                        tempj.remove(asd)
                except StopIteration:
                    break
    S.append(basis1)
    S.append(basis2)
    print('distance',Dist)
    timed=time.time()-since1
    print('本次选择耗时 {:.0f}m {:.0f}s'.format(timed//60 ,timed % 60))
    f.write('distance'+str(Dist))
    f.write('本次选择耗时 {:.0f}m {:.0f}s'.format(timed//60 ,timed % 60) )
    
    
num=len(S)
for x in Dist:
    num+=x
print(num)
f.write('当前距离Dist'+str(Dist))
f.write('所有的节点'+str(S))
f.write('整个优化需要的总门数：'+str(num))
time_=time.time()-since
f.write('本次算法的运行时间 {:.0f}m {:.0f}s'.format(time_ //60 ,time_ % 60) )
print('整个算法的运行时间 {:.0f}m {:.0f}s'.format(time_ //60 ,time_ % 60) )
f.close() 
    