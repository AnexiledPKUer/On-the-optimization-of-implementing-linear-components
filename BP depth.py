# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 11:43:12 2019

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 10:38:03 2019

@author: Administrator
"""

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
        s+=a**2
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
f=open(r'C:\Users\Administrator\Desktop\resultx.txt',mode='r+')
f.seek(0,2)
for sad in range(20):
    
    M=np.array([[1,0,0,0,0,1,0,0,1,0,0,1,0,0,0,1],
[0,1,0,0,0,0,1,0,1,1,0,0,1,0,0,0],
[0,0,1,0,1,0,0,1,0,1,1,0,0,1,0,0],
[0,0,0,1,1,0,0,0,0,0,1,0,0,0,1,1],
[0,1,0,0,1,0,0,0,0,0,0,1,1,0,0,1],
[0,0,1,0,0,1,0,0,1,0,0,0,1,1,0,0],
[1,0,0,1,0,0,1,0,0,1,0,0,0,1,1,0],
[1,0,0,0,0,0,0,1,0,0,1,1,0,0,1,0],
[1,0,0,1,0,0,0,1,1,0,0,0,0,1,0,0],
[1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0],
[0,1,1,0,0,1,0,0,0,0,1,0,1,0,0,1],
[0,0,1,0,0,0,1,1,0,0,0,1,1,0,0,0],
[0,0,0,1,1,0,0,1,0,1,0,0,1,0,0,0],
[1,0,0,0,1,1,0,0,0,0,1,0,0,1,0,0],
[0,1,0,0,0,1,1,0,1,0,0,1,0,0,1,0],
[0,0,1,1,0,0,1,0,1,0,0,0,0,0,0,1]] )
    (n,m)=M.shape
    depth=[0 for i in range(m)]
    S=[[set([i]),0] for i in range(m)]
    for i in range(m):
        S[i][1]=depth[i]
    Dist=[0 for i in range(n)]
    for i in range(n):
        Dist[i]=weight(M[i])-1
    goal_depth=[3 for i in range(n)]
    tagets=[]
    for j in range(n):
        s=[]
        for i in range(m):
            if M[j][i]==1:
                s+=[i]
        tagets+=[set(s)]
    f.write("第次测试: \n 原始矩阵 \n "+ str(M) +'\n 目标深度限制：\n'+str(goal_depth)+"\n 目标:\n"+str(tagets)+'\n')
    f.write('Dist'+str(Dist)+'\n')
    since=time.time()   
    while max(Dist)>1:
        if 1 in Dist:
            t=Dist.index(1)
            newbasis=tagets[t]
            value=list(combinations(S,2))
            x=[{},100];y=[{},100]
            for (x1,y1) in value:
                if xors(x1[0],y1[0])==newbasis and max(x1[1],y1[1])+1< max(x[1],y[1])+1:
                    x=x1;y=y1
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
                                    dept=[max(x[1],y[1])+1]
                                    for asd in list(_):
                                        dept+=[asd[1]]
                                    if feasible(dept,goal_depth[i]):
                                        pop+=[i]
                                        Dist[i]=j-1
                            for asd in pop:
                                tempj.remove(asd)
                        except StopIteration:
                            break
            f.write('newbasis'+str([newbasis,max(x[1],y[1])+1])+'\n')
            f.write('Dist'+str(Dist)+'\n')
            S.append([newbasis,max(x[1],y[1])+1])
            continue
        
        
        
        since1=time.time()
        maxbenefit=[[],0]
        value=list(combinations(S,2))
        for x in value:
            basis1=[xors(x[0][0],x[1][0]),max(x[0][1],x[1][1])+1]
            benefit=0
            tem=[i for i in range(n)]
            distances=list(set(Dist))
            distances.sort()
            for j in distances:
                if j!=0:
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
                                if re==xors(tagets[i],basis1[0]):
                                    dept=[max(x[0][1],x[1][1])+1]
                                    for asd in list(_):
                                        dept+=[asd[1]]
                                    if feasible(dept,goal_depth[i]):
                                        pop+=[i]
                                        tem.remove(i)
                                        benefit+=1
                            for asd in pop:
                                tempj.remove(asd)
                        except StopIteration:
                            break
            if benefit==maxbenefit[1]:
                maxbenefit[0]+=[x]
            if benefit>maxbenefit[1]:
                maxbenefit[1]=benefit
                maxbenefit[0]=[x]
            
        
        max_benefit=maxbenefit[0]
        ##benefit大的里面取深度小的
        if len(max_benefit)!=1:
            x1=max_benefit[0]
            temp=[]
            for x in max_benefit:
                if max(x[0][1],x[1][1])== max(x1[0][1],x1[1][1]):
                    temp+=[x]
                if max(x[0][1],x[1][1])<max(x1[0][1],x1[1][1]):
                    temp=[x]
                    x1=x  
            max_benefit=temp
        #benefit大的里面取深度小的
        f.write('max_benefit'+str(max_benefit)+'\n')
        x=max_benefit[randint(0,len(max_benefit))]
        basis1=[xors(x[0][0],x[1][0]),max(x[0][1],x[1][1])+1]
        f.write('newbasis'+str(basis1)+'\n'+'benefit:'+str(maxbenefit[1])+'\n')
        tem=[i for i in range(n)]
        distances=list(set(Dist))
        distances.sort()
        for j in distances:
            if j!=0:
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
                            if re==xors(tagets[i],basis1[0]):
                                dept=[max(x[0][1],x[1][1])+1]
                                for asd in list(_):
                                    dept+=[asd[1]]
                                if feasible(dept,goal_depth[i]):
                                    pop+=[i]
                                    tem.remove(i)
                                    Dist[i]-=1
                        for asd in pop:
                            tempj.remove(asd)
                    except StopIteration:
                        break
        S.append(basis1)
        f.write('Dist'+str(Dist)+'\n')
        timed=time.time()-since1
        print('本次选择耗时 {:.0f}m {:.0f}s'.format(timed//60 ,timed % 60))
        f.write('本次选择耗时 {:.0f}m {:.0f}s'.format(timed//60 ,timed % 60) +'\n')
    num=len(S)
    for x in Dist:
        num+=x
    print(num)
    f.write('当前距离Dist'+str(Dist)+'\n')
    f.write('所有的节点'+str(S)+'\n')
    f.write('整个优化需要的总门数：'+str(num)+'\n')
    time_=time.time()-since
    print('整个算法的运行时间 {:.0f}m {:.0f}s'.format(time_ //60 ,time_ % 60) )
    f.write('本次算法的运行时间 {:.0f}m {:.0f}s'.format(time_ //60 ,time_ % 60)+'\n' )
f.close()
    
    