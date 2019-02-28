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
def choose(weight_array):
    m=weight_array.shape[0]
    a=weight_array[0][1]
    ma=[a,[]]
    ms=[0,[]]
    for x in range(m):
        for y in range(x+1,m):
            if weight_array[x,y]>ma[0]:
                ms=copy.deepcopy(ma)
                ma[0]=weight_array[x,y];ma[1]=[[x,y]]
                continue
            if weight_array[x,y]==ma[0]:
                ma[1]+=[[x,y]]
                continue
            if weight_array[x,y]>ms[0]:
                ms[0]=weight_array[x,y];ms[1]=[[x,y]]
                continue
            if weight_array[x,y]==ms[0]:
                ms[1]+=[[x,y]]
    if ms[0]>1 and ms[0]==ma[0]-1:
        return ma[1]+ms[1]
    else:
        return ma[1]

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
        r=xors(r,v)
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

f=open(r'C:\Users\Administrator\Desktop\temp2.txt',mode='r+')
f.seek(0,2)
for sad in range(15):
    since=time.time()
    choice=[]
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
[0,0,1,1,0,0,1,0,1,0,0,0,0,0,0,1]]
            
            
            
            
            
            
            
            
            )
    (n,m)=M.shape
    m1=m
    depth=[0 for i in range(m)]#depth记录列的深度，在初始时将全部列深度置0
    tagets=[]
    for j in range(n):
        s=[]
        for i in range(m):
            if M[j][i]==1:
                s+=[i]
        tagets+=[set(s)]
    goal_depth=[10 for i in range(n)]#记录n个output的深度要求
    weights=[weight(M[i]) for i in range(n)]
    #d=max(weights)
    conl=[{i} for i in range(m)]#表示每一列代表的元素是什么 
    base=conl #此时的base是和列一样的
    #limit=200#用于记录权重
    print("原始矩阵:",M,"目标深度限制：",goal_depth,"初始重量：",weights,"目标:",tagets, sep='\n')
    f.write("第次测试: \n 原始矩阵 \n "+ str(M) +'\n 目标深度限制：\n'+str(goal_depth)+"\n 初始重量：\n"+str(weights)+"\n 目标:\n"+str(tagets)+'\n')
    while max(weights)>2 :
        (n,m)=M.shape
        print('当前矩阵的shape',n,m)
        f.write('当前矩阵的shape'+ str(n)+'  '+str(m)+'\n')
        #choose and update weights and addarray
        weight_array=np.zeros((m,m))
        if 2 in weights:
            tem=[]
            lines=[]
            for asd in range(n):
                if weights[asd]==2:
                    lines+=[asd]
            line=lines[randint(0,len(lines))]
            for kl in range(len(M[line])):
                if M[line][kl]==1:
                    tem+=[kl]
            [x,y]=tem
            df=max(depth[x],depth[y])+1
            for kx in range(m):
                newbase=(conl[x]|conl[y])-(conl[x]&conl[y])
                for ky in range(kx+1,m):
                    if (conl[kx]|conl[ky])-(conl[kx]&conl[ky])==newbase and max(depth[kx],depth[ky])+1<df:
                        df=max(depth[kx],depth[ky])+1
                        [x,y]=[kx,ky]
            depth.insert(m,df)
            choice+=[[x,y]]
            
            print("选择的合并的列号",(x,y),newbase)
            f.write("选择的合并的列为唯2的"+'\n')
            f.write("选择的合并的列号"+str([x,y])+str(newbase)+'\n')
            add_array=(np.zeros((n,1)))
            #f.write('add_array:'+str(add_array.reshape(1,n))+'\n')
            #print('add_array:',add_array.reshape(1,n))
            M=np.hstack((M,add_array))
            #updatind the asdh
            
            
#            for _ in range(n):
#                if weights[_]==2 and M[_][m]==1:
#                    tem.remove(_)
#                    M[_][x]=0;M[_][y]=0
#                if M[_][m]==1:
#                    dept=[max(depth[x],depth[y])+1]
#                    for l in range(m):
#                        if M[_][l]==1:
#                            dept+=[depth[l]]
#                    dept.remove(depth[x]);dept.remove(depth[y])
#                    if feasible(dept,goal_depth[_]):
#                        tem.remove(_)
#                        M[_][x]=0;M[_][y]=0
#                    else:
#                        M[_][m]=0
            tem=[i for i in range(n)]      
            #weights=[weight(M[i]) for i in range(n)]   
            w=weights.copy()
            w.sort()
            for j in set(w):
                j=int(j)
                if j==1:
                    continue
                if j==2:
                    for _ in range(n):
                        if tagets[_]==newbase:
                            for hjk in range(m):
                                M[_][hjk]=0
                            M[_][m]=1
                            tem.remove(_)
                
                else:
                    value=combinations(conl,int(j-2))
                    tempj=[]
                    for asd in tem:
                        if weights[asd]==j:
                            tempj+=[asd]
                        
                    while tempj!=[]:
                        try:
                            _ = next(value)
                            re=sum_(list(_)+[newbase])
                            pop=[]
                            for i in tempj:
                                if re==tagets[i]:
                                    dept=[df]
                                    for l in range(m):
                                        if conl[l] in _:
                                            dept+=[depth[l]]
                                    if feasible(dept,goal_depth[i]):
                                        #tem.remove(i)
                                        pop+=[i]
                                        for l in range(m):
                                            if conl[l] in _:
                                                M[i][l]=1
                                            else:
                                                M[i][l]=0
                                        M[i][m]=1
                            for asd in pop:
                                tempj.remove(asd)
                        except StopIteration:
                            break
            conl.insert(m,newbase)
            base=conl
            weights=[weight(M[i]) for i in range(n)]
            print('合并后的重量:',weights)
            
            print('合并后的重量:',weights)
            print('合并后的M：',M)
            f.write('合并后的重量:'+str(weights)+'\n')
            f.write('合并后的M：'+str(M)+'\n')
            continue
        #                选择合并列
        since1=time.time()
        for x in range(m):
            for y in range(x+1,m):
                hx=x;hy=y
                benefit=0
                newbase=(conl[x]|conl[y])-(conl[x]&conl[y])
                df=max(depth[x],depth[y])+1
                for kx in range(m):
                    for ky in range(kx+1,m):
                        if (conl[kx]|conl[ky])-(conl[kx]&conl[ky])==newbase and max(depth[kx],depth[ky])+1<df:
                            hx=kx;hy=ky
#                            df=max(depth[kx],depth[ky])+1
#                            [x,y]=[kx,ky]
                if [hx,hy]!=[x,y]:
                    continue

                tem=[i for i in range(n)]
#                for i in range(n):
#                    if M[i][x]==1 and M[i][y]==1:
#                        dept=[max(depth[x],depth[y])+1]
#                        for l in range(m):
#                            if M[i][l]==1:
#                                dept+=[depth[l]]
#                        dept.remove(depth[x]);dept.remove(depth[y])
#                        if feasible(dept,goal_depth[i]):
#                            benefit+=1
#                            tem.remove(i)
                w=weights.copy()
                w.sort()
                for j in set(w):
                    j=int(j)
                    if j==1:
                        continue
                    if j==2:
                        for _ in range(n):
                            if tagets[_]==newbase:
                                for hjk in range(m):
                                    M[_][hjk]=0
                                M[_][m]=1
                                tem.remove(_)
                    else:
                        
                        value=combinations(conl,j-2)
                        tempj=[]
                        for asd in tem:
                            if weights[asd]==j:
                                tempj+=[asd]
                        while tempj!=[]:
                            try:
                                _ = next(value)
                                re=sum_(list(_)+[newbase])
                                pop=[]
                                for i in tempj:
                                    if re==tagets[i]:
                                        dept=[df]
                                        for l in range(m):
                                            if conl[l] in _:
                                                dept+=[depth[l]]
                                        if feasible(dept,goal_depth[i]):
                                            #tem.remove(i)
                                            pop+=[i]
                                            benefit+=1
                                for asd in pop:
                                    tempj.remove(asd)
                            except StopIteration:
                                break
                
                
                weight_array[x,y]=benefit
        #break
        #max_xy=np.where(weight_array==np.max(weight_array))#全部最大行列的位置
        #print("选择合并列的权重矩阵：",weight_array)
        ch=choose(weight_array)
        print(ch)
        r=randint(0,len(ch))
        [x,y]=ch[r]#随机选一个,此处未考虑范数      
        choice+=[[x,y]]
        newbase=(conl[x]|conl[y])-(conl[x]&conl[y])
        df=max(depth[x],depth[y])+1  
        print("选择的合并的列号",(x,y),newbase)
        print('weight',weight_array[x,y])
        f.write("选择的合并的列号"+str([x,y])+str(newbase)+'\n')
        print('（1,1）的数量:',weight((M[:,x]*M[:,y]).reshape(n,1)))
        f.write('（1,1）的数量:'+str(weight((M[:,x]*M[:,y]).reshape(n,1))))
        f.write("选择的合并的列权重"+str(weight_array[x,y])+'\n')
        
        add_array=(np.zeros((n,1)))
        #f.write('add_array:'+str(add_array.reshape(1,n))+'\n')
        #print('add_array:',add_array.reshape(1,n))
        M=np.hstack((M,add_array))
        #updatind the asdh
        
        
#            for _ in range(n):
#                if weights[_]==2 and M[_][m]==1:
#                    tem.remove(_)
#                    M[_][x]=0;M[_][y]=0
#                if M[_][m]==1:
#                    dept=[max(depth[x],depth[y])+1]
#                    for l in range(m):
#                        if M[_][l]==1:
#                            dept+=[depth[l]]
#                    dept.remove(depth[x]);dept.remove(depth[y])
#                    if feasible(dept,goal_depth[_]):
#                        tem.remove(_)
#                        M[_][x]=0;M[_][y]=0
#                    else:
#                        M[_][m]=0
        tem=[i for i in range(n)]      
        #weights=[weight(M[i]) for i in range(n)]   
        w=weights.copy()
        w.sort()
        for j in set(w):
            j=int(j)
            if j==1:
                continue
            if j==2:
                for _ in range(n):
                    if tagets[_]==newbase:
                        for hjk in range(m):
                            M[_][hjk]=0
                        M[_][m]=1
                        tem.remove(_)
            
            else:
                value=combinations(conl,j-2)
                tempj=[]
                for asd in tem:
                    if weights[asd]==j:
                        tempj+=[asd]
                    
                while tempj!=[]:
                    try:
                        _ = next(value)
                        re=sum_(list(_)+[newbase])
                        pop=[]
                        for i in tempj:
                            if re==tagets[i]:
                                dept=[df]
                                for l in range(m):
                                    if conl[l] in _:
                                        dept+=[depth[l]]
                                if feasible(dept,goal_depth[i]):
                                    #tem.remove(i)
                                    pop+=[i]
                                    for l in range(m):
                                        if conl[l] in _:
                                            M[i][l]=1
                                        else:
                                            M[i][l]=0
                                    M[i][m]=1
                        for asd in pop:
                            tempj.remove(asd)
                    except StopIteration:
                        break
        
        timed=time.time()-since1
        print('本次选择耗时 {:.0f}m {:.0f}s'.format(timed//60 ,timed % 60))
        f.write('本次选择耗时 {:.0f}m {:.0f}s'.format(timed//60 ,timed % 60) )
        since1=time.time()
        conl.insert(m,newbase)
        depth.insert(m,max(depth[x],depth[y])+1)
        base=conl
        weights=[weight(M[i]) for i in range(n)]
        print('合并后的重量:',weights)
        
        print('合并后的重量:',weights)
        print('合并后的M：',M)
        f.write('合并后的重量:'+str(weights)+'\n')
        f.write('合并后的M：'+str(M)+'\n')
    num=m+1
    for ad in weights:
        num+=(ad-1)
    ##de计算深度
    while 2 in weights:
            tem=[]
            line=weights.index(2)
            for kl in range(len(M[line])):
                if M[line][kl]==1:
                    tem+=[kl]
            weights[line]=1
            choice+=[tem]
            [x,y]=tem
            m+=1
            conl.insert(m,(conl[x]|conl[y])-(conl[x]&conl[y]))
            depth.insert(m,max(depth[x],depth[y])+1)
    maxde=max(depth)  
    outputd=[]
    for asd in M:
        outputd+=[depth[list(asd).index(1)]]
    print('整个优化需要的总门数：',num,'深度：',maxde)
    f.write('整个优化需要的总门数：'+str(num)+'深度：'+str(maxde)+'\n')
    time_=time.time()-since
    f.write('本次算法的运行时间 {:.0f}m {:.0f}s'.format(time_ //60 ,time_ % 60) )
    f.write('所有的节点'+str(base))
    print('整个算法的运行时间 {:.0f}m {:.0f}s'.format(time_ //60 ,time_ % 60) )
    print('所有的choice：',choice)
    f.write('所有的choice：'+str(choice)+'\n')
    print(base)
    print('深度向量',depth)
    f.write('深度向量'+str(depth)+'\n')
    f.write('输出各分量的深度'+str(outputd)+'\n')
f.close()
#    select(M,depth,goal_depth)
#    update()#M,depth,weights
       
#    m=M.shape[1]
#    j=0;to_append=array([0 for i in range(n)]).reshape(n,1)
#    while j < n:
#        if weights[j]>2**(k-i):
#            #choose and update weights and addarray
#            weight_array=np.zeros((m,m))
#            for x in range(m):
#                for y in range(x+1,m):
#                    weight_array[x,y]=weight(M[:,x]*M[:,y])
#            #print("选择合并列的权重矩阵：",weight_array)
#            max_xy=np.where(weight_array==np.max(weight_array))#全部最大行列的位置
#            r=randint(0,len(max_xy[0]))
#            (x,y)=(max_xy[0][r],max_xy[1][r])#随机选一个
#            print("选择的合并的列号",(x,y))
#            new_line=M[:,x]*M[:,y]
#            to_append=np.hstack((to_append, new_line.reshape(len(new_line),1)))#记录新列在to_append里面
#            #将合并的两列做修改
#            for c in range (len(new_line)):
#                if new_line[c]==1:
#                    M[c,x]=0;M[c,y]=0
#                    weights[c]-=1
#            #print("合并后的矩阵：",M)
#            #print("合并后的行重量：",weights)
#            
#            j-=1
#        j+=1
#    if len(to_append[0])>1:
#        M=np.hstack((M,np.delete(to_append,0,axis=1)))
#    print("阶段合并后的矩阵：",M,sep='\n')
#def select(M,depth,goal_depth):
    
    