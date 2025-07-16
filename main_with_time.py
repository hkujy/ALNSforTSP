import math                         # 导入模块 math
import random                       # 导入模块 random
import pandas as pd                 # 导入模块 pandas 并简写成 pd
import numpy as np                  # 导入模块 numpy 并简写成 np YouCans
import matplotlib.pyplot as plt     # 导入模块 matplotlib.pyplot 并简写成 plt
import time

# 子程序：读取TSPLib数据
def read_TSPLib(fileName):
    res = []
    with open(fileName, 'r') as fid:
        for item in fid:
            if len(item.strip())!=0:
                res.append(item.split())
    loadData = np.array(res).astype('float')      # 数据格式：i Xi Yi
    coordinates = loadData[:,1::]
    return coordinates

# 子程序：计算各城市间的距离，得到距离矩阵
def getDistMat(nCities, coordinates):
    max=0
    distMat = np.zeros((nCities,nCities))       # 初始化距离矩阵
    for i in range(nCities):
        for j in range(i,nCities):
            # np.linalg.norm 求向量的范数（默认求 二范数），得到 i、j 间的距离
            distMat[i][j] = distMat[j][i] = round(np.linalg.norm(coordinates[i]-coordinates[j]))
            if distMat[i][j]>max:
                max=distMat[i][j]
    return distMat,max                              # 城市间距离取整（四舍五入）

#近邻表格
def getJinlin(nCities,num,distMat):
    Jinlindis = np.zeros((nCities,1))       # 近邻距离，只有这个距离之内的可以被选择
    k=0
    j=0
    Jinlin=[]
    for dis1 in distMat:
        discopy=dis1.copy()
        discopy.sort()
        Jinlindis[k]=discopy[num]               #设置每个城市第num-1距离为半径
        k=k+1
    for dis2 in distMat:
        pointset=[]
        for i in range(0,nCities):    
            if dis2[i]<=Jinlindis[j] and i!=j:
                pointset.append(i)
        Jinlin.append(pointset)
        j=j+1
    return Jinlin    

# 子程序：计算 TSP 路径长度
def calTourMileage(tourGiven, nCities, distMat):
    # custom function caltourMileage(nCities, tour, distMat):
    # to compute mileage of the given tour
    mileageTour = distMat[tourGiven[nCities-1], tourGiven[0]]   # dist((n-1),0)
    for i in range(nCities-1):                                  # dist(0,1),...dist((n-2)(n-1))
        mileageTour += distMat[tourGiven[i], tourGiven[i+1]]
    return round(mileageTour)                     # 路径总长度取整（四舍五入）

# 子程序：绘制 TSP 路径图
def plot_tour(tour, value, coordinates):
    # custom function plot_tour(tour, nCities, coordinates)
    num = len(tour)
    x0, y0 = coordinates[tour[num - 1]]
    x1, y1 = coordinates[tour[0]]
    plt.scatter(int(x0), int(y0), s=15, c='r')      # 绘制城市坐标点 C(n-1)
    plt.plot([x1, x0], [y1, y0], c='b')             # 绘制旅行路径 C(n-1)~C(0)
    for i in range(num - 1):
        x0, y0 = coordinates[tour[i]]
        x1, y1 = coordinates[tour[i + 1]]
        plt.scatter(float(x0), float(y0), s=15, c='r')  # 绘制城市坐标点 C(i)
        plt.plot([x1, x0], [y1, y0], c='b')         # 绘制旅行路径 C(i)~C(i+1)
    plt.xlabel("Total mileage of the tour:{:.1f}".format(value))
    plt.title("Optimization tour of TSP{:d}".format(num))  # 设置图形标题
    plt.show()

def distroy1(tourNow,nCities,distMat,num):                                                      #贪心摧毁前num个
    LIST=[]
    list=[]
    mileageTour0 = distMat[tourNow[nCities-1], tourNow[0]]+distMat[tourNow[0], tourNow[1]]
    list.append([tourNow[0],mileageTour0])
    for i in range(1,nCities-1):
        mileageTour = distMat[tourNow[i-1], tourNow[i]]+distMat[tourNow[i], tourNow[i+1]]
        list.append([tourNow[i],mileageTour])
    mileageTour1 = distMat[tourNow[nCities-2], tourNow[nCities-1]]+distMat[tourNow[nCities-1], tourNow[0]]
    list.append([tourNow[nCities-1],mileageTour1])
    my_lest=sorted(list , key=lambda k: k[1],reverse=True)                                     #二维坐标进行降序
    for j in range(num):
        LIST.append(my_lest[j][0])                                                             #LIST是要拆除的坐标
    tourNew=[]                                                                                  #这是初始解拆除LIST要拆除坐标后的解
    for l in range(nCities):
        if tourNow[l] not in LIST:
            tourNew.append(tourNow[l])
    return LIST,tourNew                                                                     #返回的是拆除名单和出去拆除名单后的解

def distroy2(tourNow,nCities,num):                                                              #随机摧毁num个
    LIST=random.sample(range(nCities),num)
    tourNew=[]                                                                                  #这是初始解拆除LIST要拆除坐标后的解
    for l in range(nCities):
        if tourNow[l] not in LIST:
            tourNew.append(tourNow[l])
    return LIST,tourNew                                                                       #返回的是拆除名单和出去拆除名单后的解

def distroy3(tourNow,nCities,num):                                                              #随机摧毁num个连续城市
    LIST=[]
    x=random.randint(0,nCities-1)
    tourNow1=[]
    for i in range(nCities):
        tourNow1.append(tourNow[i])
    tourNow2= tourNow1[x:] + tourNow1[:x]
    for j in range(num):
        LIST.append(tourNow2[j])
    tourNew=[]                                                                                  #这是初始解拆除LIST要拆除坐标后的解
    for l in range(nCities):
        if tourNow[l] not in LIST:
            tourNew.append(tourNow[l])
    return LIST,tourNew                                                                       #返回的是拆除名单和出去拆除名单后的解

def distroy4(tourNow,nCities,num,jinlin):                                              #非近邻摧毁
    LIST=[]
    while True:
        x=random.randint(0,nCities-2)
        if tourNow[x] not in LIST:
            if tourNow[x+1] in jinlin[tourNow[x]] and tourNow[x-1] in jinlin[tourNow[x]]:
                p=0.6
            elif tourNow[x+1] in jinlin[tourNow[x]] or tourNow[x-1] in jinlin[tourNow[x]]:
                p=0.8
            else:
                p=1
            r=random.random()
            if r<p:
                LIST.append(tourNow[x])
        if len(LIST)==num:
            break
    tourNew=[]                                                                                  #这是初始解拆除LIST要拆除坐标后的解
    for l in range(nCities):
        if tourNow[l] not in LIST:
            tourNew.append(tourNow[l])    
    return LIST,tourNew 

def distroy5(tourNow,nCities,num,jinlin,Jinnum):                                              #近邻摧毁
    LIST=[]
    x=random.randint(0,nCities-1)
    if num>Jinnum:
        for ii in range(Jinnum):
            LIST.append(jinlin[tourNow[x]][ii])
        while True:
            y=random.randint(0,nCities-1)
            if tourNow[y] not in LIST:
                LIST.append(tourNow[y])
                if len(LIST)==num:
                    break
    else:
        y=random.randint(0,nCities-1)
        LIST1=random.sample(range(Jinnum),num)
        for ii in range(num):
            LIST.append(jinlin[tourNow[x]][LIST1[ii]])
    tourNew=[]                                                                                  #这是初始解拆除LIST要拆除坐标后的解
    for l in range(nCities):
        if tourNow[l] not in LIST:
            tourNew.append(tourNow[l])    
    return LIST,tourNew 

def repair1(LIST,tourNew,distMat,num):                                  #噪声贪心修复
    tourstart=tourNew
    for i in range(num):
        value=[]
        for j in range(len(tourstart)-1):
            if j==0:
                value1=distMat[tourstart[j], LIST[i]]+distMat[tourstart[-1], LIST[i]]-distMat[tourstart[-1], tourstart[j]]
                value.append(value1)
            else:
                value1=distMat[tourstart[j], LIST[i]]+distMat[tourstart[j-1], LIST[i]]-distMat[tourstart[j-1], tourstart[j]]
                value.append(value1)
        bestvalue=min(value)
        position=value.index(bestvalue)
        tourstart.insert(position,LIST[i])
    return tourstart

def repair2(LIST,tourNew,num,jinlin):                                #k近邻修复
    tourNow1=[]
    for i in range(len(tourNew)):
        tourNow1.append(tourNew[i])
    for j in range(num):
        x=LIST[j]
        n=0
        while True:
            n=n+1
            y=random.sample(jinlin[x],1)
            if y in tourNow1:
                break
            if n>20:
                zindex=random.randint(0, len(tourNew)-1)
                y=tourNew[zindex]
                break
        yindex=tourNow1.index(y)
        tourNow1.insert(yindex,LIST[j])
    return tourNow1

def repair3(LIST,tourNew,distMat,num,u,temp_max):                     #噪声贪心修复，u表示噪音系数
    tourstart=tourNew
    for i in range(num):
        value=[]
        for j in range(len(tourstart)-1):
            r3=random.random()
            if j==0:
                value1=distMat[tourstart[j], LIST[i]]+distMat[tourstart[-1], LIST[i]]-distMat[tourstart[-1], tourstart[j]]+temp_max*(2*r3-1)*u
                value.append(value1)
            else:
                value1=distMat[tourstart[j], LIST[i]]+distMat[tourstart[j-1], LIST[i]]-distMat[tourstart[j-1], tourstart[j]]+temp_max*(2*r3-1)*u
                value.append(value1)
        bestvalue=min(value)
        position=value.index(bestvalue)
        tourstart.insert(position,LIST[i])
    return tourstart        


def crossover(father,mother):
    num_city = len(father)
    #indexrandom = [i for i in range(int(0.4*cronum),int(0.6*cronum))]
    index_random = [i for i in range(num_city)]
    pos = random.choice(index_random)
    son1 = father[0:pos]
    son2 = mother[0:pos]
    son1.extend(mother[pos:num_city])
    son2.extend(father[pos:num_city])
    
    index_duplicate1 = []
    index_duplicate2 = []
    for i in range(pos, num_city):
        for j in range(pos):
            if son1[i] == son1[j]:
                index_duplicate1.append(j)
            if son2[i] == son2[j]:
                index_duplicate2.append(j)
    num_index = len(index_duplicate1)
    for i in range(num_index):
        son1[index_duplicate1[i]], son2[index_duplicate2[i]] = son2[index_duplicate2[i]], son1[index_duplicate1[i]]
    return son1,son2

# 子程序：初始化模拟退火算法的控制参数
def initParameter():
    nMarkov =80   # Markov链长度，也即内循环运行次数
    Maxinter=150
    Jinnum=20
    nummax=30
    a=0.2
    b=0.96
    u=0.1
    return nMarkov,Maxinter,Jinnum,nummax,a,b,u

def initT(nCities,distMat):
    N=100
    group=[]
    for i in range(N):
        tourNow=np.random.permutation(nCities)
        valueNow= calTourMileage(tourNow,nCities,distMat)
        group.append(valueNow)
    max1=max(group)
    min1=min(group)
    # T=max1-min1
    # T=200*nCities
    T=200
    return T

def main():
    start=time.time()
    coordinates =read_TSPLib('a280.txt')                       #案例点的坐标
    nMarkov,Maxinter,Jinnum,nummax,a,b,u= initParameter()           # 调用子程序，获得设置参数                                                         #近邻系数
    nCities = coordinates.shape[0]                                       #根据输入的城市坐标 获得城市数量 nCities
    distMat,temp_max = getDistMat(nCities, coordinates) 
    # print(temp_max)                                                     #最长的两点距离
    jinlin=getJinlin(nCities,Jinnum,distMat)                              #制作近邻表
    tourNow=np.random.permutation(nCities)                                #随机生成初始解
    valueNow = calTourMileage(tourNow,nCities,distMat)
    recordBest = []                                                       # 初始化 最优路径记录表
    recordNow  = []                                                   # 初始化 最优路径记录表
    tourBest  = tourNow                   
    valueBest = valueNow     
    T=initT(nCities,distMat)
    pd1=0.2
    pd2=0.2
    pd3=0.2
    pd4=0.2
    pd5=0.2
    pr1=0.5
    pr3=0.5
    SL=[]                      #记录参数变化
    # PP1=[]
    # PP2=[]
    # PP3=[]
    # PP4=[]
    # PP5=[]
    for wai in range(Maxinter):
            T=T*0.99
            timed1=0
            timed2=0
            timed3=0
            timed4=0
            timed5=0
            timer1=0
            timer3=0
            sd1=0 
            sd2=0 
            sd3=0  
            sd4=0 
            sd5=0 
            sr1=0 
            sr3=0 
            td1=0
            td2=0
            td3=0
            td4=0
            td5=0
            tr1=0
            tr2=0
            n=0
            a=a*b
            while n<nMarkov:
                n=n+1            
                r1=random.uniform(0,pd1+pd2+pd3+pd4+pd5)                                           #r1控制破坏
                r2=random.uniform(0,pr1+pr3)                                         #r2控制修复
                num=random.randint(2,nummax)                                        #破坏点的个数(不改了，效果不大) 
                if r1<pd1:                                                                 # 五种破坏算子
                    start_time = time.perf_counter()
                    LIST,tourNew=distroy1(tourNow,nCities,distMat,num) 
                    timed1=timed1+1
                    end_time = time.perf_counter()
                    td1=td1+end_time-start_time
                elif r1<pd2+pd1 and r1>=pd1:
                    start_time = time.perf_counter()
                    LIST,tourNew=distroy2(tourNow,nCities,num)
                    timed2=timed2+1
                    end_time = time.perf_counter()
                    td2=td2+end_time-start_time
                elif r1<pd3+pd2+pd1 and r1>=pd2+pd1:
                    start_time = time.perf_counter()
                    LIST,tourNew=distroy3(tourNow,nCities,num)
                    timed3=timed3+1
                    end_time = time.perf_counter()
                    td3=td3+end_time-start_time
                elif r1<pd4+pd3+pd2+pd1 and r1>=pd3+pd2+pd1:
                    start_time = time.perf_counter()
                    LIST,tourNew=distroy4(tourNow,nCities,num,jinlin)    
                    timed4=timed4+1 
                    end_time = time.perf_counter()
                    td4=td4+end_time-start_time  
                else:
                    start_time = time.perf_counter()
                    LIST,tourNew=distroy5(tourNow,nCities,num,jinlin,Jinnum) 
                    timed5=timed5+1   
                    end_time = time.perf_counter()
                    td5=td5+end_time-start_time
                random.shuffle(LIST)                          #移除表格顺序打乱     
                if r2<pr1:
                    start_time = time.perf_counter()
                    tourNew=repair1(LIST,tourNew,distMat,num)    
                    timer1=timer1+1  
                    end_time = time.perf_counter()
                    tr1=tr1+end_time-start_time             
                else:
                    start_time = time.perf_counter()
                    tourNew=repair3(LIST,tourNew,distMat,num,u,temp_max)  
                    timer3=timer3+1
                    end_time = time.perf_counter()
                    tr2=tr2+end_time-start_time   
                valueNew = calTourMileage(tourNew,nCities,distMat)
                if valueNew<=valueNow:
                    if r1<pd1:                                                                 # 五种破坏算子
                        sd1=sd1+2
                    elif r1<pd2+pd1 and r1>=pd1:
                        sd2=sd2+2
                    elif r1<pd3+pd2+pd1 and r1>=pd2+pd1:
                        sd3=sd3+2
                    elif r1<pd4+pd3+pd2+pd1 and r1>=pd3+pd2+pd1:
                        sd4=sd4+2
                    else:
                        sd5=sd5+2
                    if r2<pr1:
                        sr1=sr1+2             
                    else:
                        sr3=sr3+2                             
                    accept = True
                    if valueNew < valueBest:
                        if r1<pd1:                                                                 # 五种破坏算子
                            sd1=sd1+2
                        elif r1<pd2+pd1 and r1>=pd1:
                            sd2=sd2+2
                        elif r1<pd3+pd2+pd1 and r1>=pd2+pd1:
                            sd3=sd3+2
                        elif r1<pd4+pd3+pd2+pd1 and r1>=pd3+pd2+pd1:
                            sd4=sd4+2
                        else:
                            sd5=sd5+2
                        if r2<pr1:
                            sr1=sr1+2           
                        else:
                            sr3=sr3+2     
                        tourBest = tourNew
                        valueBest = valueNew 
                else:
                    # pAccept= math.exp((valueNow-valueNew)/T)
                    # pp=random.random()
                    # if pAccept > pp :
                    #      accept = True
                    # else:
                    #     accept = False
                    if valueNew<valueBest+a*valueBest:
                        accept = True
                    else:
                        accept = False
                # 保存新解
                if accept == True:                      # 如果接受新解，则将新解保存为当前解
                    tourNow = tourNew               
                    valueNow = valueNew
                    if r1<pd1:                                                                 # 五种破坏算子
                        sd1=sd1+1
                    elif r1<pd2+pd1 and r1>=pd1:
                        sd2=sd2+1
                    elif r1<pd3+pd2+pd1 and r1>=pd2+pd1:
                        sd3=sd3+1
                    elif r1<pd4+pd3+pd2+pd1 and r1>=pd3+pd2+pd1:
                        sd4=sd4+1  
                    else:
                        sd5=sd5+1
                    if r2<pr1:
                        sr1=sr1+1             
                    else:
                        sr3=sr3+1          
            sdall=sd1/(timed1+0.001)+sd2/(0.001+timed2)+sd3/(0.001+timed3)+sd4/(0.001+timed4)+sd5/(timed5+0.001)   #概率动态调整
            srall=sr1/(timer1+0.001)+sr3/(timer3+0.001)
            pd1=pd1*0.3+(sd1/(timed1+0.001))/(0.001+sdall)*0.7
            pd2=pd2*0.3+(sd2/(0.001+timed2))/(0.001+sdall)*0.7
            pd3=pd3*0.3+(sd3/(0.001+timed3))/(0.001+sdall)*0.7
            pd4=pd4*0.3+(sd4/(0.001+timed4))/(0.001+sdall)*0.7
            pd5=pd5*0.3+(sd5/(timed5+0.001))/(0.001+sdall)*0.7
            pr1=pr1*0.3+(sr1/(0.001+timer1))/(srall+0.001)*0.7
            pr3=pr3*0.3+(sr3/(0.001+timer3))/(srall+0.001)*0.7
            recordBest.append(valueBest)                
            recordNow.append(valueNow)                  # 将当前路径长度追加到 当前路径记录表
            pdall=pd1/(td1*td1+0.001)+pd2/(td2*td2+0.001)+pd3/(td3*td3+0.001)+pd4/(td4*td4+0.001)+pd5/(td5*td5+0.001)
            prall=pr1/(tr1*tr1+0.001)+pr3/(tr2*tr2+0.001)
            pd1=pd1/(td1*td1+0.001)/pdall
            pd2=pd2/(td2*td2+0.001)/pdall
            pd3=pd3/(td3*td3+0.001)/pdall
            pd4=pd4/(td4*td4+0.001)/pdall
            pd5=pd5/(td5*td5+0.001)/pdall
            pr1=pr1/(tr1*tr1+0.001)/prall
            pr3=pr3/(tr2*tr2+0.001)/prall          






            # PP1.append(pr1)
            # PP2.append(pr3)
            # PP3.append(pd3)
            # PP4.append(pd4)
            # PP5.append(pd5)
            # SL.append(valueBest)
            # print(f'求解{wai}次阈值为{T}历史最优结果为     {valueBest} {valueNow}   ')  
            print(f'求解{wai}次阈值为{a}历史最优结果为 {valueBest} {valueNow}   {pd1}  {pd2}  {pd3}  {pd4}  {pd5}  {pr1}  {pr3}        ')  
    end=time.time()
    print(f'时间为{end-start}秒')  
    # print(valueBest)
    print(tourBest)
    # print(PP1)
    # print(PP2)
    # print(PP3)
    # print(PP4)
    # print(PP5)
    # print(SL)
    print(recordBest)
    plot_tour(tourBest, valueBest, coordinates)
    plt.title("Optimization result of TSP{:d}".format(nCities)) # 设置图形标题
    plt.plot(np.array(recordBest),'b-', label='Best')           # 绘制 recordBest曲线
    plt.plot(np.array(recordNow),'g-', label='Now')             # 绘制 recordNow曲线
    plt.xlabel("iter")                                          # 设置 x轴标注
    plt.ylabel("mileage of tour")                               # 设置 y轴标注
    plt.legend()                                                # 显示图例
    plt.show()
    exit()

if __name__ == '__main__':
    main()

