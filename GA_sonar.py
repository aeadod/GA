import numpy as np
import pandas as pd
import random
import math
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  


sonar = pd.read_csv('sonar.all-data',header=None,sep=',',engine='python')
sonar1 = sonar.iloc[0:208,0:60]
sonar2 = np.mat(sonar1)

popsize = 20  # 初始化种群规模 20个
maxIteration = 150 # 最大迭代次数
Pcrossover = 0.9  # 交配概率
Pmutation = 0.001  # 变异概率

# 遗传算法 d表示选择d维特征
def GA(d):
    # 初始化种群
    # populations = np.random.randint(0,2,(popsize,BitLength))
    populations = np.zeros((popsize, 60))

    for ii in range(popsize):  # 定义种群个体数维popsize
        a = np.zeros(60-d)
        b = np.ones(d)          # 将选择的d维特征定义维个体c中的1
        c = np.append(a,b)
        np.random.shuffle(c)     # 随机生成一个d维的个体
        populations[ii] = c      # 初代的种群为 population，共有n个个体
    
    # 遗传算法的迭代次数为 maxIteration
    fitness_change = np.zeros(maxIteration)
    for ii in range(maxIteration):
        fitness = np.zeros(popsize)     # fitness为每一个个体的适应度值
        for jj in range(popsize):
            fitness[jj] = Jd(populations[jj], d)   # 计算每一个个体的适应度值
        populations = selection(populations, fitness) # 通过概率选择产生新一代的种群
        populations = crossover(populations)          # 通过交叉产生新的个体
        populations = mutation(populations)           # 通过变异产生新的个体
        fitness_change[ii] = max(fitness)           # 找出每一代的适应度最大的染色体的适应度值

    # 随着迭代的进行，每个个体的适应度应该会不断增加，所以总的适应度值fitness求平均应该会变大
    best_fitness = max(fitness)
    best_people = populations[fitness.argmax()]

    return best_people,best_fitness,fitness_change,populations

# 个体适应度函数 Jd(x)，x是d维特征向量
def Jd(x,d):
    # 从特征向量x中提取出相应的特征
    Feature = np.zeros(d)   # 数组Feature存放 x选择的是哪d个特征
    k = 0
    for ii in range(60):
        if (x[ii]==1 and k<30):
            Feature[k] = ii
            k+=1

    # 将30个特征从sonar2数据集中取出重组成一个208*d的矩阵sonar3
    sonar3 = np.zeros((208, 1))
    for ii in range(d):
        p = Feature[ii]
        p = p.astype(int)
        q = sonar2[:,p]
        q = q.reshape(208, 1)
        sonar3 = np.append(sonar3,q,axis=1)
    sonar3 = np.delete(sonar3,0,axis=1)

    # 求类间离散度矩阵Sb
    sonar3_1 = sonar3[0:97,:]   # sonar数据集分为两类
    sonar3_2 = sonar3[97:208,:]
    m = np.mean(sonar3,axis=0)  # 总体均值向量
    m1 = np.mean(sonar3_1,axis=0)   #第一类的均值向量
    m2 = np.mean(sonar3_2,axis=0)   #第二类的均值向量
    m = m.reshape(d,1)
    m1 = m1.reshape(d,1)
    m2 = m2.reshape(d,1)
    Sb = ((m1 - m).dot((m1 - m).T)*(97/208) + (m2 - m).dot((m2 - m).T)*(111/208)) 

    #求类内离散度矩阵Sw
    S1 = np.zeros((d,d))
    S2 = np.zeros((d,d))
    for ii in range(97):
        S1 += (sonar3_1[ii].reshape(d,1)-m1).dot((sonar3_1[ii].reshape(d,1)-m1).T)
    S1 = S1/97
    for ii in range(111):
        S2 += (sonar3_2[ii].reshape(d,1)-m2).dot((sonar3_2[ii].reshape(d,1)-m2).T)
    S2 = S2/111

    Sw = (S1*(97/208)) + S2*(111/208)

    J1 = np.trace(Sb)
    J2 = np.trace(Sw)
    Jd = J1/J2
    return Jd

#轮盘赌选择
def selection(populations, fitness):
    fitness_sum = np.zeros(popsize)
    for ii in range(popsize):
        if ii == 0:
            fitness_sum[ii] = fitness[ii]
        else:
            fitness_sum[ii] = fitness[ii] + fitness_sum[ii-1]
    
    for ii in range(popsize):
        fitness_sum[ii] = fitness_sum[ii] / sum(fitness)
    
    # 选择新的种群
    populations_new = np.zeros((popsize, 60))
    for ii in range(popsize):
        rand = np.random.uniform(0, 1)
        for jj in range(popsize):
            if jj == 0:
                if rand<=fitness_sum[jj]:
                    populations_new[ii] = populations[jj]
            else:
                if fitness_sum[jj-1]<rand and rand<=fitness_sum[jj]:
                    populations_new[ii] = populations[jj]

    return populations_new

# 交叉操作
def crossover(populations):
    father = populations[0:10,:]
    mother = populations[10:,:]
    np.random.shuffle(father)
    np.random.shuffle(mother)
    for ii in range(10):
        father_1 = father[ii]
        mother_1 = mother[ii]
        one_zero = []
        zero_one = []
        for jj in range(60):
            if father_1[jj]==1 and mother_1[jj]==0:
                one_zero.append(jj)
            if father_1[jj]==0 and mother_1[jj]==1:
                zero_one.append(jj)
        length1 = len(one_zero)
        length2 = len(zero_one)
        length = max(length1,length2)
        half_length = int(length/2)     # half_length作为交叉位数
        # 进行交叉操作
        for k in range(half_length):
            p = one_zero[k]
            q = zero_one[k]
            father_1[p] = 0
            mother_1[p] = 1
            father_1[q] = 1
            mother_1[q] = 0
        father[ii] = father_1       # 将交叉后的个体替换原来的个体
        mother[ii] = mother_1
    populations = np.append(father, mother, axis=0)
    return populations

# 变异操作
def mutation(populations):
    for ii in range(popsize):
        c = np.random.uniform(0,1)
        if c<=Pmutation:
            mutation_s = populations[ii]
            zero = []           # zero存的是变异个体中的第几个数为0
            one = []            # one存的是变异个体中第几个数为1
            for jj in range(60):
                if mutation_s[jj]==0:
                    zero.append(jj)
                else:
                    one.append(jj)
            
            if (len(zero)!=0) and (len(one)!=0):
                a = np.random.randint(0, len(zero))     # e是随机选择由0变为1的位置
                b = np.random.randint(0, len(one))      # f是随机选择由1变为0的位置
                e = zero[a]
                f = one[b]
                mutation_s[e] = 1
                mutation_s[f] = 0
                populations[ii] = mutation_s
    return populations

if __name__=='__main__':
    best_d = np.zeros(60)       #judge存的是每一个维数的最优适应度
    for d in range(20,21):
        best_people, best_fitness, fitness_change, best_popuplation = GA(d)     # fitness_change是遗传算法在迭代过程中适应度变化
        best_d[d-1] = best_fitness      # best是每一维数迭代到最后的最优适应度，用于比较
        print('在取%d维的时候，通过遗传算法得出的最优适应度值为：%.6f'%(d,best_fitness))
        print('最优染色体为：')
        print(best_people)
        localtion = np.array(np.where(best_people == 1))[0]
        localtion = localtion + 1
        localtion = localtion.tolist()
        print('选择的特征分别为：')
        print(localtion)

    # 画图
    # x = np.arange(1,61,1)
    # plt.xlabel('dimension')
    # plt.ylabel('fitness')
    # # plt.ylim((0,0.3))
    # plt.plot(x, best_d, 'r')
    # plt.savefig("Sonar_best_d.jpg", dpi=2000)
    # plt.show()

    
    x = np.arange(0, maxIteration, 1)
    plt.xlabel('Number of iteration')
    plt.ylabel('fitness')
    # plt.ylim((0, 0.1))
    plt.plot(x, fitness_change, 'r')
    plt.show()

