import matplotlib.pyplot as plt
import matplotlib
import numpy as np

N_CITIES = 15  # 城市数量（DNA长度）
CROSS_RATE = 0.1    #交叉概率
MUTATE_RATE = 0.02  #变异概率
POP_SIZE = 500      #个体数量
N_GENERATIONS = 500 #迭代次数

class GA(object):
    def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size, ):#城市初始化
        self.DNA_size = DNA_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size
        self.pop = np.vstack([np.random.permutation(DNA_size) for _ in range(pop_size)])

    def translateDNA(self, DNA, city_position):  #将个体携带的基因转化为坐标
        line_x = np.empty_like(DNA, dtype=np.float64)
        line_y = np.empty_like(DNA, dtype=np.float64)
        for i, d in enumerate(DNA):
            city_coord = city_position[d]
            line_x[i, :] = city_coord[:, 0]
            line_y[i, :] = city_coord[:, 1]
        return line_x, line_y

    def get_fitness(self, line_x, line_y):#采用上文提到的适应度计算方法
        total_distance = np.empty((line_x.shape[0],), dtype=np.float64)
        for i, (xs, ys) in enumerate(zip(line_x, line_y)):
            total_distance[i] = np.sum(np.sqrt(np.square(np.diff(xs)) + np.square(np.diff(ys))))
        fitness = np.exp(self.DNA_size * 2 / total_distance)
        return fitness, total_distance


    def select(self, fitness):#选择算子，轮盘赌算法
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size,
                               replace=True, p=fitness / fitness.sum())
        return self.pop[idx]



    def crossover(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            # 随机选择另一个父代个体
            i_ = np.random.randint(0, self.pop_size, size=1)
            # 随机选择交叉点
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)
            keep_city = parent[~cross_points]
            swap_city = pop[i_, np.isin(pop[i_].ravel(), keep_city, invert=True)]
            parent[:] = np.concatenate((keep_city, swap_city))
        return parent



    def mutate(self, child):#随机选择两个位置交换基因
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                swap_point = np.random.randint(0, self.DNA_size)
                swapA, swapB = child[point], child[swap_point]
                child[point], child[swap_point] = swapB, swapA
        return child


    def evolve(self, fitness):#将选择交叉和变异算子应用到种群中
        pop = self.select(fitness)
        pop_copy = pop.copy()
        for parent in pop:
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop


class TravelSalesPerson(object):
    def __init__(self, n_cities):
        self.city_position = np.random.rand(n_cities, 2)
        plt.ion()

    def plotting(self, lx, ly, total_d):
        plt.cla()
        plt.scatter(self.city_position[:, 0].T, self.city_position[:, 1].T, s=100, c='k')
        plt.plot(lx.T, ly.T, 'r-')
        plt.text(-0.05, -0.05, "Total distance=%.2f" % total_d, fontdict={'size': 20, 'color': 'red'})
        plt.xlim((-0.1, 1.1))
        plt.ylim((-0.1, 1.1))
        plt.pause(0.01)

ga = GA(DNA_size=N_CITIES, cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE)
env = TravelSalesPerson(N_CITIES)
aa=[]
for generation in range(N_GENERATIONS):
    lx, ly = ga.translateDNA(ga.pop, env.city_position)
    fitness, total_distance = ga.get_fitness(lx, ly)
    ga.evolve(fitness)
    best_idx = np.argmax(fitness)
    print('Gen:', generation, '| best fit: %.2f' % fitness[best_idx],)
    aa.append(fitness[best_idx])
    env.plotting(lx[best_idx], ly[best_idx], total_distance[best_idx])

plt.ioff()
plt.show()

#x = np.linspace(0, 2 * np.pi, 100)


plt.plot(aa)

plt.xlabel('generation')
plt.ylabel('fitness')

plt.show()