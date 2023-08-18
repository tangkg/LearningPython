"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
###速度=5m/s，一个格子的长度为50m
import numpy as np


class Maze(object):
    def __init__(self):
        super(Maze, self).__init__()

        self.n_sensor = 2 ##sensor点的个数
        ###动作空间对应着无人机的飞行移动方向和选哪个sensor进行信息更新的组合
        self.n_actions = 5*(self.n_sensor+1)
        self.n_features = self.n_sensor + 4
        ##具有的状态特征的个数：包含每个sensor在数据中心处的信息年龄，UAV的位置,
        # UAV剩余时间和到达终点需要时间的差值,无人机剩余能量和到达终点需要能量的差值(不包含，用拉格朗日乘子方法合并到主函数中)
        self.width = 20
        self.length = 20
        ##uav飞行的范围，单位化之后的值
        self.sensor_x = [5, 10,]  ## 12, 14, 4, 7, 17,2,
        self.sensor_y = [9, 8,]  ## 14, 5, 18, 13, 12,9
        self.stop_point = [19, 19] ##代表uav的终点
        self.start_point = [0, 0] ##代表uav的起点
        # self.base = [10, 10]  ##数据中心的位置
        self.uav_p = self.start_point.copy()  # UAV当前的位置
        self.max_e = 85 ##uav最大能量
        self.remaining_e = self.max_e   ##UAV的剩余能量
        self.lamda = [1,1,1,1,1,1,1]  ##代表每个sensor点的信息的重要性
        self.max_aoi = 30  ###定义AOI的最大值
        self.max_time = 60  ##定义UAV的最大飞行时间
        # self.bs_i = [1,1] ##sensor的信息在数据中心处的信息年龄
        self.aoi = [1,1,1,1,1,1,1] ##sensor的信息在UAV处的信息年龄, 1, 1
        self.uav_path = [] ##记录UAV的飞行轨迹
        self.fly_e = 1.27##36.2 #旋翼UAV飞行时消耗的能量                （209  ##UAV直线飞行50m(一个格子)所需要的能量）
        self.hover_e = 1##33.3   ###旋翼UAV悬停时消耗的能量                                   （#873.5 ##UAV转弯90时需要多耗费的能量）
        self.sensor1_aoi = [1]
        self.sensor2_aoi = [1]
        self.sensor3_aoi = [1]
        self.sensor4_aoi = [1]
        self.sensor5_aoi = [1]
        self.sensor6_aoi = [1]
        self.sensor7_aoi = [1]
        # self.sensor8_aoi = [1]
        # self.sensor9_aoi = [1]
        self.flag3 = False

    def initialize(self):
        self.uav_path = []
        self.uav_p = self.start_point.copy() #初始化UAV的初始位置
        self.uav_path.append(self.uav_p.copy())
        self.remaining_e = self.max_e  ##UAV的剩余能量
        self.remaining_t = self.max_time  ##UAV剩余的飞行时间
        self.aoi = [1,1,1,1,1,1,1]  ##sensor的信息在UAV处的信息年龄的初始化, 1, 1
        self.aoi_ = [1,1,1,1,1,1,1]  ##, 1, 1
        self.sensor1_aoi = [1]
        self.sensor2_aoi = [1]
        self.sensor3_aoi = [1]
        self.sensor4_aoi = [1]
        self.sensor5_aoi = [1]
        self.sensor6_aoi = [1]
        self.sensor7_aoi = [1]
        # self.sensor8_aoi = [1]
        # self.sensor9_aoi = [1]
        ##将state的信息存入到列表state中
        state = []
        for i in range(self.n_sensor):  ###sensor在UAV处的信息年龄
            state.append(self.aoi[i]/self.max_time)
        state.append(self.uav_p[0]/(self.length-1))   ###UAV的位置
        state.append(self.uav_p[1]/(self.width-1))
        ###UAV剩余的时间和到达终点所需要的时间差值
        more_time = self.remaining_t-(abs(self.uav_p[0]-self.stop_point[0])+
                                        abs(self.uav_p[1]-self.stop_point[1]))
        state.append(more_time/self.max_time)
        ###UAV的剩余能量与到达终点并维持飞行N个时隙的能量差值
        # state.append((self.remaining_e-(abs(self.uav_p[0]-self.stop_point[0])+
        #                                 abs(self.uav_p[1]-self.stop_point[1]))*self.fly_e
        #               -(more_time*self.hover_e))/self.max_e)
        state.append((self.remaining_e-self.remaining_t*self.fly_e)/self.max_e)
        return np.array(state)

    def step(self, action):
        move = action % 5  ###选择的是上、下、左、右、不动中的某个动作
        sensor = action // 5  ##选择哪个snesor更新aoi
        ####UAV位置的更新
        if move == 0: ##上
            if self.uav_p[1] < self.width-1:  ###判断UAV是否在环境的上边界，不是
                self.uav_p[1] += 1
                self.remaining_e -= self.fly_e
            else:
                self.remaining_e -= self.hover_e
        elif move == 1:  ##下
            if self.uav_p[1] > 0:
                self.uav_p[1] -= 1
                self.remaining_e -= self.fly_e
            else:
                self.remaining_e -= self.hover_e
        elif move == 2:  ##左
            if self.uav_p[0] > 0:
                self.uav_p[0] -= 1
                self.remaining_e -= self.fly_e
            else:
                self.remaining_e -= self.hover_e
        elif move == 3:  ##右
            if self.uav_p[0] < self.length-1:
                self.uav_p[0] += 1
                self.remaining_e -= self.fly_e
            else:
                self.remaining_e -= self.hover_e
        elif move == 4:  ##不动
            self.remaining_e -= self.hover_e

        for i in range(self.n_sensor):  ##首先将每个sensor的aoi值加1
            self.aoi[i] += 1
            if self.aoi[i] < self.max_aoi:
                self.aoi_[i] += 1

        ##判断UAV选择哪一个sensor进行信息更新
        if sensor != 0:
            if (self.uav_p[0]-self.sensor_x[sensor-1])**2+(self.uav_p[1]-self.sensor_y[sensor-1])**2 <= 9:###判断UAV是否
                #和选择的sensor点传数据，要在sensor的功率发射的覆盖范围内
                # if self.remaining_e >= (1.03*10**(-3) * (((self.uav_p[0] - self.base[0])*50) ** 2 +
                #                                             ((self.uav_p[1] - self.base[1])*50) ** 2 + 100**2))/100:
                self.aoi[sensor-1] = 1   ###将要上传的sensor在数据中心处的aoi值置为1
                self.aoi_[sensor - 1] = 1
        self.remaining_t -= 1
        self.sensor1_aoi.append(self.aoi_[0])
        self.sensor2_aoi.append(self.aoi_[1])
        self.sensor3_aoi.append(self.aoi_[2])
        self.sensor4_aoi.append(self.aoi_[3])
        self.sensor5_aoi.append(self.aoi_[4])
        self.sensor6_aoi.append(self.aoi_[5])
        self.sensor7_aoi.append(self.aoi_[6])
        # self.sensor8_aoi.append(self.aoi_[7])
        # self.sensor9_aoi.append(self.aoi_[8])
        self.uav_path.append(self.uav_p.copy())

        ###判断UAV能否飞回终点
        more_time = self.remaining_t - (abs(self.uav_p[0]-self.stop_point[0]) + abs(self.uav_p[1]-self.stop_point[1]))
        if (more_time) < 0:
            flag1 = True ##不能飞到终点
        else:
            flag1 = False
        # ##UAV剩余的能量不足以飞回终点并且飞行N个时隙
        # # if (self.remaining_e - (abs(self.uav_p[0]-self.stop_point[0])+
        # #                         abs(self.uav_p[1]-self.stop_point[1]))*self.fly_e - (more_time*self.hover_e)) < 0:
        if self.remaining_e - self.remaining_t*self.fly_e < 0:
            flag2 = True
        else:
            flag2 = False
        #
        if self.remaining_t == 0:
            flag3 = True
        else:
            flag3 = False

        r_aoi = 0
        r_aoi_ = 0
        ###AOI
        for i in range(self.n_sensor):
            r_aoi -= self.lamda[i] * self.aoi[i]/self.max_time
            r_aoi_ -= self.lamda[i] * self.aoi_[i]
        r = r_aoi
        r_ = r_aoi_
        # c_r = np.min([0, self.remaining_e])
        ###########若UAV在规定时间内不能到达终点，将reward值要加上在UAV到达终点所需要的最短时间内的每个时隙内aoi的值
        if flag1:
            self.flag3 = False
            done = True
            r -= (abs(self.uav_p[0]-self.stop_point[0]) + abs(self.uav_p[1]-self.stop_point[1]))
        elif flag2:
            self.flag3 = False
            done = True
            r -= self.remaining_t
        elif flag3:
            self.flag3 = True
            done = True
            r += self.n_sensor
        else:
            self.flag3 = False
            done = False

        state = []
        for i in range(self.n_sensor):
            state.append(self.aoi[i]/self.max_time)
        state.append(self.uav_p[0]/(self.length - 1))
        state.append(self.uav_p[1] / (self.width - 1))
        state.append(more_time/self.max_time)
        # state.append((self.remaining_e - (abs(self.uav_p[0]-self.stop_point[0]) +
        #                                   abs(self.uav_p[1]-self.stop_point[1]))*self.fly_e-
        #               more_time*self.hover_e)/self.max_e)
        state.append((self.remaining_e-self.remaining_t*self.fly_e)/self.max_e)
        return np.array(state), r, r_, done