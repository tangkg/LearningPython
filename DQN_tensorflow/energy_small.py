import numpy as np 


def a_base_line(uav_p, final_p):  ###UAV根据当前位置和要到达位置选择动作
    if uav_p[0] > final_p[0]:  ###UAV位于sensor的右方
        action = 3
    elif uav_p[0] < final_p[0]:  ###UAV位于sensor的左方
        action = 4
    elif uav_p[1] > final_p[1]:  ###UAV位于sensor的上方
        action = 2
    elif uav_p[1] < final_p[1]:  ###UAV位于sensor的下方
        action = 1
    return action


def run():
    sensors = [[2, 4], [5, 8], [7, 3]]  ###sensor的位置
    sensor_n = 3
    start_p = [0, 0]  ###起点的位置
    stop_p = [10, 0]  ###终点的位置
    base_p = [10, 10]  ###基站的位置
    length = 11  ###飞行区域长度
    width = 11  ###飞行区域宽度
    fly_line = 2  ###UAV直飞一步的能量
    turn_90 = 9  ###UAV转弯90多花的能量
    turn_180 = 36  ###UAV转弯180多花的能量
    max_aoi = 200
    max_e = 400  ###UAV的总能量
    lamda = [1/3,1/3,1/3]  ###每个sensor的重要性权重

    for episode in range(1):
        bs_i = [1, 1, 1]  ##基站处的每个sensor的信息年龄
        remaining_e = max_e  ###UAV剩余的能量
        uav_p = start_p.copy()  ###UAV的起始位置
        uav_m =[]  ###用于存放UAV收集到的sensor的信息：包含信息年龄和属于哪个sensor
        sensor1_aoi = [1]  ###第一个sensor的信息年龄
        sensor2_aoi = [1]  ###第二个sensor的信息年龄
        sensor3_aoi = [1]  ###第三个sensor的信息年龄
        pre_a = [0]   ##记录UAV前一次的飞行方向
        r_aoi = 0 ###记录信息年龄
        r_e = 0 ###记录所耗的电量
        for step in range(max_aoi):
            if uav_p[0] == stop_p[0] or uav_p[1] == stop_p[1]:
                if (remaining_e - (abs(uav_p[0] - stop_p[0]) + abs(uav_p[1] - stop_p[1])) * fly_line) <= 2*turn_90:
                    action = a_base_line(uav_p, stop_p)
                else:
                    if len(uav_m) == 0: ###UAV选择往sensor的信息年龄最大的方向移动
                        sensor_index = np.argmax(np.array(bs_i))
                        action = a_base_line(uav_p,sensors[sensor_index])
                    else:
                        action = a_base_line(uav_p, base_p)
            else:
                if (remaining_e - (abs(uav_p[0] - stop_p[0]) +
                                   abs(uav_p[1] - stop_p[1])) * fly_line - turn_90) <= 2*turn_90:
                    action = a_base_line(uav_p, stop_p)
                else:
                    if len(uav_m) == 0: ###UAV选择往sensor的信息年龄最大的方向移动
                        sensor_index = np.argmax(np.array(bs_i))
                        action = a_base_line(uav_p,sensors[sensor_index])
                    else:
                        action = a_base_line(uav_p, base_p)
            ###UAV的移动
            if action == 1:
                if uav_p[1] < width-1:
                    uav_p[1] += 1
            elif action == 2:
                if uav_p[1] > 0:
                    uav_p[1] -= 1
            elif action == 3:
                if uav_p[0] > 0:
                    uav_p[0] -= 1
            else:
                if uav_p[0] < length-1:
                    uav_p[0] += 1

            ###计算UAV的飞行能耗
            if pre_a[0] == 0 or pre_a[0] == action:
                remaining_e -= fly_line
                r_e += fly_line
            else:
                remaining_e -= turn_90
                r_e += turn_90

            if len(uav_m) != 0:  ###若UAV的缓存中有sensor信息则将aoi值加1
                uav_m[0][0] += 1
            ###判断UAV是否在某个sensor的上方
            if uav_p in sensors:
                index = sensors.index(uav_p)
                if len(uav_m) == 0:   ###若此时UAV的缓存中没有sensor信息，则将此sensor的信息缓存到UAV中
                    uav_m.append([1, index])

            ###将基站处的每个sensor的信息年龄加一
            for i in range(sensor_n):
                bs_i[i] += 1
            if uav_p == base_p:  ###判断UAV是否到达基站处进行信息交付
                if len(uav_m) != 0:
                    if remaining_e >= (1.03*(10**(-3))*(100**2))/100:
                        bs_i[uav_m[0][1]] = uav_m[0][0]
                        uav_m = []  ###将UAV缓存的信息置为空
                        remaining_e -= (1.03*(10**(-3))*(100**2))/100
                        r_e += (1.03*(10**(-3))*(100**2))/100
            sensor1_aoi.append(bs_i[0])
            sensor2_aoi.append(bs_i[1])
            sensor3_aoi.append(bs_i[2])
            for i in range(sensor_n):
                r_aoi += lamda[i]*bs_i[i]

            if uav_p == stop_p:
                r = r_aoi*0.5 + r_e*0.5
                r /= step
                break
        print('r_aoi: ', r_aoi/step)
        print('r_e: ', r_e/step)
        print('0.5*r_aoi+0.5*r_e: ', r)
        print('sensor1_aoi: ', sensor1_aoi)
        print('sensor2_aoi: ', sensor2_aoi)
        print('sensor3_aoi: ', sensor3_aoi)
        print('remianing_e: ', remaining_e)


if __name__ == "__main__":
    run()


