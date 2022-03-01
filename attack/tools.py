import asyncio
import logging
import os
import time
import warnings
from multiprocessing import Pool, cpu_count

import pymysql
from numpy.linalg import matrix_rank

# from attack.setting import big_cell_list, big_cell_dict
from attack.smallsolver import solver_small


# logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
#                     filename=r'..\log\new.log',
#                     filemode='a',  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
#                     # a是追加模式，默认如果不写的话，就是追加模式
#                     format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
#                     # 日志格式
#                     )


# logging.info("开始计算某一块")
def getmylist():
    a = []
    db = pymysql.connect(user='root', password='123456', database='ijg')
    cs = db.cursor()
    sql = 'select j from jxyz'
    cs.execute(sql)
    for i in cs.fetchall():
        a.append(i)
    cs.close()
    db.close()

    # yz = set()
    # mydict = {}
    # with open(abspath + r'\jxyz', 'r') as jxyz:
    #     for jxyz_line in jxyz.readlines():
    #         yz.add(int(jxyz_line.strip().split()[0]))
    # a = list(yz)
    # a.sort()
    # for i in range(len(yz)):
    #     mydict[a[i]] = i  # todo 用list代替dict是否更好
    return a


# def raydict_cross_i(abspath: str, cs, j: int) -> list:
#     """
#     穿过第j个大格子的真实长度G的集合
#
#     :param abspath: 文件的绝对路径
#     :param j: 第j个大格子
#     :return: 真实长度的集合
#     """
#     # ray_dict = {}
#
#     ray_list = []
#     with open(abspath + r"\ij", 'r') as f, open(abspath + r"\G", 'r') as g:
#         for ij_line, G_line in zip(f.readlines(), g.readlines()):
#             ray_locate = ij_line.strip().split()
#             ray, locate = int(ray_locate[0]), int(ray_locate[1])
#             if locate == j:
#                 ray_list.append(float(G_line.strip()))
#     return ray_list


def ij_G(cs, list_j, add: list):
    """
    得到矩阵Ax=b中的A

    :param add: 误差列
    :param list_j: j的集合
    :return:
    """
    ray_sum = []
    sql = 'select i,j,g from ijg1 where j in %s'
    cs.execute(sql, (list_j,))
    for ijg in cs.fetchall():
        ray_sum.append(ijg)
    # with open(abspath + r"\ij1", 'r') as ij, open(abspath + r"\G1", 'r') as g:
    #     for ij_line, G_line in zip(ij.readlines(), g.readlines()):
    #         ray_locate = ij_line.strip().split()
    #         ray, locate = int(ray_locate[0]), int(ray_locate[1])
    #         if locate in set_j:
    #             G = float(G_line.strip())
    #             ray_sum.append([ray, locate, G])
    rayset = set()
    # locateset = set()
    # raydict = dict()
    raylist = []
    # locatedict = {}
    # rayi = 0
    # locatei = 0
    for ray in ray_sum:
        if ray[0] not in rayset:
            # raydict[ray[0]] = rayi
            raylist.append(ray[0])
            rayset.add(ray[0])
            # rayi += 1
        # if ray[1] not in locateset:
        #     locatedict[ray[1]] = locatei
        #     locateset.add(ray[1])
        #     locatei += 1
    # list_j = list(list_j)
    # list_j.sort()  # todo 这里去掉应该也可以,now,不可以,哎，好像又可以
    # for seq, j in enumerate(list_j):
    #     locatedict[j] = seq
    print(f'穿过大格子包含的小格子的射线总数：{len(rayset)}')
    g = [[0 for _ in range(len(list_j) + len(add))] for _ in range(len(add))]

    for mat in ray_sum:
        # print(mat[0:2])
        g[raylist.index(mat[0])][list_j.index(mat[1])] = mat[2]
    for i in range(len(add)):
        g[i][len(list_j) + i] = add[i][0]
    return g


# def get_cell_dict(abspath: str) -> dict:
#     """
#     得到大格子与化简后的值一一对应的字典
# 
#     :param abspath: 绝对路径
#     :return: 对应的字典
#     """
#     warnings.warn("有其他可代替的函数实现此功能", DeprecationWarning)
#     yz = set()
#     cell_dict = {}
#     with open(abspath + r'\ij', 'r') as ij:
#         for ij_line in ij.readlines():
#             yz.add(int(ij_line.strip().split()[1]))
#     for i in range(len(yz)):
#         cell_dict[list(yz)[i]] = i  # todo 修改成list是否更为有利
#     return cell_dict


# 只执行一次
def get_density(abspath: str):
    """
    得到第i个大格子的密度,i大于1

    :param abspath: 文件的绝对路径
    :param i: 第i个大格子
    :return: 大格子的密度
    """
    den = []

    with open(abspath + r'\before_model', 'r') as f:
        for line in f.readlines():
            density = float(line.strip())
            den.append(density)
    # todo den.append(0)是否多余
    # den.append(0)
    return den


# def ray_start_xyz(i: int, nx: float, ny: float, nz: float):
#     """
#     经过大格子的射线的起点坐标
# 
#     :param i: 第i个格子
#     :param nx: x轴分割的间隔
#     :param ny: y轴分割的间隔
#     :param nz: z轴分割的间隔
#     :return:
#     """
#     # todo
#     # 射线所属的探测器的起始坐标点，假设为x,y,z
#     x, y, z = 1, 1, 1
#     q, p = 2, 2
#     # 求出格子所在的确定坐标，假设为x1,y1,z1
#     x1, y1, z1 = 2, 2, 2


# def getG(abspath: str):
#     """
#     得到射线经过每一个格子的真实长度
# 
#     :param abspath: 绝对路径
#     :return:
#     """
#     # todo


# 只执行一次
# def getcoord(abspath: str):
#     """
#     大格子的坐标
#
#     :param abspath: 结对路径
#     :return: 大格子位置与坐标的字典
#     """
#     dict = {}
#     # a = set()
#     with open(abspath + r'\jxyz', 'r', ) as f:
#         for line in f.readlines():
#             jxyz = line.strip().split()
#             j = int(jxyz[0])
#             # if j in a:
#             #     continue
#             # key = j  # todo 去掉key
#             value = [int(i) for i in jxyz[1:]]
#             dict[j] = value
#             # a.add(j)
#     return dict


def get_small_cell(cs, scale: int, x, y, z):
    """
    大块中包含的小块，返回小块的位置和x,y,z坐标

    :param scale: 大块与小块的缩放倍数
    :return: 集合
    """
    # x, y, z = need_dict[big_cell]
    jxyz_dict = {}
    nx, ny, xz = 93, 44, 13  # todo 需要改
    print('大格子的坐标：', x, y, z)
    # 确定j的范围
    sxmin = scale * (x - 1) + 1
    sxmax = scale * x
    symin = scale * (y - 1) + 1
    symax = scale * y
    szmin = scale * (z - 1) + 1
    szmax = scale * z
    # min = sxmin + (symin - 1) * nx + (szmin - 1) * nx * ny  # sxmin是(sxmin-1)+1得来的
    # max = sxmax + (symax - 1) * nx + (szmax - 1) * nx * ny  # sxmax是(sxmax-1)+1得来的
    j_list = []
    for x in range(sxmin, sxmax + 1):
        for y in range(symin, symax + 1):
            for z in range(szmin, szmax + 1):
                j = x + (y - 1) * nx + (z - 1) * nx * ny
                j_list.append(j)

    sql = 'select * from jxyz1 where j in %s'
    cs.execute(sql, (j_list,))
    for i in cs.fetchall():
        jxyz_dict[i[0]] = i[1:]

    # a = set()
    # with open(abspath + r'\jxyz1', 'r', ) as f:
    #     for line in f.readlines():
    #         jxyz1 = line.strip().split()
    #         j = int(jxyz1[0])
    #         if j not in a:
    #             sx, sy, sz = [int(i) for i in jxyz1[1:]]
    #             if scale * (x - 1) < sx <= scale * x and scale * (y - 1) < sy <= scale * y and scale * (
    #                     z - 1) < sz <= scale * z:
    #                 a.add(j)
    #                 jxyz_dict[j] = [sx, sy, sz]
    return jxyz_dict


def ensure_solvable(a, addlist):
    start = matrix_rank(a)
    print('初始秩：', start)
    for i, add in enumerate(addlist):
        a[i].append(add)
    end = matrix_rank(a)
    print('终止秩：', end)
    if start == end:
        result = '有解'
    else:
        result = '无解'
    return result


def ray_d(cs, big_cell, density):
    """
    射线在某个大格子中的等效长度

    :return:
    """
    # mylist = raydict_cross_i(abspath,cs, big_cell)
    mylist = []
    sql = 'select i from ijg where j=%s'
    cs.execute(sql, big_cell)
    for i in cs.fetchall():
        mylist.append([i[0] * density])

    print(f'穿过第{big_cell}个大格子的射线总数：{len(mylist)}')
    # cell_dict = big_cell_dict
    # todo len(big_density_list) - 1)是否多余
    # density = big_density_list[cell_dict.get(big_cell - 1, len(big_density_list) - 1)]
    # density = big_density_list[big_cell_list.index(big_cell)]
    # for ii in mydict.keys():
    #     mydict[ii] *= density
    # mydict2list = list(mydict.values())
    # todo 直接用字典.keys应该更好
    # return [[i] for i in mydict.values()]
    # return [[i * density] for i in mylist]
    return mylist


def judge_isneed(x, y):
    # try:
    #     x, y, z = need_dict[big_cell]
    # except Exception as e:
    #     logging.exception(e)
    # x, y, z = need_dict[big_cell]
    if 19 < x <= 26:
        return True
    return False


def queryxyz(cs, data):
    sql = "select x,y,z from jxyz where j=%s"
    cs.execute(sql, data)
    x, y, z = cs.fetchone()

    return x, y, z


def queryp(cs, data):
    sql = 'select p from jxyz where j=%s'
    cs.execute(sql, data)
    return cs.fetchone()[0]


async def ryjmain(abspath, big_cell, scale):
    # print(f'我是{os.getppid()}的儿子{os.getpid()},数字是:{big_cell}')
    db = pymysql.connect(user='root', password='123456', database='ijg')
    cs = db.cursor()
    x, y, z = queryxyz(cs, big_cell)
    isneed = judge_isneed(x, y)
    # 大格子包括的小格子
    jxyzdict = get_small_cell(cs, scale, x, y, z)  # todo 分别写到if else里是否更合适，else只需要函数返回的第二项
    p = queryp(cs, big_cell)
    if isneed:
        # print('大格子中包含的小格子有：', cell_set)
        # print('111')
        # mydict2list = list(mydict.values())
        mydict2list = ray_d(cs, big_cell, p)
        # print('222')
        # 得到矩阵A
        iijj = ij_G(cs, list(jxyzdict.keys()), mydict2list)
        row = len(iijj)
        col = len(iijj[0])
        # print('行列：', row, col)
        # res = ensure_solvable(iijj, mydict2list[:3000])
        # print(res)
        await solver_small(abspath, row, col, iijj, mydict2list[:3000], jxyzdict)
    else:
        # d = big_density_list[big_cell_list.index(big_cell)]
        with open(abspath + r'\beforeee_model', 'a') as f:
            for v in jxyzdict.values():
                x, y, z = v
                f.write(f'{x} {y} {z} {p}\n')
    cs.close()
    db.close()


def asy(abspath, big_cell_s, scale):
    # 协程
    loop = asyncio.get_event_loop()
    tasks = [ryjmain(abspath, big_cell, scale) for big_cell in big_cell_s]
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()


if __name__ == '__main__':
    st = time.time()
    scale = 2
    abspath = r'D:\Projects\fortran\python'
    print(f'我是他们的爸爸，我的pid: {os.getpid()}')

    big_cell_list = getmylist()
    # big_cell_list = [int(i) for i in big_cell_dict.keys()]

    # big_density_list = get_density(abspath)
    # need_dict = getcoord(abspath)

    p = Pool(cpu_count())

    # a = big_cell_list
    big_cell_slist = []
    bb = []
    for index, aa in enumerate(big_cell_list):
        if index == 0 or index % 4 != 0:
            bb.append(aa)
        else:
            big_cell_slist.append(bb)
            bb = [aa]
        if index == len(big_cell_list) - 1:
            big_cell_slist.append(bb)

    # big_cell_list = [20, 21, 66,11,2,4,5]
    # 多线程
    for big_cell_s in big_cell_slist:
        p.apply_async(asy, args=(abspath, big_cell_s, scale))
    p.close()
    p.join()
    fin = time.time()
    print(f'间隔:{fin - st}')
