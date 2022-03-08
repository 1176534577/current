import numpy

# import os
# print(os.getcwd())
# print(os.path.dirname(__file__))
# print(os.getcwd()+'\..\..\fortran\before_model.txt')
# print(os.path.abspath(os.getcwd()+r'\..\..\fortran\before_model.txt'))
import pymysql


def myreadxyz(abspath):
    # den = []
    # lenden = 0
    # with open(abspath+r'\beforeee_model', 'r') as f:
    #     for line in f.readlines():
    #         rou = float(line.strip())
    #         den.append(rou)
    #         # if rou != 0:
    #         #     lenden += 1
    # dict = {}
    # data = [[] for _ in range(lenden)]
    # a = set()
    # with open(abspath+r'\jxyz1', 'r', ) as f:
    #     for line in f.readlines():
    #         jxyz = line.strip().split()
    #         j = int(jxyz[0])
    #         if j in a:
    #             continue
    #         key = j
    #         value = [int(i) for i in jxyz[1:]]
    #         # value.append(den[j - 1])
    #         value.append(den[mydict[j - 1]])
    #         dict[key] = value
    #         a.add(j)

    # with open(abspath + r'\need_value1', 'r') as f:
    #     f.readline()
    #     nxyz = f.readline().strip().split()
    #     nx, ny, nz = int(nxyz[0]), int(nxyz[1]), int(nxyz[2])
    nx, ny, nz = 185, 88, 26
    p = numpy.zeros((nx, ny, nz))
    with open(abspath + r'\beforeee.txt', 'r') as f:
        for val_line in f.readlines():
            val = val_line.strip().split()
            x = int(val[0]) - 1
            y = int(val[1]) - 1
            z = int(val[2]) - 1
            # if 0 <= x < sx and 0 <= y < sy and 0 <= z < sz:
            try:
                p[x][y][z] = float(val[3])
            except IndexError:
                print(val)
    with open(abspath + r'\model', 'w') as f:
        for yy in range(ny):
            for xx in range(nx):
                for zz in range(nz)[::-1]:
                    f.write(str(p[xx][yy][zz]) + '\n')
    print('写入完成')


if __name__ == '__main__':
    abspath = r'D:\Projects\fortran\python'
    myreadxyz(abspath)
