import logging
import time
from math import pi, cos, sin, tan, ceil, floor


def getG(abspath: str):
    """
    得到射线经过每一个格子的真实长度

    :param abspath: 绝对路径
    :return:
    """
    # with open(abspath + r'\14_obs.dat', 'r') as f, open(abspath + r'\python\ij', 'w') as pij, open(
    #         abspath + r'\python\need_value', 'w') as pnv, open(abspath + r'\python\G', 'w') as pg, open(
    #     abspath + r'\python\d', 'w') as pd, open(abspath + r'\python\jxyz', 'w') as pjxyz,open(abspath+r'\python\mesh','w') as pm:
    with open(abspath + r'\14_obs.dat', 'r') as f, open(abspath + r'\python\ij1', 'w') as pij, open(
            abspath + r'\python\need_value1', 'w') as pnv, open(abspath + r'\python\G1', 'w') as pg, open(
        abspath + r'\python\d1', 'w') as pd, open(abspath + r'\python\jxyz1', 'w') as pjxyz, open(
        abspath + r'\python\mesh1', 'w') as pm:
        ns, delx, dely, delz = [float(i) for i in f.readline().strip().split()[:4]]
        dxmax, dxmin, dymax, dymin, dzmax, dzmin = [float(i) for i in f.readline().strip().split()]
        wx, wy, wz, wsm = [float(i) for i in f.readline().strip().split()]
        sx, sy, sz, nray = [0 for _ in range(int(ns))], [0 for _ in range(int(ns))], [0 for _ in range(int(ns))], [0 for
                                                                                                                   _ in
                                                                                                                   range(
                                                                                                                       int(ns))]
        for i in range(int(ns)):
            sx[i], sy[i], sz[i], nray[i] = [float(i) for i in f.readline().strip().split()]
        detid = [0 for _ in range(int(ns))]
        theta = [0 for _ in range(100000)]
        phi = [0 for _ in range(100000)]
        alen = [0 for _ in range(100000)]
        # d = [0 for _ in range(100000)]
        rx = [0 for _ in range(100000)]
        ry = [0 for _ in range(100000)]
        rz = [0 for _ in range(100000)]
        size = 2500000
        # bigsize = 5000000
        # h = [0 for _ in range(size)]
        xx = [0 for _ in range(size)]
        yy = [0 for _ in range(size)]
        zz = [0 for _ in range(size)]
        g = [0 for _ in range(size)]
        # irow = [0 for _ in range(bigsize)]
        # icol = [0 for _ in range(bigsize)]
        # a = [0 for _ in range(bigsize)]
        xmin = ymin = zmin = 100000
        xmax = ymax = zmax = -100000
        irec = [0 for _ in range(int(ns))]
        k = 0  # 符合条件的射线总数
        for i in range(int(ns)):
            for j in range(int(nray[i])):
                detid[i], a2, a3, a4, theta[k], phi[k], alen[k], extlmeas, er = [float(i) for i in
                                                                                 f.readline().strip().split()]
                if er < 0.001:
                    continue
                if phi[k] < 0:
                    phi[k] += 2 * pi
                # d[k] = extlmeas
                rx[k] = sx[i] + alen[k] * sin(theta[k]) * cos(phi[k])
                ry[k] = sy[i] + alen[k] + sin(theta[k]) * sin(phi[k])
                rz[k] = sz[i] + alen[k] * cos(theta[k])
                # if rx[k] > dxmax or rx[k] < dxmin or ry[k] > dymax or ry[k] < dymin or rz[k] > dzmax or rz[k] < dzmin or alen[k] < 0.1:
                if alen[k] < 0.1:
                    continue
                # if rx[k] > xmax:
                #     xmax = rx[k]
                # if rx[k] < xmin:
                #     xmin = rx[k]
                # if ry[k] > ymax:
                #     ymax = ry[k]
                # if ry[k] < ymin:
                #     ymin = ry[k]
                # if rz[k] > zmax:
                #     zmax = rz[k]
                # if rz[k] < zmin:
                #     zmin = rz[k]
                irec[i] += 1
                pd.write(f'{extlmeas} {er}\n')
                k = k + 1
            # pnv.write(f"第{i}个探测器满足的射线条数：{irec[i]}\n")
        print(k)
        ny = ceil((dymax - dymin) / dely)
        nx = ceil((dxmax - dxmin) / delx)
        nz = ceil((dzmax - dzmin) / delz)
        m = nx * ny * nz
        pnv.write(f'{k} {m}\n')
        pnv.write(f'{nx} {ny} {nz}\n')
        # pnv.write(f"总共符合条件的射线数：{k}\n")
        pm.write(f'{nx} {ny} {nz}\n')
        pm.write(f'{dxmin} {dymin} {dzmax}\n')
        pm.write(f'{nx}*{delx}\n')
        pm.write(f'{ny}*{dely}\n')
        pm.write(f'{nz}*{delz}\n')
        nel = 0
        if zmax > dzmax:
            print('ERROR: zmax >  dzmax ')
            exit(1)
        xmin = dxmin
        ymin = dymin
        zmin = dzmin
        nxy = nx * ny
        i = 0
        aset = set()
        for k in range(int(ns)):
            for ll in range(irec[k]):
                # if i == 0:
                #     pass
                tot = 0
                izst = floor((sz[k] - zmin) / delz + 1)  # math.floor向下取整
                izfin = floor((rz[i] - zmin) / delz + 1)
                # izst = ceil((sz[k] - zmin) / delz)  # math.ceil向上取整
                # izfin = ceil((rz[i] - zmin) / delz)
                for kz in range(izst, izfin + 1):
                    zdiff = delz
                    if kz == izst:
                        zdiff = kz * delz - sz[k] + zmin
                        t0 = zdiff / cos(theta[i])
                    if kz == izfin:
                        zdiff = rz[i] - (kz - 1) * delz - zmin
                    if izst == izfin:
                        zdiff = rz[i] - sz[k]  # todo 自己加的
                    if kz == izst:
                        tlen = 0
                    else:
                        tlen = (kz - izst - 1) * delz / cos(theta[i]) + t0
                    xst = tlen * sin(theta[i]) * cos(phi[i]) + sx[k]
                    yst = tlen * sin(theta[i]) * sin(phi[i]) + sy[k]
                    tlen = zdiff / cos(theta[i])
                    xnd = tlen * sin(theta[i]) * cos(phi[i]) + xst
                    # ynd = tlen * sin(theta[i]) * sin(phi[i]) + yst
                    ixst = floor((xst - xmin) / delx + 1)
                    # if ixst == 38:
                    #     print('你奶奶的')
                    ixfin = floor((xnd - xmin) / delx + 1)
                    # iyst = floor((yst - ymin) / dely + 1)
                    # iyfin = floor((ynd - ymin) / dely + 1)
                    # ixint = 1
                    # negx = False
                    if ixst > ixfin:
                        # negx = True
                        ixint = -1
                        yr = yst - ymin
                        for kk in range(ixst, ixfin - 1, ixint):
                            iyr = floor(yr / dely)
                            yl = yr - tan(phi[i]) * delx
                            if kk == ixst:
                                # yl = yr - tan(phi[i]) * (xst - xmin - (ixst - 1) * delx)
                                yl = yr - tan(phi[i]) * (xst - xmin - (kk - 1) * delx)
                            if kk == ixfin:
                                # yl = yr - tan(phi[i]) * (ixfin * delx - xnd + xmin)
                                yl = yr - tan(phi[i]) * (kk * delx - xnd + xmin)
                            if ixst == ixfin:
                                yl = yr - tan(phi[i]) * (xst - xnd)
                            iyl = floor(yl / dely)
                            idiff = abs(iyr - iyl) - 1
                            if idiff < 0:
                                if abs(phi[i] - pi) < 0.1e-6:
                                    dl = delx
                                    if kk == ixst:
                                        dl = xst - xmin - (ixst - 1) * delx
                                    if kk == ixfin:
                                        dl = ixfin * delx - xnd + xmin
                                    if ixst == ixfin:
                                        dl = xst - xnd
                                else:
                                    dl = (yr - yl) / sin(phi[i])
                                j = iyl * nx + kk - 1 + (kz - 1) * nxy
                                # h[j] = h[j] + 1
                                xx[j] = kk - 1
                                yy[j] = iyl
                                zz[j] = kz - 1
                                g[j] = abs(dl) / sin(theta[i])
                                tot = tot + abs(dl) / sin(theta[i])
                            elif idiff == 0:
                                if phi[i] < pi:
                                    dl = ((iyr + 1) * dely - yr) / sin(phi[i])
                                    j = iyr * nx + kk - 1 + (kz - 1) * nxy
                                    # h[j] = h[j] + 1
                                    xx[j] = kk - 1
                                    yy[j] = iyr
                                    zz[j] = kz - 1
                                    g[j] = abs(dl) / sin(theta[i])
                                    tot = tot + abs(dl) / sin(theta[i])
                                    dl = (yl - iyl * dely) / sin(phi[i])
                                    jj = iyl * nx + kk - 1 + (kz - 1) * nxy
                                    # h[jj] = h[jj] + 1
                                    xx[jj] = kk - 1
                                    yy[jj] = iyl
                                    zz[jj] = kz - 1
                                    g[jj] = abs(dl) / sin(theta[i])
                                    tot = tot + abs(dl) / sin(theta[i])
                                else:
                                    dl = (yr - iyr * dely) / sin(phi[i])
                                    j = iyr * nx + kk - 1 + (kz - 1) * nxy
                                    # h[j] = h[j] + 1
                                    xx[j] = kk - 1
                                    yy[j] = iyr
                                    zz[j] = kz - 1
                                    g[j] = abs(dl) / sin(theta[i])
                                    tot = tot + abs(dl) / sin(theta[i])
                                    dl = ((iyl + 1) * dely - yl) / sin(phi[i])
                                    jj = iyl * nx + kk - 1 + (kz - 1) * nxy
                                    # h[jj] = h[jj] + 1
                                    xx[jj] = kk - 1
                                    yy[jj] = iyl
                                    zz[jj] = kz - 1
                                    g[jj] = abs(dl) / sin(theta[i])
                                    tot = tot + abs(dl) / sin(theta[i])
                            else:
                                if phi[i] < pi:
                                    dl = ((iyr + 1) * dely - yr) / sin(phi[i])
                                    j = iyr * nx + kk - 1 + (kz - 1) * nxy
                                    # h[j] = h[j] + 1
                                    xx[j] = kk - 1
                                    yy[j] = iyr
                                    zz[j] = kz - 1
                                    g[j] = abs(dl) / sin(theta[i])
                                    tot = tot + abs(dl) / sin(theta[i])
                                    dl = dely / sin(phi[i])
                                    for intt in range(1, idiff + 1):
                                        jj = (iyr + intt) * nx + kk - 1 + (kz - 1) * nxy
                                        # h[jj] = h[jj] + 1
                                        xx[jj] = kk - 1
                                        yy[jj] = iyr + intt
                                        zz[jj] = kz - 1
                                        g[jj] = abs(dl) / sin(theta[i])
                                        tot = tot + abs(dl) / sin(theta[i])
                                    dl = (yl - iyl * dely) / sin(phi[i])
                                    jjj = iyl * nx + kk - 1 + (kz - 1) * nxy
                                    # h[jjj] = h[jjj] + 1

                                    xx[jjj] = kk - 1
                                    yy[jjj] = iyl
                                    zz[jjj] = kz - 1
                                    g[jjj] = abs(dl) / sin(theta[i])
                                    tot = tot + abs(dl) / sin(theta[i])
                                else:
                                    dl = (yr - iyr * dely) / sin(phi[i])
                                    j = iyr * nx + kk - 1 + (kz - 1) * nxy
                                    # h[j] = h[j] + 1
                                    xx[j] = kk - 1
                                    yy[j] = iyr
                                    zz[j] = kz - 1
                                    g[j] = abs(dl) / sin(theta[i])
                                    tot = tot + abs(dl) / sin(theta[i])
                                    dl = dely / sin(phi[i])
                                    for intt in range(1, idiff + 1):
                                        jj = (iyr - intt) * nx + kk - 1 + (kz - 1) * nxy
                                        # h[jj] = h[jj] + 1
                                        xx[jj] = kk - 1
                                        yy[jj] = iyr - intt
                                        zz[jj] = kz - 1
                                        g[jj] = abs(dl) / sin(theta[i])
                                        tot = tot + abs(dl) / sin(theta[i])
                                    dl = ((iyl + 1) * dely - yl) / sin(phi[i])
                                    jjj = iyl * nx + kk - 1 + (kz - 1) * nxy
                                    # h[jjj] = h[jjj] + 1

                                    xx[jjj] = kk - 1
                                    yy[jjj] = iyl
                                    zz[jjj] = kz - 1
                                    g[jjj] = abs(dl) / sin(theta[i])
                                    tot = tot + abs(dl) / sin(theta[i])
                            yr = yl
                    else:
                        yl = yst - ymin
                        for kk in range(ixst, ixfin + 1):
                            iyl = floor(yl / dely)
                            yr = yl + tan(phi[i]) * delx
                            if kk == ixst:
                                # yr = yl + tan(phi[i]) * (ixst * delx - xst + xmin)
                                yr = yl + tan(phi[i]) * (kk * delx - xst + xmin)
                            if kk == ixfin:
                                # yr = yl + tan(phi[i]) * (xnd - xmin - (ixfin - 1) * delx)
                                yr = yl + tan(phi[i]) * (xnd - xmin - (kk - 1) * delx)
                            if ixst == ixfin:
                                yr = yl + tan(phi[i]) * (xnd - xst)
                            iyr = floor(yr / dely)
                            idiff = abs(iyr - iyl) - 1
                            if idiff < 0:
                                if abs(phi[i]) < 0.1e-6:
                                    dl = delx
                                    if kk == ixst:
                                        # dl = ixst * delx - xst + xmin
                                        dl = kk * delx - xst + xmin
                                    if kk == ixfin:
                                        # dl = xnd - xmin - (ixfin - 1) * delx
                                        dl = xnd - xmin - (kk - 1) * delx
                                    if ixst == ixfin:
                                        dl = xnd - xst
                                else:
                                    dl = (yr - yl) / sin(phi[i])
                                j = iyl * nx + kk - 1 + (kz - 1) * nxy
                                # kz是从1开始的，kk是从1开始的，iyl是从0开始的
                                # zz第几层，yy第几行，xx第yy行的第几个
                                # h[j] += 1
                                xx[j] = kk - 1
                                yy[j] = iyl
                                zz[j] = kz - 1
                                g[j] = abs(dl) / sin(theta[i])
                                tot = tot + abs(dl) / sin(theta[i])
                            elif idiff == 0:
                                if phi[i] < pi:
                                    dl = ((iyl + 1) * dely - yl) / sin(phi[i])
                                    j = iyl * nx + kk - 1 + (kz - 1) * nxy
                                    # h[j] = h[j] + 1

                                    xx[j] = kk - 1
                                    yy[j] = iyl
                                    zz[j] = kz - 1
                                    g[j] = abs(dl) / sin(theta[i])
                                    tot = tot + abs(dl) / sin(theta[i])

                                    dl = (yr - iyr * dely) / sin(phi[i])
                                    jj = iyr * nx + kk - 1 + (kz - 1) * nxy
                                    # h[jj] = h[jj] + 1
                                    xx[jj] = kk - 1
                                    yy[jj] = iyr
                                    zz[jj] = kz - 1
                                    g[jj] = abs(dl) / sin(theta[i])
                                    tot = tot + abs(dl) / sin(theta[i])
                                else:
                                    dl = (yl - iyl * dely) / sin(phi[i])
                                    j = iyl * nx + kk - 1 + (kz - 1) * nxy
                                    # h[j] = h[j] + 1

                                    xx[j] = kk - 1
                                    yy[j] = iyl
                                    zz[j] = kz - 1
                                    g[j] = abs(dl) / sin(theta[i])
                                    tot = tot + abs(dl) / sin(theta[i])

                                    dl = ((iyr + 1) * dely - yr) / sin(phi[i])
                                    jj = iyr * nx + kk - 1 + (kz - 1) * nxy
                                    # h[jj] = h[jj] + 1
                                    xx[jj] = kk - 1
                                    yy[jj] = iyr
                                    zz[jj] = kz - 1
                                    g[jj] = abs(dl) / sin(theta[i])
                                    tot = tot + abs(dl) / sin(theta[i])
                            else:
                                if phi[i] < pi:
                                    dl = ((iyl + 1) * dely - yl) / sin(phi[i])
                                    j = iyl * nx + kk - 1 + (kz - 1) * nxy
                                    # h[j] = h[j] + 1
                                    xx[j] = kk - 1
                                    yy[j] = iyl
                                    zz[j] = kz - 1
                                    g[j] = abs(dl) / sin(theta[i])
                                    tot = tot + abs(dl) / sin(theta[i])
                                    dl = dely / sin(phi[i])
                                    for intt in range(1, idiff + 1):
                                        jj = (iyl + intt) * nx + kk - 1 + (kz - 1) * nxy
                                        # h[jj] = h[jj] + 1
                                        xx[jj] = kk - 1
                                        yy[jj] = iyl + intt
                                        zz[jj] = kz - 1
                                        g[jj] = abs(dl) / sin(theta[i])
                                        tot = tot + abs(dl) / sin(theta[i])
                                    dl = (yr - iyr * dely) / sin(phi[i])
                                    jjj = iyr * nx + kk - 1 + (kz - 1) * nxy
                                    # h[jjj] = h[jjj] + 1

                                    xx[jjj] = kk - 1
                                    yy[jjj] = iyr
                                    zz[jjj] = kz - 1
                                    g[jjj] = abs(dl) / sin(theta[i])
                                    tot = tot + abs(dl) / sin(theta[i])
                                else:
                                    dl = (yl - iyl * dely) / sin(phi[i])
                                    j = iyl * nx + kk - 1 + (kz - 1) * nxy
                                    # h[j] = h[j] + 1

                                    xx[j] = kk - 1
                                    yy[j] = iyl
                                    zz[j] = kz - 1
                                    g[j] = abs(dl) / sin(theta[i])
                                    tot = tot + abs(dl) / sin(theta[i])
                                    dl = dely / sin(phi[i])
                                    for intt in range(1, idiff + 1):
                                        jj = (iyl - intt) * nx + kk - 1 + (kz - 1) * nxy
                                        # h[jj] = h[jj] + 1
                                        xx[jj] = kk - 1
                                        yy[jj] = iyl - intt
                                        zz[jj] = kz - 1
                                        g[jj] = abs(dl) / sin(theta[i])
                                        tot = tot + abs(dl) / sin(theta[i])
                                    dl = ((iyr + 1) * dely - yr) / sin(phi[i])
                                    jjj = iyr * nx + kk - 1 + (kz - 1) * nxy
                                    # h[jjj] = h[jjj] + 1

                                    xx[jjj] = kk - 1
                                    yy[jjj] = iyr
                                    zz[jjj] = kz - 1
                                    g[jjj] = abs(dl) / sin(theta[i])
                                    tot = tot + abs(dl) / sin(theta[i])
                            yl = yr
                for j in range(m):
                    # try:
                    if g[j] > 0.0:
                        # irow[nel] = i
                        # icol[nel] = j
                        # a[nel] = g[j]
                        nel = nel + 1
                        pij.write(f'{i + 1} {j + 1}\n')
                        pg.write(f'{g[j]}\n')
                        if j not in aset:
                            aset.add(j)
                            pjxyz.write(f'{j + 1} {xx[j] + 1} {yy[j] + 1} {zz[j] + 1}\n')
                    g[j] = 0
                    # print(a[nel])
                    # except IndexError as e:
                    #     logging.exception(e)
                    #     print(j)
                    # print(nel)
                    # exit(2)
                print(f'射线:{i}')
                i += 1
            print(f'探测器:{k}已完成')
                # for num in range(size):
                #     g[num] = 0
        pnv.write(f"nel的大小为：{nel}\n")


if __name__ == '__main__':
    st = time.time()
    abspath = r'D:\Projects\fortran'
    getG(abspath)
    fin = time.time()
    print(f'间隔:{fin - st}')
