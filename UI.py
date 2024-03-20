# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : ${2023/10/27} ${12:28}
# @Author : Lizhan Hong, Helin Gong
# @Email : lzhong2048@sjtu.edu.cn
# @Software: ${RODT}
# @Lab: AISEA PLATFORM
# Input: the state or other data.
# Output: the UI product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as spr

# path names:
path_control = 'RawData/index.xlsx'
path_field_predicted_knn = 'Outputs/FieldPredictedKNN'
path_field_predicted_nn = 'Outputs/FieldPredictedNN'
path_field_predicted_dt = 'Outputs/FieldPredictedDT'

# save path:
path_plot3d_knn = 'Outputs/Plot3dKNN.png'
path_plot3d_nn = 'Outputs/Plot3dNN.png'
path_plot3d_dt = 'Outputs/Plot3dDT.png'


def ReconstructToOrigin(PowerRaw: np.ndarray, numGrid: int, numSection: int, numGridQuarter: int, lstGridControl: list,
                        rangeUtile, choice):
    '''
    Reconstruct the quatered data back to the origin.
    K_ul for upperleft;
    K_ur for upperright;
    K_ll for lowerleft;
    K_lr for lowerright

    :param PowerRaw: The raw data
    :param PowerRaw: The Power matrix we obtained.
    :param numGrid: The number of grids per section  (In our case, numGrid = 177).
    :param numSection: The number of vertical sections  (In our case, numSection = 28).
    :param numGridQuarter: The number of the grid_column per row in a quarter section (In our case, numGridQuarter = 52).
    :param lstGridControl: The list for all the way we place the grid in a quarter (In our case,
    lstGridControl = [8,8,8,7,7,6,5,3] )
    :param rangeUtile: The range of the chosen quarter section, where we can ignore the others by symetry.
    (In our case, rangeUtile = [[88,96],[103,111],[118,126],[132,139],[145,152],[157,163],[167,172],[174,177]])
    :return: The Power matrix simplified by symetry
    '''

    # axis_horizontal = [89, 90, 91, 92, 93, 94, 95]
    # axis_vertical = [103, 118, 132, 145, 157, 167, 174]
    index_ur = [74, 75, 76, 77, 78, 79, 80, 59, 60, 61, 62, 63, 64, 65, 45, 46, 47, 48, 49, 50, 32, 33,
                34, 35, 36, 37, 20, 21, 22, 23, 24, 10, 11, 12, 13, 3, 4]
    index_ll = [102, 101, 100, 99, 98, 97, 96, 117, 116, 115, 114, 113, 112, 111, 131, 130,
                129, 128, 127, 126, 144, 143, 142, 141, 140, 139, 156, 155, 154, 153, 152, 166,
                165, 164, 163, 173, 172]
    index_anti = [9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28,
                  29, 30, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43, 45, 46, 47, 48, 50, 51]

    def iter(PowerRaw):
        index_core_sym_raw = [j for range_tuple in rangeUtile for j in range(range_tuple[0], range_tuple[1])]
        index_core_sym = index_core_sym_raw.copy()
        index_core_sym.pop(0)

        def generSelectMat(data, row, col):
            '''
            Make the grid select matrix with size of (length of numGrid) * (numGrid)
            :return: sparce select matrix applied on one section.
            '''
            return spr.csc_matrix((data, (row, col)), shape=(numGrid, numGridQuarter)).todense()

        # ul
        def K_ul():
            row = []
            col = []
            for j in range(len(index_core_sym)):
                row.append(88 * 2 - index_core_sym[j])
                col.append(j + 1)
            if len(row) != len(col):
                raise ValueError('In the ul part, row do not fit the col')
            return generSelectMat(np.ones(len(row)), row, col)

        # ur
        def K_ur():
            if len(index_anti) != len(index_ur):
                raise ValueError('In the ur part, len(index_anti) != len(index_ur)')
            row = index_ur
            col = index_anti
            if len(row) != len(col):
                raise ValueError('In the ur part, row do not fit the col')
            return generSelectMat(np.ones(len(row)), row, col)

        # ll
        def K_ll():
            if len(index_anti) != len(index_ll):
                raise ValueError('In the ll part, len(index_anti) != len(index_ll)')
            row = index_ll
            col = index_anti
            if len(row) != len(col):
                raise ValueError('In the ll part, row do not fit the col')
            return generSelectMat(np.ones(len(row)), row, col)

        # lr
        def K_lr():
            row = index_core_sym_raw
            col = [k for k in range(len(index_core_sym_raw))]
            if len(row) != len(col):
                raise ValueError('In the lr part, row do not fit the col')
            return generSelectMat(np.ones(len(row)), row, col)

        # selection matrix for one submatrix(section)
        Kl = K_ul() + K_lr() + K_ll() + K_ur()
        if choice == '((numSection) * numGrid)':
            K = np.kron(Kl, np.eye(numSection))
        elif choice == '((numGrid) * numSection)':
            K = np.kron(np.eye(numSection), Kl)
        else:
            raise ValueError('Your choice is not in the valid set')

        return K @ PowerRaw.T

    return iter(PowerRaw)


def reactorGetSectionIndex(pathControl: str):
    '''
    Generate the index of the one section data in a grid square.
    :param pathControl: The path of our one section data.
    :return:
    '''
    data = pd.read_excel(pathControl, header=None)
    lst = []
    # data.dropna(axis=1, how='any', inplace=True)
    for i in range(len(data.iloc[0])):
        for j in range(len(data.iloc[1])):
            if data.iloc[i, j] > 0:
                lst.append(i * len(data.iloc[0]) + j)

    return lst


def reactorPlot3D(pathControl: str, path: str, numLength: int, numWidth: int, numGrid: int,
                  numSection: int):
    '''
    Plot the 3Dheatmap of the whole reactor.

    :param pathControl: The path of data we want to plot.
    :param path: The path of data we want to plot.
    :param nSample: The number of sample we want (ranging from 0 to 5516).
    :param numLength: The length of one section.
    :param numWidth: The width of one section.
    :param numGrid: The number of grids per section  (In our case, numGrid = 177).
    :param numSection: The number of vertical sections  (In our case, numSection = 28).
    :return:
    '''
    # The final data is in shape of (numGrid,numSection)
    data = np.loadtxt(path)
    data = np.array(data)
    lst = reactorGetSectionIndex(pathControl)

    def select():
        # first the righter loops(inner) then the lefter loops(outer)
        row = [i * numSection + k for i in lst for k in range(numSection)]
        col = np.arange(numGrid * numSection)
        data = np.ones(numGrid * numSection)
        return spr.csc_matrix((data, (row, col)),
                              shape=(numLength * numWidth * numSection, numGrid * numSection)).todense()

    maindata = select() @ data
    maindata = maindata.T

    hist = np.zeros((numLength, numWidth, numSection))
    for j in range(numGrid):
        if lst[j] % numWidth != 0:
            [*hist[(lst[j] // numWidth), (lst[j] % numWidth), :]] = maindata[
                                                                    lst[j] * numSection:(lst[j] + 1) * numSection]
        else:
            [*hist[(lst[j] // numWidth), (lst[j] % numWidth), :]] = maindata[
                                                                    lst[j] * numSection:(lst[j] + 1) * numSection]

    return hist


def GerDataRound2Square(data, nSection, choice):
    '''
    Generate the nSection^th section in one state of reactor.

    :param data:
    :param nSection:
    :param nSample:
    :return:
    '''
    numLength = 15
    numWidth = 15
    numGrid = 177
    numSection = 28

    # The final data is in shape of (numGrid,1)
    data = np.asarray(data)
    if np.ndim(data) == 1:
        # 5517
        if choice == '((numSection) * numGrid)':
            data = [data[nSection + i * numSection] for i in range(numGrid)]
        # 18480
        elif choice == '((numGrid) * numSection)':
            data = [data[nSection * numGrid + i] for i in range(numGrid)]
        else:
            raise ValueError('Your choice is not in the valid set')
    elif np.ndim(data) == 2:
        if choice == '((numSection) * numGrid)':
            data = [data[:, nSection + i * numSection] for i in range(numGrid)]
        elif choice == '((numGrid) * numSection)':
            data = [data[:, nSection * numGrid + i] for i in range(numGrid)]
        else:
            raise ValueError('Your choice is not in the valid set')
    else:
        raise ValueError('The data is not 1-dim nor 2-dim')
    data = np.array(data)
    lst = reactorGetSectionIndex(path_control)

    # generate the kernel projection matrix onto the square manner base.
    def select():
        row = lst
        col = np.arange(numGrid)
        data = np.ones(numGrid)
        return spr.csc_matrix((data, (row, col)), shape=(numLength * numWidth, numGrid)).todense()

    maindata = select() @ data
    maindata = maindata.reshape((numLength, numWidth))

    return maindata


def plot3D():
    # hist = GerDataRound2Square(data=np.loadtxt(path_field_predicted),
    #                            )
    hist = reactorPlot3D(path_control, path_field_predicted_knn, 15, 15, 177, 28)

    fig = plt.figure()
    ax = fig.add_subplot(221, projection='3d')

    # make the color matrix
    viridis = plt.cm.get_cmap('viridis', 256)
    colors1 = viridis(hist)
    # cmap1 = ax1.voxels(hist, facecolors=colors1, alpha=0.8)
    cmap1 = plt.cm.ScalarMappable(norm=None, cmap=viridis)
    cmap1.set_array(hist.flatten())
    ax.voxels(hist, facecolors=colors1, alpha=0.8)

    # Add colorbar
    fig.colorbar(cmap1, ax=ax, pad=0.15)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def TestdataIAEA(data_path: str, pathControl, pathSave):
    numLength = 15
    numWidth = 15
    numGrid = 177
    numSection = 28
    rangeUtile = [[88, 96], [103, 111], [118, 126], [132, 139], [145, 152], [157, 163], [167, 172], [174, 177]]

    def reactorPlot3D(pathControl: str, data_path: str, numLength: int, numWidth: int, numGrid: int,
                      numSection: int):
        '''
        Plot the 3Dheatmap of the whole reactor.

        :param pathControl: The path of data we want to plot.
        :param data_path: The path of data we want to plot.
        :param nSample: The number of sample we want (ranging from 0 to 5516).
        :param numLength: The length of one section.
        :param numWidth: The width of one section.
        :param numGrid: The number of grids per section  (In our case, numGrid = 177).
        :param numSection: The number of vertical sections  (In our case, numSection = 28).
        :return:
        '''
        # The final data is in shape of (numGrid,numSection)
        data = np.loadtxt(data_path)
        data = ReconstructToOrigin(data, numGrid=177, numSection=28, numGridQuarter=52,
                                lstGridControl=[8, 8, 8, 7, 7, 6, 5, 3],
                                rangeUtile=rangeUtile, choice='((numSection) * numGrid)')
        data = np.asarray(data).T
        lst = reactorGetSectionIndex(pathControl)
        print(lst)

        def select():
            # first the righter loops(inner) then the lefter loops(outer)
            row = [i * numSection + k for i in lst for k in range(numSection)]
            col = np.arange(numGrid * numSection)
            data = np.ones(numGrid * numSection)
            return spr.csc_matrix((data, (row, col)),
                                  shape=(numLength * numWidth * numSection, numGrid * numSection)).todense()

        maindata = select() @ data
        maindata = maindata.T

        hist = np.zeros((numLength, numWidth, numSection))
        for j in range(numGrid):
            if lst[j] % numWidth != 0:
                for i in range(numSection):
                    [hist[(lst[j] // numWidth), (lst[j] % numWidth), i]] = \
                        maindata[:, lst[j] * numSection + i]
            else:
                for i in range(numSection):
                    [hist[(lst[j] // numWidth), (lst[j] % numWidth), i]] = \
                        maindata[:, lst[j] * numSection + i]

        return hist

    hist = reactorPlot3D(pathControl, data_path, 15, 15, 177, 28)

    fig = plt.figure(figsize=(15, 7))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)
    plt.tight_layout()

    # ===============
    #  First subplot
    # ===============
    # set up the axes for the first plot
    ax1 = fig.add_subplot(111, projection='3d')

    # make the color matrix
    viridis = plt.cm.get_cmap('viridis', 256)
    colors1 = viridis(hist)
    # cmap1 = ax1.voxels(hist, facecolors=colors1, alpha=0.8)
    cmap1 = plt.cm.ScalarMappable(norm=None, cmap=viridis)
    cmap1.set_array(hist.flatten())
    ax1.voxels(hist, facecolors=colors1, alpha=0.8)

    # Add colorbar
    fig.colorbar(cmap1, ax=ax1, pad=0.15)

    # Set labels
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    plt.savefig(pathSave)
    plt.show()
    plt.close(fig=fig)


def plot_result(choice:str):
    if choice == 'knn':
        TestdataIAEA(data_path=path_field_predicted_knn, pathControl=path_control, pathSave=path_plot3d_knn)
    elif choice == 'nn':
        TestdataIAEA(data_path=path_field_predicted_nn, pathControl=path_control, pathSave=path_plot3d_nn)
    elif choice == 'dt':
        TestdataIAEA(data_path=path_field_predicted_dt, pathControl=path_control, pathSave=path_plot3d_dt)
    else:
        raise ValueError('Your choice is not in the available set!')
