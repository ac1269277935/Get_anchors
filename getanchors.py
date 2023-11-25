'''
Created on Feb 20, 2017

@author: jumabek
'''
from os import listdir
from os.path import isfile, join
import numpy as np
import sys
import os
import shutil
import random
import math

# 计算交并比
def IOU(x, centroids):
    similarities = []
    k = len(centroids)
    for centroid in centroids:
        c_w, c_h = centroid
        w, h = x
        if c_w >= w and c_h >= h:
            similarity = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:
            similarity = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w * h / (w * h + c_w * (c_h - h))
        else:  # means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w * c_h) / (w * h)
        similarities.append(similarity)  # will become (k,) shape
    return np.array(similarities)

# 计算平均交并比
def avg_IOU(X, centroids):
    n, d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        # note IOU() will return array which contains IoU for each centroid and X[i] // slightly ineffective, but I am too lazy
        sum += max(IOU(X[i], centroids))
    return sum / n

def write_anchors_to_file(centroids, X, anchor_file, width_in_cfg_file, height_in_cfg_file):
    f = open(anchor_file, 'w')

    anchors = centroids.copy()
    print(anchors.shape)

    for i in range(anchors.shape[0]):
        anchors[i][0] *= width_in_cfg_file
        anchors[i][1] *= height_in_cfg_file

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    print('Anchors = ', anchors[sorted_indices])

    for i in sorted_indices[:-1]:
        f.write('%0.2f,%0.2f, ' % (anchors[i, 0], anchors[i, 1]))

    # there should not be comma after last anchor, that's why
    f.write('%0.2f,%0.2f\n' % (anchors[sorted_indices[-1:], 0], anchors[sorted_indices[-1:], 1]))

    f.write('%f\n' % (avg_IOU(X, centroids)))
    print()

# kmeans聚类算法
def kmeans(X, centroids, eps, anchor_file, width_in_cfg_file, height_in_cfg_file):

    # N是真实框的个数
    N = X.shape[0]
    iterations = 0
    # 中心点的个数和维数，k个dim维的中心点
    k, dim = centroids.shape

    # prev_assignments是N个-1
    prev_assignments = np.ones(N) * (-1)
    iter = 0

    # old_D是N个k维的全0的array
    old_D = np.zeros((N, k))

    while True:
        D = []
        iter += 1

        # 遍历每个真实框
        for i in range(N):

            # 1-IOU是loss，越小越好
            # 每个真实框都要和所有中心点计算交并比，每次得到6个数
            d = 1 - IOU(X[i], centroids)

            # 把当前loss加入d
            D.append(d)
        D = np.array(D)  # D.shape = (N,k)

        print("iter {}: dists = {}".format(iter, np.sum(np.abs(old_D - D))))

        # assign samples to centroids
        # assignments是D每行最小的元素的下标
        assignments = np.argmin(D, axis=1)

        # .all() 方法返回 True 如果输入数组的所有元素都是 True，否则为 False
        # 如果当前结果和之前结果全部相等
        if (assignments == prev_assignments).all():
            print("Centroids = ", centroids)

            # 把anchor存到文件里，并退出
            write_anchors_to_file(centroids, X, anchor_file, width_in_cfg_file, height_in_cfg_file)
            return

        # calculate new centroids
        # 创建一个全零数组，形状为[6,2]
        centroid_sums = np.zeros((k, dim), np.float64)

        # N是真实框的个数，遍历每个真实框
        for i in range(N):
            # centroid_sums求中心点坐标总和
            # centroid_sums[assignments[i]]是第i个真实框对应的中心点坐标，与真实框坐标相加
            centroid_sums[assignments[i]] += X[i]

        # k是中心点的个数
        # 更新每个中心点的坐标，更新后的坐标是当前中心点积及对应的总和/个数
        for j in range(k):
            centroids[j] = centroid_sums[j] / (np.sum(assignments == j))

        # 使用copy复制一份，保存到新的内存空间，并改名为prev_assignments用于对比
        prev_assignments = assignments.copy()
        old_D = D.copy()

def main():

    num_clusters = 6
    f = open("dataset/train.txt")

    # lines是几行字，每行是一个图片的地址，line.rstrip('\n')用来去除每行最后的换行符
    lines = [line.rstrip('\n') for line in f.readlines()]

    annotation_dims = []

    size = np.zeros((1, 1, 3))

    # 遍历每张图像
    for line in lines:

        # 把图片地址修改为标签地址
        line = line.replace('JPEGImages', 'labels')

        # 把地址中的jpg修改为txt
        line = line.replace('.jpg', '.txt')

        # 把地址中的png修改为txt
        line = line.replace('.png', '.txt')

        # 打开图片对应的txt文件地址
        f2 = open(line)

        # 文件地址每一行是一组真实框
        for line in f2.readlines():

            # 去掉行末尾的换行符
            line = line.rstrip('\n')

            # 记录宽高(类别,x,y,w,h)的w,h
            w, h = line.split(' ')[3:]
            # print(w,h)

            # annotation_dims会增加一个包含两个浮点数的元组
            # 把w、h转换为float，再转换为元组，加入annotation_dims中
            # 它的shape[0]是真实框的个数
            annotation_dims.append(tuple(map(float, (w, h))))

    annotation_dims = np.array(annotation_dims)

    eps = 0.005
    # width_in_cfg_file = args.input_width
    # height_in_cfg_file = args.input_height
    # 输入网络的图片宽高
    width_in_cfg_file = 56
    height_in_cfg_file = 56

    # 生成一个新的文件，里面保存了生成的anchor大小
    anchor_file = join('./', 'anchors%d.txt' % (num_clusters))

    # 常用于 k-means 聚类算法的初始聚类中心的选择
    # 生成 num_clusters 个随机索引
    # 在所有的真实框里面选出num_clusters个随机的作为初始聚类中心，indices是num_clusters个元素的列表，保存了随机作为中心的索引
    indices = [random.randrange(annotation_dims.shape[0]) for i in range(num_clusters)]

    centroids = annotation_dims[indices]

    # 调用kmeans函数，传入annotation_dims(所有真实框)，随机选的聚类中心，保存文件，图像宽、高
    kmeans(annotation_dims, centroids, eps, anchor_file, width_in_cfg_file, height_in_cfg_file)

    print('centroids.shape', centroids.shape)

if __name__ == "__main__":
    main()