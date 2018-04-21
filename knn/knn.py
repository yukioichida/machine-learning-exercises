#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import math

'''
    Euclidian Distance for two-dimensional data
'''


def euclidian_distance(p1, p2):
    x = pow(p1[0] - p2[0], 2)  # (x1 - x2) ^ 2
    y = pow(p1[1] - p2[1], 2)  # (y1 - y2) ^ 2
    return math.sqrt(x + y)


'''
    Read dataset using pandas
'''


def read_dataset():
    data_file = '../datasets/gafanhotos.csv'
    data_frame = pd.read_csv(data_file, names=['class', 'x', 'y'], delim_whitespace=True)
    return data_frame


'''
    k-Nearest Neighbors
'''


def knn(new_sample, k):
    data = read_dataset()
    # creates a new column with the distance between the line and the unseen data
    data['distance'] = data.apply(lambda row: euclidian_distance(new_sample, (row['x'], row['y'])), axis=1)
    # get k nearest data from data frame
    nearest_data = data.nsmallest(k, 'distance')
    # get the most frequent class in nearest data
    return nearest_data['class'].value_counts().idxmax()


if __name__ == "__main__":
    print knn((5, 2), 1)
    print knn((5, 2), 3)
