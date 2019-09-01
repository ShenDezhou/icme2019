#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2019年9月1日
create unique id of user and author

@author: Administrator
'''
import pandas
import numpy
from boltons.setutils import IndexedSet

track1 = pandas.read_csv('../input/final_track1_train.txt', sep='\t', iterator=True)
print(track1)

idset = []
Loop = True
while Loop:
    try:
        track = track1.get_chunk(1e8).values
        track = track[:, [0, 3, 6]]
        track = track[track[:, 2] > 0]
        print(track)
        
        idset.append(numpy.unique(track[:, [0, 1]], axis=0))

    except StopIteration:
        Loop = False
        print('stop iteration')

ids = numpy.unique(numpy.concatenate(idset), axis=0)       

track2 = pandas.read_csv('../input/final_track2_train.txt', sep='\t').values
track2 = track2[:, [0, 3, 6]]
track2 = track2[track2[:, 2] > 0]
print(track2)
ids = numpy.unique(numpy.concatenate((numpy.unique(track2[:, [0, 1]], axis=0), ids)), axis=0)
numpy.savez_compressed('../numpy/finish.npz', finish=ids)
print('done')

