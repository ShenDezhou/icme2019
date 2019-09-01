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
        track = track[:, [0, 3]]
        print(track)
        
        idset.append(numpy.unique(track.flatten()))
        
    except StopIteration:
        Loop = False
        print('stop iteration')

ids = numpy.unique(numpy.concatenate(idset))       
print(ids)

track2 = pandas.read_csv('../input/final_track2_train.txt', sep='\t').values
track2 = track2[:, [0, 3]]
print(track2)
ids = numpy.unique(numpy.concatenate((track2.flatten(), ids)))
numpy.savez_compressed('../numpy/id.npz', id=ids)
print('done')

