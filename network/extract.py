#!/usr/bin/python
# -*- coding: utf-8 -*-
#14101338
import numpy
import pandas
#
if False:
    np=numpy.load("../numpy/id.npz")
    print(np["id"].shape)
    print(np["id"])
    pd = pandas.DataFrame(np["id"])
    pd.to_csv("../numpy/id.txt")

if False:
    np=numpy.load("../numpy/finish.npz")
    np=numpy.save("../numpy/finish.npy", arr=np['finish'])


np=numpy.load("../numpy/like.npz")
np=numpy.save("../numpy/like.npy", arr=np['like'])
