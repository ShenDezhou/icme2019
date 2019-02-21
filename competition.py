#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 2019年2月20日

@author: BD-PC50
'''
import pandas as pd
data = pd.read_csv('./input/final_track2_train.txt', sep='\t', names=[
        'uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id', 'did', 'creat_time', 'video_duration'])

print(data)    