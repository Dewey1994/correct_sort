#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
desc: 初始化字典
author: 老马啸西风
date: 2021-11-24
'''
def initDict(path):
   dict = {}
   with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f.readlines():
            # 移除换行符，并且根据空格拆分
            splits = line.strip('\n').split(' ')
            key = splits[0]
            value = splits[1]
            dict[key] = value
   return dict
   
# 字典初始化 
bihuashuDict = initDict('db/bihuashu_2w.txt')
hanzijiegouDict = initDict('db/hanzijiegou_2w.txt')
pianpangbushouDict = initDict('db/pianpangbushou_2w.txt')
sijiaobianmaDict = initDict('db/sijiaobianma_2w.txt')

# 权重定义（可自行调整）
hanzijiegouRate = 10
sijiaobianmaRate = 8
pianpangbushouRate = 6
bihuashuRate = 2

# 计算核心方法
'''
desc: 笔画数相似度
'''
def bihuashuSimilar(charOne, charTwo): 
    valueOne = bihuashuDict[charOne]
    valueTwo = bihuashuDict[charTwo]
    
    numOne = int(valueOne)
    numTwo = int(valueTwo)
    
    diffVal = 1 - abs((numOne - numTwo) / max(numOne, numTwo))
    return bihuashuRate * diffVal * 1.0

    
'''
desc: 汉字结构数相似度
'''
def hanzijiegouSimilar(charOne, charTwo): 
    valueOne = hanzijiegouDict[charOne]
    valueTwo = hanzijiegouDict[charTwo]
    
    if valueOne == valueTwo:
        # 后续可以优化为相近的结构
        return hanzijiegouRate * 1
    return 0
    
'''
desc: 四角编码相似度
'''
def sijiaobianmaSimilar(charOne, charTwo): 
    valueOne = sijiaobianmaDict[charOne]
    valueTwo = sijiaobianmaDict[charTwo]
    
    totalScore = 0.0
    minLen = min(len(valueOne), len(valueTwo))
    
    for i in range(minLen):
        if valueOne[i] == valueTwo[i]:
            totalScore += 1.0
    
    totalScore = totalScore / minLen * 1.0
    return totalScore * sijiaobianmaRate

'''
desc: 偏旁部首相似度
'''
def pianpangbushoutSimilar(charOne, charTwo): 
    valueOne = pianpangbushouDict[charOne]
    valueTwo = pianpangbushouDict[charTwo]
    
    if valueOne == valueTwo:
        # 后续可以优化为字的拆分
        return pianpangbushouRate * 1
    return 0  
