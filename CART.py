# -*- coding: utf-8 -*-
## 参考《机器学习》（Tom M. Mitchell） 第三章 决策树学习
## 《机器学习》（周志华）， 第四章 决策树

from math import log
import operator
import treePlotter

def calcGini(dataSet):
	"""
	输入：数据集
	输出：数据集的基尼指数
	描述：计算给定数据集的基尼指数
	"""
	numEntries = len(dataSet) #数据组数
	labelCounts = {}
	for featVec in dataSet: #该循环 遍历数据集，计算出标签N，Y出现的次数，存放在labelCounts里
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	gini = 0.0
	for key in labelCounts:
		pk = float(labelCounts[key])/numEntries # 计算出现概率pk
		gini += pow(pk, 2)
	gini = 1 - gini
	return gini

def splitDataSet(dataSet, axis, value):
	"""
	输入：数据集，选择维度，选择值 axis表第几个特征 value表该特征集合里的某个取值
	输出：划分数据集
	描述：按照给定特征划分数据集；去除选择维度中等于选择值的项
	"""
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reduceFeatVec = featVec[:axis]
			reduceFeatVec.extend(featVec[axis+1:])#将具有axis特征为value的值剔除
			retDataSet.append(reduceFeatVec)#再放入retDataSet
	return retDataSet

def chooseBestFeatureToSplit(dataSet):
	"""
	输入：数据集
	输出：最好的划分维度
	描述：选择最好的数据集划分维度
	"""
	numFeatures = len(dataSet[0]) - 1 #特征数量
	bestGini = float("inf") # 设置为正无穷大 
	bestFeature = -1
	for i in range(numFeatures):
		featList = [example[i] for example in dataSet] #每一循环存放第i特征值在每个分组中的数值
		uniqueVals = set(featList) #转换集合，重复值消除
		newGini = 0.0
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value) # 取value值，对数据集划分，得到自己subDataSet
			setRatio = len(subDataSet)/float(len(dataSet)) # 子集与整个数据集大小比值 Dv/D
			newGini += setRatio*calcGini(subDataSet) # sum((Dv/D)*Gini(Dv))
		if(newGini < bestGini): # 最优解替换
			bestGini = newGini
			bestFeature = i
	return bestFeature 

def majorityCnt(classList):
	"""
	输入：分类类别列表
	输出：子节点的分类
	描述：数据集已经处理了所有属性，但是类标签依然不是唯一的，
		  采用多数判决的方法决定该子节点的分类
	"""
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote] = 0
		classCount[vote] += 1
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reversed=True)
	return sortedClassCount[0][0]

def createTree(dataSet, labels):
	"""
	输入：数据集，特征标签
	输出：决策树
	描述：递归构建决策树，利用上述的函数
	"""
	classList = [example[-1] for example in dataSet] #-1是最后元素，classlist存储了数据集的因变量
	if classList.count(classList[0]) == len(classList):
		# 类别完全相同，停止划分 第一个元素出现的次数和整体长度相同，即全部一样
		return classList[0] # 将该元素返回
	if len(dataSet[0]) == 1:
		# 遍历完所有特征时返回出现次数最多的 
		return majorityCnt(classList)
	bestFeat = chooseBestFeatureToSplit(dataSet) # 选取最优特征的编号
	bestFeatLabel = labels[bestFeat] # 最好的特征是哪个
	myTree = {bestFeatLabel:{}} # 树用字典表示
	del(labels[bestFeat]) # 删除了最佳特征 用del()方法
	featValues = [example[bestFeat] for example in dataSet] # 得到列表包括节点所有的属性值
	uniqueVals = set(featValues) #强制转换成集合
	for value in uniqueVals:
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels) #创建树
	return myTree

def classify(inputTree, featLabels, testVec):
	"""
	输入：决策树，分类标签，测试数据
	输出：决策结果
	描述：跑决策树
	"""
	firstStr = list(inputTree.keys())[0]
	secondDict = inputTree[firstStr]
	featIndex = featLabels.index(firstStr)
	for key in secondDict.keys():
		if testVec[featIndex] == key:
			if type(secondDict[key]).__name__ == 'dict':
				classLabel = classify(secondDict[key], featLabels, testVec)
			else:
				classLabel = secondDict[key]
	return classLabel

def classifyAll(inputTree, featLabels, testDataSet):
	"""
	输入：决策树，分类标签，测试数据集
	输出：决策结果
	描述：跑决策树
	"""
	classLabelAll = []
	for testVec in testDataSet:
		classLabelAll.append(classify(inputTree, featLabels, testVec))
	return classLabelAll

def storeTree(inputTree, filename):
	"""
	输入：决策树，保存文件路径
	输出：
	描述：保存决策树到文件
	"""
	import pickle
	fw = open(filename, 'wb')
	pickle.dump(inputTree, fw)
	fw.close()

def grabTree(filename):
	"""
	输入：文件路径名
	输出：决策树
	描述：从文件读取决策树
	"""
	import pickle
	fr = open(filename, 'rb')
	return pickle.load(fr)

def createDataSet():
	"""
	outlook->  0: sunny | 1: overcast | 2: rain
	temperature-> 0: hot | 1: mild | 2: cool
	humidity-> 0: high | 1: normal
	windy-> 0: false | 1: true 
	"""
	dataSet = [[0, 0, 0, 0, 'N'], 
			   [0, 0, 0, 1, 'N'], 
			   [1, 0, 0, 0, 'Y'], 
			   [2, 1, 0, 0, 'Y'], 
			   [2, 2, 1, 0, 'Y'], 
			   [2, 2, 1, 1, 'N'], 
			   [1, 2, 1, 1, 'Y']]
	labels = ['outlook', 'temperature', 'humidity', 'windy']
	return dataSet, labels

def createTestSet():
	"""
	outlook->  0: sunny | 1: overcast | 2: rain
	temperature-> 0: hot | 1: mild | 2: cool
	humidity-> 0: high | 1: normal
	windy-> 0: false | 1: true 
	"""
	testSet = [[0, 1, 0, 0], 
			   [0, 2, 1, 0], 
			   [2, 1, 1, 0], 
			   [0, 1, 1, 1], 
			   [1, 1, 0, 1], 
			   [1, 0, 1, 0], 
			   [2, 1, 0, 1]]
	return testSet

def main():
	dataSet, labels = createDataSet()
	labels_tmp = labels[:] # 拷贝，createTree会改变labels
	desicionTree = createTree(dataSet, labels_tmp)
	#storeTree(desicionTree, 'classifierStorage.txt')
	#desicionTree = grabTree('classifierStorage.txt')
	print('desicionTree:\n', desicionTree)
	treePlotter.createPlot(desicionTree)
	testSet = createTestSet()
	print('classifyResult:\n', classifyAll(desicionTree, labels, testSet))

if __name__ == '__main__':
	main()