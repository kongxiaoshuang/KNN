#-*- coding: utf-8 -*-
from numpy import *
import operator

import matplotlib
import matplotlib.pyplot as plt

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k): #inX为用于分类的输入向量，dataSet为输入的训练样本集， labels为训练标签，k表示用于选择最近的数目
    dataSetSize = dataSet.shape[0] #dataSet的行数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet #将inX数组复制成与dataSet相同行数，与dataSet相减，求坐标差
    sqDiffMat = diffMat**2 #diffMat的平方
    sqDistances = sqDiffMat.sum(axis=1) #将sqDiffMat每一行的所有数相加
    distances = sqDistances**0.5 #开根号，求点和点之间的欧式距离
    sortedDistIndicies = distances.argsort() #将distances中的元素从小到大排列，提取其对应的index，然后输出到sortedDistIndicies
    classCount = {} #创建字典
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]] #前k个标签数据
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 #判断classCount中有没有对应的voteIlabel，
        # 如果有返回voteIlabel对应的值，如果没有则返回0，在最后加1。为了计算k个标签的类别数量
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True) #生成classCount的迭代器，进行排序，
                # operator.itemgetter(1)以标签的个数降序排序
    return sortedClassCount[0][0] #返回个数最多的标签

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines() #读入所有行
    numberOfLines = len(arrayOLines) #行数
    returnMat = zeros((numberOfLines, 3))  #创建数组，数据集
    classLabelVector = [] #标签集
    index = 0
    for line in arrayOLines:
        line = line.strip()   #移除所有的回车符
        listFromLine = line.split('\t')  #把一个字符串按\t分割成字符串数组
        returnMat[index,:] = listFromLine[0:3] #取listFromLine的前三个元素放入returnMat
        classLabelVector.append(int(listFromLine[-1])) #选取listFromLine的最后一个元素依次存入classLabelVector列表中
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0) #0表示从列中选取最小值
    maxVals = dataSet.max(0) #选取最大值
    ranges = maxVals-minVals
    normDataSet = zeros(shape(dataSet))  #创建一个与dataSet大小相同的零矩阵
    m = dataSet.shape[0] #取dataSet得行数
    normDataSet = dataSet - tile(minVals, (m, 1))  #将minVals复制m行 与dataSet数据集相减
    #归一化相除
    normDataSet = normDataSet/tile(ranges, (m, 1)) #将最大值-最小值的值复制m行 与normDataSet相除，即归一化
    return normDataSet, ranges, minVals #normDataSet为归一化特征值，ranges为最大值-最小值

def datingClassTest():
    hoRatio = 0.10 #测试数据占总数据的百分比
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt') #将文本信息转成numpy格式
    #datingDataMat为数据集，datingLabels为标签集
    normMat, ranges, minVals = autoNorm(datingDataMat)  #将datingDataMat数据归一化
    #normMat为归一化数据特征值，ranges为特征最大值-最小值，minVals为最小值
    m = normMat.shape[0] #取normMat的行数
    numTestVecs = int(m*hoRatio) #测试数据的行数
    errorCount = 0.0 #错误数据数量
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        #classify0为kNN分类器，normMat为用于分类的输入向量，normMat为输入的训练样本集（剩余的90%）
        #datingLabels为训练标签，3表示用于选择最近邻居的数目
        print("the classifier came back with: %d, the real answer is: %d" %(classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):errorCount += 1.0 #分类器结果和原标签不一样，则errorCount加1
    print("the total error rate is : %f" %(errorCount/float(numTestVecs)))


# datingClassTest()



# datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
#
# normDataSet, ranges, minVals = autoNorm(datingDataMat)

# fig = plt.figure()
# ax = fig.add_subplot(111) #一行一列一个
# ax.scatter(datingDataMat[:,1], datingDataMat[:,2],
#            15.0*array(datingLabels), 15.0*array(datingLabels))   #scatter画散点图，使用标签属性绘制不同颜色不同大小的点
# plt.show()

# #测试分类器
# group, labels = createDataSet()
# label = classify0([1,1], group, labels, 3)
# print(label)

from os import listdir

def img2vector (filename):
    returnVect = zeros((1, 1024)) #创建一个1*1024的数组
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline() #每次读入一行
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = [] #标签集
    trainingFileList = listdir('E:/digits/trainingDigits') #listdir获取训练集的文件目录
    m = len(trainingFileList) #文件数量
    trainingMat = zeros((m, 1024)) #一个数字1024个字符，创建m*1024的数组
    for i in range(m):
        fileNameStr = trainingFileList[i] #获取文件名
        fileStr = fileNameStr.split('.')[0] #以'.'将字符串分割，并取第一项，即0_0.txt取0_0
        classNumStr = int(fileStr.split('_')[0]) #以'_'将字符串分割，并取第一项
        hwLabels.append(classNumStr) #依次存入hwLabels标签集
        trainingMat[i, :] = img2vector('E:/digits/trainingDigits/%s' % fileNameStr) #将每个数字的字符值依次存入trainingMat
    testFileList = listdir('E:/digits/testDigits') #读入测试数据集
    errorCount = 0.0 #测试错误数量
    mTest = len(testFileList) #测试集的数量
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0]) #测试数据标签
        vectorUnderTest = img2vector('E:/digits/testDigits/%s' % fileNameStr) #读入测试数据
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3) #分类器kNN算法，3为最近邻数目
        print("the calssifier came back with: %d, the real answer is : %d" %(classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount +=1.0
    print("\nthe total number of errors is : %f" % errorCount)
    print("\nthe total error rate is :%f" % (errorCount/float(mTest)))

handwritingClassTest()