from numpy import *
import operator
from os import listdir#从os中导入函数listdir，它可以列出给定目录的文件名
def dict2list(dic:dict):#将字典转化为列表
    keys=dic.keys( )
    vals=dic.values( )
    lst=[(key,val)for key,val in zip(keys,vals)]
    return lst
def createDataSet():  #函数作用：构建一组训练数据（训练样本），共4个样本，同时给出了这4个样本的标签，及labels
    group=array([[1.0,1.1],[1.0,1.0],[0.0,0.0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels
def classify0(inX,dataSet,labels,k):
    #inX是输入的测试样本，是【x，y】样式的，dataset是训练样本集，labels是训练样本标签，k表示选择最近邻的数目
    #其中标签向量的元素数目和矩阵dataset的行数相同
    dataSetSize=dataSet.shape[0]  #shape返回矩阵的【行数，列数】，返回数据集的行数
    #求距离过程使用欧式距离公式
    diffMat=tile(inX,(dataSetSize ,1))-dataSet  # tile（a，（b，c）将a沿x轴复制c倍，沿y复制b倍，复制一倍相当于没有复制
    #diffMat就是输入样本与每个训练样本的差值，然后对其每个x和y的差值进行平方运算。
    # diffMat是一个矩阵，矩阵**2表示对矩阵中的每个元素进行**2操作，即平方。
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)  #axis=1表示按照横轴，sum表示累加，即按照行进行累加
    distances=sqDistances**0.5  #对平方和进行开根号
    sortedDistIndicies=distances.argsort()  #按照升序进行快速排序，返回的是原数组的下标。比如，x = [30, 10, 20, 40]，升序排序后应该是[10,20,30,40],他们的原下标是[1,2,0,3]
    classCount={ }  #存放最终的分类结果及相应的结果投票数
    # 投票过程，就是统计前k个最近的样本所属类别包含的样本个数
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]] #index = sortedDistIndicies[i]是第i个最相近的样本下标，voteIlabel = labels[index]是样本index对应的分类结果('A' or 'B')
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1  #classCount.get(voteIlabel, 0)返回voteIlabel的值，如果不存在，则返回0，然后将票数增1
    #把分类结果进行排序，然后返回得票数最多的分类结果
    sortedClassCount=sorted( dict2list(classCount),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount [0][0]
def img2vector(filename) : #将图像转化为向量，把一个32*32的二进制图像矩阵转换为1*1024的向量
    returnVect=zeros((1,1024)) #创建1*1024的Numpy的数组
    fr=open(filename) #打开给定的文件
    for i in range (32): #循环读出文件的前32行
        lineStr=fr.readline( )
        for j in range (32): #将每行的头32个字符存储在Numpy数组中
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect
def handwritingClassTest(): #测试分类器的代码
    hwlabels=[]
    #加载训练数据
    trainingFileList=listdir('trainingDigits') #将traingDigits目录中的文件内容存储在列表中
    m=len(trainingFileList)
    trainingMat=zeros((m,1024)) #创建一个m行1024列的训练矩阵，该矩阵的每行数据存储一个图像
    for i in range(m):
        #从文件名中解析出分类数字
        #文件名格式是0_3.txt表示图片数字是0，它是数字0的第3个实例
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split(' . ')[0]
        classNumStr=int(fileStr.split('_')[0])
        hwlabels.append(classNumStr) #将类代码存储在hwLabel向量中
        trainingMat[i,:]=img2vector('trainingDigits/%s ' % fileNameStr) #使用img2vector函数载入图像
   #加载测试数据
    testFileList=listdir('testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range (mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        vectorUnderTest=img2vector('testDigits/%s'%fileNameStr)
        classifierResult=classify0(vectorUnderTest,trainingMat,hwlabels,3) #使用classify0（）函数测试该目录下的每个文件
        print('the classifier came back with:%d,the real answer is: %d '%(classifierResult,classNumStr))
        if (classifierResult!=classNumStr):
            errorCount+=1.0
    print("\n the total number of errors is:%d" % errorCount)
    print("\n the total error rate is :%f" %(errorCount/float(mTest)))