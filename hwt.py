import KNN
group,labels=KNN.createDataSet( )
print(group)
print(labels)
KNN.classify0([0,0],group,labels,3)
testVector=KNN.img2vector('testDigits/0_13.txt')
print(testVector[0,0:31])
print(testVector[0,32:63])
KNN.handwritingClassTest( )