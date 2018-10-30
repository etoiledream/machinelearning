#-*-coding:utf-8-*-
#梯度的下降法实现对率回归
# author: WJX

import matplotlib.pyplot as plt
import numpy as np
import xlrd

#获取数据,留出4组检验
def getData():
    #read data from xlsx file
    workbook = xlrd.open_workbook("3.0alpha.xlsx")
    sheet = workbook.sheet_by_name("Sheet2")
    X1 = np.array(sheet.row_values(0)) #first row
    X2 = np.array(sheet.row_values(1)) #second row

    #this is the extension of x total 1
    X3 = np.array(sheet.row_values(2)) #third row

    y = np.array(sheet.row_values(3)) #forth row
    X = np.vstack([X1, X2, X3]).T
    y = y.reshape(-1, 1)
    return X, y

#梯度下降法
def GraDes(X, y):
    #阈值
    epsilon = 0.0001
    #最大迭代次数
    max_iters = 2000
    #学习率
    alpha = 0.08
    loss = 1
    count = 0
    iter_list = []
    loss_list = []

    m,n = X.shape

    #initialize paramter
    theta = np.zeros(3)
    #当损失大于阈值且迭代未达到最大次数，循环
    while loss > epsilon and count < max_iters:
        loss = 0
        for i in range(m):
            #假设函数
            h = np.dot(theta, X[i].T)
            #同时更新theta
            for j in range(len(theta)):
                theta[j] = theta[j] - alpha*(1/m)*(h - y[i])*X[i][j]
        #计算损失
        for i in range(m):
            h = np.dot(theta.T, X[i])
            every_loss = (1/(m*n))*np.power((h - y[i]),2)
            loss = loss + every_loss
        #print("iter_count: ",count,"the loss: ",loss)

        iter_list.append(count)
        loss_list.append(loss)

        count += 1
    plt.plot(iter_list, loss_list)
    plt.xlabel("iters")
    plt.ylabel("loss")
    plt.show()
    return theta

#用余下的4组值做预测
def predict(x,theta):
    y=np.dot(theta,x.T)
    return y


if __name__=="__main__":

    X, y = getData()
    theta = GraDes(X,y)
    print(theta)

    #predict
    x = np.array([[0.666,0.091,1],[0.593,0.042,1],
        [0.697,0.46,1],[0.774,0.376,1]])
    print(predict(x,theta))


