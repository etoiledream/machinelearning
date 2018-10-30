# -*- coding:UTF-8 -*-
import numpy as np

# neural network class definition
class neuralNetwork(object):#继承object对象


    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        # 权重采用随机 -0.5到0.5间

        self.wih = np.random.rand(hiddennodes, inputnodes) - 0.5
        self.who = np.random.rand(outputnodes, hiddennodes) - 0.5

        # learning rate
        self.lr = learningrate

        # activation function is the sigmoid function


    def sigmoid(self, x):
        return 1.0/(1.0+np.exp(-x))

    def softmax(self, x):
        pass

    def SGD(self):
        pass

    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        # 列表转换数组
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.sigmoid(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.sigmoid(final_inputs)

        # output layer error is the (target - actual)
        # 计算输出与实际输出的误差
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        # 反向传播误差
        hidden_errors = np.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

        pass


    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        # 输入层到隐藏层：隐藏前权重*输入向量
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        # 经过一次激活函数sigmoid
        hidden_outputs = self.sigmoid(hidden_inputs)

        # calculate signals into final output layer
        # 隐藏层到输出层：隐藏后权重*激活后向量
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        # 输出层激活函数softmax，得到最终输出向量
        final_outputs = self.sigmoid(final_inputs)

        return final_outputs


input_nodes = 25
hidden_nodes = 50
output_nodes = 6
learning_rate = 0.1

n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

training_data_file = open("train.txt", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 10000

for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (np.asfarray(all_values[1:]) * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = np.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

test_data_file = open("test.txt", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

scorecard = []

# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (np.asfarray(all_values[1:]) * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = np.argmax(outputs)
    #print correct value and calculated value
    print("correct:"+str(correct_label)+" network:"+str(label))
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass

    pass

scorecard_array = np.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)
