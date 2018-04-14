import numpy as np
import matplotlib.pyplot as plt

def readData(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y

def cross_entropy(y_bar,y_onehot):
    num_elements = y_onehot.shape[0]
    return -(1/num_elements)* np.sum(np.multiply(y_onehot,np.log(y_bar))) 


def softmax(x):
    
    """
    Compute softmax function for input. 
    Use tricks from previous assignment to avoid overflow
    """
	### YOUR CODE HERE
    num_elements= x.shape[0]
    s = np.zeros(x.shape)
    for i in range(0,num_elements):
         s[i,:]=np.exp(x[i,:]-np.max(x[i,:]))/sum(np.exp(x[i,:]-np.max(x[i,:])))  

	### END YOUR CODE
    return s

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    """
    ### YOUR CODE HERE
    s=1/(np.exp(-x)+1)
#    s=  1/(1+np.exp(-x))
    ### END YOUR CODE
    return s

def forward_prop(data, labels, params):
    """
    return hidder layer, output(softmax) layer and loss
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    X=data
    ### YOUR CODE HERE
    z1 = X.dot(W1) + b1
    a1=sigmoid(z1)
    z2 = a1.dot(W2) + b2
    a2=softmax(z2)
    y=a2
    h=a1
    cost = cross_entropy(a2,labels)
    ### END YOUR CODE
    return h, y, cost

def backward_prop(data, labels, params):
    """
    return gradient of parameters
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    X=data
    m=data.shape[0]
    ### YOUR CODE HERE
    z1 = X.dot(W1) + b1
    a1=sigmoid(z1)
    z2 = a1.dot(W2) + b2
    a2=softmax(z2)
    
    grad = {}


    gradW2 = (1/m)*a1.T.dot(a2-labels)
    gradb2 = (1/m)*np.sum((a2-labels), axis=0, keepdims=True)
    delta2 = (a2-labels).dot(W2.T)*a1*(1-a1)
    gradW1 = (1/m)*np.dot(X.T, delta2)
    gradb1 = (1/m)*np.sum(delta2, axis=0)
    
    
    grad['W1'] = gradW1
    grad['W2'] = gradW2
    grad['b1'] = gradb1
    grad['b2'] = gradb2
    ### END YOUR CODE
    
    
        # Gradient descent parameter update
#        W1 += -epsilon * gradW1
#        b1 += -epsilon * gradb1
#        W2 += -epsilon * gradW2
#        b2 += -epsilon * gradb2

    return grad

def nn_train(trainData, trainLabels, devData, devLabels):
    (m, n) = trainData.shape
    num_hidden = 300
    learning_rate = 5
    mini_batch_size = 1000
    num_batch = m/mini_batch_size 
    
    epochs=30
     
    np.random.seed(0)
    W1 = np.random.randn(n,num_hidden) / np.sqrt(n)
    b1 = np.zeros((1, num_hidden))
    W2 = np.random.randn(num_hidden, 10) / np.sqrt(num_hidden)
    b2 = np.zeros((1, 10))
    
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
   # train_loss = np.zeros[epochs,1]
       
    ### YOUR CODE HERE
    for j in range(0,epochs):
        cost_per_epoch=0
        avg_cost=0
        for ii in range(0,50):
            
            start_idx = (ii)*mini_batch_size
            end_idx = mini_batch_size + start_idx
            
            X =  trainData[start_idx:end_idx,:]
            y_onehot = trainLabels[start_idx:end_idx,:]
            
            (h, y, new_cost) = forward_prop(X,y_onehot,params)
#            print(new_cost)
            cost_per_epoch = cost_per_epoch + new_cost
            gradients = backward_prop(X,y_onehot,params)
     
            epsilon = learning_rate 
            params['W1'] += -epsilon * (gradients['W1'])#+2*.0001*params['W1']) 
            params['b1'] += -epsilon * gradients['b1']
            params['W2'] += -epsilon * (gradients['W2'])#+2*.0001*params['W2'])
            params['b2'] += -epsilon * gradients['b2']

        avg_cost = cost_per_epoch/num_batch
        #train_loss[j]=avg_cost
        print("Avg cost for epoch #%f is %f",j,avg_cost)

        train_accuracy = nn_test(trainData, trainLabels, params)
        print( 'Train accuracy: %f' % train_accuracy)

        dev_accuracy = nn_test(devData, devLabels, params)
        print( 'Dev accuracy: %f' % dev_accuracy)
           
        
    ### END YOUR CODE
        
    return params

def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy

def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) == np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels

def main():
    np.random.seed(100)
    trainData, trainLabels = readData('images_train.csv', 'labels_train.csv')
    trainLabels = one_hot_labels(trainLabels)
    p = np.random.permutation(60000)
    trainData = trainData[p,:]
    trainLabels = trainLabels[p,:]

    devData = trainData[0:10000,:]
    devLabels = trainLabels[0:10000,:]
    trainData = trainData[10000:,:]
    trainLabels = trainLabels[10000:,:]

    mean = np.mean(trainData)
    std = np.std(trainData)
    trainData = (trainData - mean) / std
    devData = (devData - mean) / std

    testData, testLabels = readData('images_test.csv', 'labels_test.csv')
    testLabels = one_hot_labels(testLabels)
    testData = (testData - mean) / std
    params = nn_train(trainData, trainLabels, devData, devLabels)


    readyForTesting = True
    if readyForTesting:
        accuracy = nn_test(testData, testLabels, params)
        print( 'Test accuracy: %f' % accuracy)

if __name__ == '__main__':
    main()