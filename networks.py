#sample_submission.py
import numpy as np

class xor_net(object):
    """
    This is a sample class for miniproject 1.

    Args:
        data: Is a tuple, ``(x,y)``
              ``x`` is a two or one dimensional ndarray ordered such that axis 0 is independent 
              data and data is spread along axis 1. If the array had only one dimension, it implies
              that data is 1D.
              ``y`` is a 1D ndarray it will be of the same length as axis 0 or x.   
                          
    """
    def __init__(self, data, labels):
        self.x = data
        # normalize input data
        self.x = (self.x-np.mean(self.x,axis=0)) / np.std(self.x,axis=0)
        self.y = labels
        self.lr = 0.01
        self.hidden_nodes = 200
        self.hidden_nodes_2 = 50
        self.input_nodes = self.x.shape[1]
        self.output_nodes = 2
        self.iterations = 30000
        self.alpha = 0.1
        self.params = []
        self.training()

    def cost_function(self):
        """
        This method is used to calculate cost at each iteration
        Args:
            W1,b1: weights and bias between input layer and hidden layer 1
            W2,b2: weights and bias between hidden layer 1 and hidden layer 2
            W3,b3: weights and bias between hidden layer 2 and output layer
            l1: input layer, l2: hidden layer 1, l3 : hidden layer 2
            s_m: softmax value
            L2: L2 norm

        Returns:
            over all cost/error incurred with current weights

        """
        W_l1 = self.params[0][0]
        b_l1 = self.params[0][1]
        W_l2 = self.params[1][0]
        b_l2 = self.params[1][1]
        W_l3 = self.params[2][0]
        b_l3 = self.params[2][1]
        ip_l1 = self.x.dot(W_l1) + b_l1
        op_l1 = np.tanh(ip_l1)
        ip_l2 = op_l1.dot(W_l2) + b_l2
        op_l2 = np.tanh(ip_l2)
        ip_l3 = op_l2.dot(W_l3) + b_l3
        op_l3 = np.exp(ip_l3)
        s_m = op_l3 / np.sum(op_l3, axis=1, keepdims=True)
        nll = -np.log(s_m[range(len(self.x)), self.y])
        error = np.sum(nll)
        W_l1_L2 = np.sum(np.square(W_l1))
        W_l2_L2 = np.sum(np.square(W_l2))
        W_l3_L2 = np.sum(np.square(W_l3))
        L2 = W_l1_L2+ W_l2_L2 + W_l3_L2
        error = error+ self.alpha / 2 * L2
        return 1. / len(self.x) * error

    def training(self):
        """
        In this method we are cooking the network.
        We go formward propagation in the below manner: input layer -> hidden layer 1 -> hidden layer 2 -> output layer
        We have 3 connections: we are using tanh as activation function in first 2 connection and softmax in last connection
        Once we have overall error we are back propogating, all the formulas are derived using partial derivatives.
        After calculating deltas, we can do weight updations.
        Returns:
        The trained weights
        """
        print "X shape:", self.x.shape
        print "Input dimension:", self.input_nodes
        print "Y shape:", self.y.shape
        np.random.seed(1)
        w_l1 = np.random.randn(self.input_nodes,self.hidden_nodes)
        b1 = np.ones((1,self.hidden_nodes))
        print "w_l1  shape:", w_l1.shape
        print "b1 shape:", b1.shape
        w_l2 = np.random.randn(self.hidden_nodes,self.hidden_nodes_2)
        b2 = np.ones((1,self.hidden_nodes_2))
        print "w_l2 shape:", w_l2.shape
        print "b2 shape:", b2.shape
        w_l3 = np.random.randn(self.hidden_nodes_2,self.output_nodes)
        b3 = np.ones((1,self.output_nodes))
        print "w_l3  shape:", w_l3.shape
        print "b3 shape:", b3.shape
        print "My code is running for 30000 iterations on my 4GB Mac it will take 4 mins to runs. You can see progress for every 10k iterations"

        for i in range(0, self.iterations):
            # forward propogation
            ip_hidden = self.x.dot(w_l1)+b1
            op_hidden = np.tanh(ip_hidden)
            ip_hidden_2 = op_hidden.dot(w_l2)+b2
            op_hidden_2 = np.tanh(ip_hidden_2)
            ip_output = op_hidden_2.dot(w_l3)+b3
            op_output = np.exp(ip_output)
            p = op_output/np.sum(op_output, axis = 1, keepdims = True)

            #diff_output is the error difference
            diff_output = p
            diff_output[range(len(self.x)),self.y] -= 1

            # distribute the error over all the layers
            diff_hidden_weights_2 = (op_hidden_2.T).dot(diff_output)
            diff_hidden_bias_2 = np.sum(diff_hidden_weights_2, axis=0, keepdims=True)
            diff_hidden_2 = diff_output.dot(w_l3.T) * (1 - np.power(op_hidden_2, 2))

            diff_hidden_weights = (op_hidden.T).dot(diff_hidden_2)
            diff_hidden_bias = np.sum(diff_hidden_weights, axis=0, keepdims=True)
            diff_hidden = diff_hidden_2.dot(w_l2.T) * (1 - np.power(op_hidden, 2))

            diff_input_weights = np.dot(self.x.T, diff_hidden)
            diff_input_bias = np.sum(diff_hidden, axis=0)

            #update weight with L2 norm and learning rate
            w_l1 = w_l1 -self.lr * (diff_input_weights + self.alpha * w_l1)
            b1 = b1 -self.lr * diff_input_bias
            w_l2 = w_l2 -self.lr * (diff_hidden_weights + self.alpha * w_l2)
            b2 = b2 -self.lr * diff_hidden_bias
            w_l3 = w_l3 -self.lr * (diff_hidden_weights_2+self.alpha * w_l3)
            b3 = b3 -self.lr * diff_hidden_bias_2
            self.params = [[w_l1,b1], [w_l2,b2], [w_l3,b3]]
            if( i% 10000 == 0):
                print "completed iterations:", i
        return self.params

    def get_params (self):
        """
        Method that should return the model parameters.

        Returns:
            tuple of numpy.ndarray: (w, b).

        Notes:
            This code will return an empty list for demonstration purposes. A list of tuples of
            weoghts and bias for each layer. Ordering should from input to outputt

        """
        return self.params

    def get_predictions (self, x):
        """
        Method should return the outputs given unseen data

        Args:
            x: array similar to ``x`` in ``data``. Might be of different size.

        Returns:
            numpy.ndarray: ``y`` which is a 1D array of predictions of the same length as axis 0 of
                            ``x``
        Notes:
            Temporarily returns random numpy array for demonstration purposes.
        """
        # normalize input data
        x = (x-np.mean(x,axis=0)) / np.std(x,axis=0)
        W_l1 = self.params[0][0]
        b_l1 = self.params[0][1]
        W_l2 = self.params[1][0]
        b_l2 = self.params[1][1]
        W_l3 = self.params[2][0]
        b_l3 = self.params[2][1]
        # forward caluculation to get our prediction
        ip_l1 = x.dot(W_l1) + b_l1
        op_l1 = np.tanh(ip_l1)
        ip_l2 = op_l1.dot(W_l2) + b_l2
        op_l2 = np.tanh(ip_l2)
        ip_l3 = op_l2.dot(W_l3) + b_l3
        op_l3 = np.exp(ip_l3)
        s_m = op_l3 / np.sum(op_l3, axis=1, keepdims=True)
        return np.argmax(s_m, axis=1)

class mlnn(xor_net):
    """
    At the moment just inheriting the network above. 
    """
    def __init__ (self, data, labels):
        super(mlnn,self).__init__(data, labels)


if __name__ == '__main__':
    pass 
