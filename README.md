# Multi_Layer_Perceptron_IrisData_C-_CUDA

Build a multilayer neural network in C/C++ using the algorithm on page 78 in your book. Use a single hidden layer.
Your network will classify iris (the flower) species by looking at four lengths of the flower. This problem is not as easy.
But doable for a multilayer network. For each of the four measures there is a classification 0, 1, or 2 which is the species, however,
I have converted this to three channels of output for you. Build a network with 4 inputs nodes and 3 output nodes, one for each possible classification.
Assume a bias node of -1. Use the classic sigmoid function with a slope of your choice. Normalize the input appropriately as discussed in class.
*****************************
****** The Training: ********
*****************************
Your program will read in training data from standard input. Your program can then train as much as you believe is needed and then print the W matrix. Mine did fine on 2000 passes. It only miss a couple of the cases.
It will then read in test data. For this assignment use all the training data and then use the test data to see how well your net was trained. Do not train on the test data. The input file format looks like:
#inputs
#numberOfHiddenNodes
#rows #cols
row1
row2
 ...
lastrow
#rows #cols
row1
row2
 ...
lastrow

The training and test data each look like a matrix specification. First number is the number of features or inputs. The second is the number of hidden nodes. Then comes the dimensions of the training data: number of rows and columns. The #rows in the training matrix is greater than #inputs. The #rows in the second matrix is the same as #inputs. Your program will read in the training data and then train. Then it will read in the test data. (The test data testDataA3.tar shows examples of the input format.)
Your program will then print exactly:
BEGIN TESTING
Then for each test case it will print out on one line the input test case followed output using your learned W matrix.
The test script will compare everything after the "BEGIN TESTING" mark with the expected results. Numerical results will be expected to be within 10% to be a match.
Do not use any prepackaged software. You should write the whole program for each task from scratch.
