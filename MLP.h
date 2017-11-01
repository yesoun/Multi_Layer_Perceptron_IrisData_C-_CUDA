#ifndef MLP_CUDA_H
#define MLP_CUDA_H

#include <iostream> // for std:: cout and std::cin
#include <fstream> // input/output stream class to operate on files
#include <stdlib.h> // or <cstdlib> defines several general purpose functions, including dynamic memory management, random number generation, communication with the environment, integer arithmetics, searching and converting
#include <math.h> // or <cmath> declares a set of functions to compute common mathematica operations and transformations
#include <vector> //
#include <iterator>
#include <string>
#include <sstream> // Stream Class to Operate on strings and objects of this class use a string buffer that contains a sequence of characters. This sequence of characters can be directly accessed as a string object
#include <iomanip> // for the std::setprecision that is used to limitate prinitin to 2 digits in output of tested data


class MLP
{
    public:
            /// These Parameters have to be private and I should define for each one an
            // number of hidden neurons extracted from the irisTest.txt
            int numberOfHiddenNeurons;
            // matrix containing all the weights between the hidden neuron and the output neurons
            double** weightsAfterHidden;
            // contains how the number of inputs neurons + Bias
            int numberOfInputNeurons;
            // matrix containing all the weights between the inputs neuron and the hidden neurons
            double** weights;
            // for all the output neurons activation value with sum of wij*xi then applying sigmoid Function
            double* OutputActivationArray;
            // for all the hidden neurons activation value with sum of wij*xi then applying sigmoid Function
            double* HiddenNeuronActivationArray;
            // matrix containing all the inputs of the perceptron network with a -1 Bias.
            double** inputs;
            // matrix containing all the Outputs of training data of the perceptron network.
            double** outputs;
            // matrix containing all the test data used with perceptron network after finding all the weights to test.
            double** testData;
            // matrix containing all the copy of test data used with MLP network without normalisation nor bias added.
            double** CopytestData;
            // learning rate eita that describes how much fast the network learns and controls how much to change the weights by.
            double eita;
            // contains how the number of output neurons
            int numberOfOutputNeurons;
            // contains how many row of inputs or inputs pattern we have
            int numberOfPatterns;
            // contains the number of patterns of the test data
            int numberOfTestDataPatterns;

            void readingFileSettingParameters(MLP& perceptron);
            // delete the memory of the allocated memory especially for the  variables of the MultiLayerPerceptronNetwork
            void deletingMemory(double** weightsAfterHidden, int sizeweightsAfterHidden, double** weights, int sizeweights, double** inputs,int sizeinputs,
                                double** outputs, int sizeoutputs, double** testData, int sizetestData, double** CopytestData, int sizeCopytestData,
                                double* HiddenNeuronActivationArray, double* OutputActivationArray );
            // normaliwe the matrix by using the chiSquareNormalize function for normalization and not the min max one of the single layer perceptron
            double** chiSquareNormalize(double** inputWithoutBias, int rowsinputWithoutBias, int columnsinputWithoutBias, double** meanLast, double** standardDeviation);
            // calculate the standard deviation of each vector and return a 1*N matrix with value of the standard deviation of each column of the input matrix
            double** CalculateStandardDeviation (double** input, int rowsinputWithoutBias, int columnsinputWithoutBias);
            // will create a vector containing all the mean value per each columnsinputWithoutBias
            double** calculateMean( double**inputWithoutBias, int rowsinputWithoutBias, int columnsinputWithoutBias);
            // sigmoid Activation function
            double SigmoidFunction(double x);
            // step function NOT USED IN THIS EXAMPLE OR IRIS DTAT, BUT USED FOR THE CUBE DATA PROBLEM
            double StepFunction(double x);
            // initilise activation array for the three neurons to 0 for each one of them.
            double* InitialiseActivationArray(int number);
            double** RandomDoubleWeights(double min, double max, int NumRows, int NumCols);

};
#endif
