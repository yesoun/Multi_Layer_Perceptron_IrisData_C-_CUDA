/*   Author: Yassine Maalej (Github: yesoun) 
   email: maalej.yessine@gmail.com && maal4948@vandals.uidaho.edu
   Date: Frebuary 2016
   Class: CS541 Advanced Operating Systems
   Insitution: University Of Idaho
*/
/*
Build a multilayer neural network in C/C++ using the algorithm on page 78 in your book. Use a single hidden layer.
Your network will classify iris (the flower) species by looking at four lengths of the flower. This problem is not as easy.
But doable for a multilayer network. For each of the four measures there is a classification 0, 1, or 2 which is the species, however,
I have converted this to three channels of output for you. Build a network with 4 inputs nodes and 3 output nodes, one for each possible classification.
Assume a bias node of -1. Use the classic sigmoid function with a slope of your choice. Normalize the input appropriately as discussed in class.
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

Normalization+Training+Inference are parallelised in CUDA
*/

#include <iostream> // for std:: cout and std::cin
#include <fstream> // input/output stream class to operate on files
#include <stdlib.h> // or <cstdlib> defines several general purpose functions, including dynamic memory management, random number generation, communication with the environment, integer arithmetics, searching and converting
#include <math.h> // or <cmath> declares a set of functions to compute common mathematica operations and transformations
#include <vector> //
#include <iterator>
#include <string>
#include <sstream> // Stream Class to Operate on strings and objects of this class use a string buffer that contains a sequence of characters. This sequence of characters can be directly accessed as a string object
#include <iomanip> // for the std::setprecision that is used to limitate prinitin to 2 digits in output of tested data
#include "MLP.h"
using namespace std;
// create a weight matrix with random variables between min and max with NumRows Rows and NumCols Columns
// in this problem of IRIS Data min=.1 and max=.5 and NumRows=150 and NumCols=4
double** MLP::RandomDoubleWeights(double min, double max, int NumRows, int NumCols)
{
	double** weights= new double*[NumRows];
	double var= 0;
	for (int i=0;i<NumRows;i++){
        weights[i] = new double[NumCols];
        for (int j=0;j<NumCols;j++){
            var= (double)rand()/(double)(RAND_MAX);
            var= min + var*(max-min);
            weights[i][j]=var;
        }
	}
	return weights;
}
// initilise activation array for the three neurons to 0 for each one of them.
double* MLP::InitialiseActivationArray(int number)
{
	double* activationArray= new double[number];
	double var= 0;
	for (int i=0;i<number;i++){
        activationArray[i] = var;
	}
	return activationArray;
}
// sigmoid Activation function
double MLP::SigmoidFunction(double x)
{
	double y = 1.0/(1.0+ exp(-1.0*x));
	return y;
}
// step function NOT USED IN THIS EXAMPLE OR IRIS DTAT, BUT USED FOR THE CUBE DATA PROBLEM
double MLP::StepFunction(double x)
{
	if(x<=0){
        x=0;
	}
	else{
        x=1;
	}
	return x;
}
// will create a vector containing all the mean value per each columnsinputWithoutBias
double** MLP::calculateMean( double**inputWithoutBias, int rowsinputWithoutBias, int columnsinputWithoutBias)
{
	double** mean = new double*[1];
	mean[0]= new double[columnsinputWithoutBias];
	//cout<< " ********* printing the mean ********* "<< endl;
	// we now calculate the mean per average
	for(int i=0;i<columnsinputWithoutBias;i++){
		//mean[0][i]=0.0;
		for(int j=0;j<rowsinputWithoutBias;j++){
			mean[0][i]+=inputWithoutBias[j][i];
            //cout << "mean[0]["<<i<<"] "<<mean[0][i] << " " << endl;
            //cout << "inputWithoutBias[j"<<"]["<<"i]"<<inputWithoutBias[j][i] << " " << endl;
		}
		mean[0][i]=(mean[0][i])/rowsinputWithoutBias;
		//cout << "division with mean[0]["<<i<<"] "<<mean[0][i] << " " << endl;
		//cout<< "rowsrows ="<< rowsinputWithoutBias << endl;
	}
	//cout<< " ********* END printing the mean ********* "<< " rows rows "<< rowsinputWithoutBias<<endl;
	return mean;
}
// calculate the standard deviation of each vector and return a 1*N matrix with value of the standard deviation of each column of the input matrix
double** MLP::CalculateStandardDeviation (double** input, int rowsinputWithoutBias, int columnsinputWithoutBias)
{
    double** result=new double*[1];
	result[0]= new double[columnsinputWithoutBias];

    for( int i=0; i<columnsinputWithoutBias; i++)
    {
        result[0][i]=0.0;

    }
	double** mean =calculateMean(input,rowsinputWithoutBias,columnsinputWithoutBias);
	//cout<< " mean o o " << mean[0][0]<<endl;
	//calculate stdDevaition
	//cout<< " ********* printing the standard Deviation ********* "<< endl;
	for(int i=0;i<columnsinputWithoutBias;i++){
		for(int j=0;j<rowsinputWithoutBias;j++){
			result[0][i]+=pow(input[j][i]-mean[0][i],2);
		}
		result[0][i]=result[0][i]/rowsinputWithoutBias;
		//cout<< result[0][i] << " ";
	}
	//cout<< " ********* END printing the standard Deviation ********* "<< endl;

	return result;
}
// normaliwe the matrix by using the chiSquareNormalize function for normalization and not the min max one of the single layer perceptron
double** MLP::chiSquareNormalize(double** inputWithoutBias, int rowsinputWithoutBias, int columnsinputWithoutBias, double** meanLast, double** standardDeviation)
{
    //double** meanLast = calculateMean (inputWithoutBias,rowsinputWithoutBias,columnsinputWithoutBias);
    //double** standardDeviation=CalculateStandardDeviation(inputWithoutBias,rowsinputWithoutBias,columnsinputWithoutBias);
    //cout << "meanLast[0][0]" << meanLast[0][1]<< endl;

    // will contain the normalized matrix but without bias
    double** result;
    result= new double*[rowsinputWithoutBias];
    for (int i=0; i<rowsinputWithoutBias; i++)
    {
        result[i]= new double[columnsinputWithoutBias];
    }
    for (int i=0; i<rowsinputWithoutBias; i++)
    {
        for (int j=0; j<columnsinputWithoutBias; j++)
        {
            result[i][j]= inputWithoutBias[i][j];
        }
    }
    for(int i=0;i<rowsinputWithoutBias;i++){
		for(int j=0;j<columnsinputWithoutBias;j++){
			result[i][j]=(inputWithoutBias[i][j] - meanLast[0][j])/standardDeviation[0][j];
		}
	}
	return result;
}
// delete the memory of the allocated memory especially for the  variables of the MultiLayerPerceptronNetwork
void MLP::deletingMemory(double** weightsAfterHidden, int sizeweightsAfterHidden, double** weights, int sizeweights, double** inputs,int sizeinputs,
                    double** outputs, int sizeoutputs, double** testData, int sizetestData, double** CopytestData, int sizeCopytestData,
                    double* HiddenNeuronActivationArray, double* OutputActivationArray )
{
    //freeing Pointer to Pointer (2d array)
    for(int i=0;i<sizeweightsAfterHidden;i++)
        delete [] weightsAfterHidden[i];
    delete [] weightsAfterHidden;

    for(int i=0;i<sizeweights;i++)
        delete [] weights[i];
    delete [] weights;

    for(int i=0;i<sizeinputs;i++)
        delete [] inputs[i];
    delete [] inputs;

    for(int i=0;i<sizeoutputs;i++)
        delete [] outputs[i];
    delete [] outputs;

    for(int i=0;i<sizetestData;i++)
        delete [] testData[i];
    delete [] testData;


    for(int i=0;i<sizeCopytestData;i++)
        delete [] CopytestData[i];
    delete [] CopytestData;


    //freeing Pointer (1d array)
    delete [] HiddenNeuronActivationArray;
    delete [] OutputActivationArray;


}
/*
// structure that contains all the parameteers related to the multi layer Perceptron Network
struct MultiLayerPerceptronNetwork
{
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
	MultiLayerPerceptronNetwork(){eita=0; numberOfPatterns; numberOfTestDataPatterns; numberOfInputNeurons=0;numberOfHiddenNeurons=0;}
};
*/
// readingFileSettingParameters() by opening the input file given as standard input, and set all the parameters.
void MLP::readingFileSettingParameters(MLP& perceptron)
{

    bool result=true;
    double * extractedInputs;
    double * extractedOutputs;
    double * extractedTestData;
    int RowCounter1=0; // counter used for extractedInputs
	int RowCounter2=0; // counter used for extractedOutputs
    int RowCounter3=0; // counter used for extractedTestData
    std::string::size_type sz;
///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
///////////Parse line by line the given file as input from command line irisData.txt///////////
// Extracting training input data, output data, test data and many values like number /////////
/////////// of inputs and outputs neuron and number of training data patterns etc ..///////////
///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
    int i=1; // line counter
    for (std::string line; std::getline(std::cin, line);){
	// extract information from second line
	if(i==1){
        perceptron.numberOfInputNeurons = atoi(line.c_str());
	}
	//extract the number of hidden neurons
	else if(i==2){
        perceptron.numberOfHiddenNeurons = atoi(line.c_str());
	}
	else if (i==3){
	string tempLine=line;
	string delimeter = " ";
	size_t posDelimeter = 0;
	int count=0;
	std::string temp;
	std::stringstream iss(tempLine);
	while(iss.good()){
        getline(iss,temp,' ');
        if(count==0){
            // number of perceptron.numberOfPatterns
            perceptron.numberOfPatterns=atoi(temp.c_str());
        }
        else if(count==1){
            //number of output nembers
            perceptron.numberOfOutputNeurons=atoi(temp.c_str());

        }
        count=count+1;
	}

	perceptron.numberOfOutputNeurons = perceptron.numberOfOutputNeurons - perceptron.numberOfInputNeurons;
	// we add the bias the the input neurons
	perceptron.numberOfInputNeurons = perceptron.numberOfInputNeurons +1;
	// create and initialise inputs of the perceptron
	perceptron.inputs= new double*[perceptron.numberOfPatterns];
	for (int i=0;i<perceptron.numberOfPatterns;i++){
        perceptron.inputs[i] = new double[perceptron.numberOfInputNeurons];
        for (int j=0;j<perceptron.numberOfInputNeurons;j++){
            // the columns of the bias with -1
            if(j==perceptron.numberOfInputNeurons-1){
                perceptron.inputs[i][j]=-1;
            }
            // initialise all the inputs to 0 so that we set them later
            else{
                perceptron.inputs[i][j]=0;
            }
        }
	} // end initilisation input matrix

	// create and initialise inputs of the perceptron
	perceptron.outputs= new double*[perceptron.numberOfPatterns];
	for (int i=0;i<perceptron.numberOfPatterns;i++){
        perceptron.outputs[i] = new double[perceptron.numberOfOutputNeurons];
        for (int j=0;j<perceptron.numberOfOutputNeurons;j++){
            // initialise all the inputs to 0 so that we set them later
            perceptron.outputs[i][j]=0;
        }
	} // end initilisation of output matrix
	// initialise double * extractedInputs and double * extractedOutputs;
	int numberOfExtractedInputs=(perceptron.numberOfInputNeurons-1)*perceptron.numberOfPatterns;
	extractedInputs= new double[numberOfExtractedInputs];
	int numberOfRowsExtractedOutputs=(perceptron.numberOfOutputNeurons)*perceptron.numberOfPatterns;
	extractedOutputs= new double[numberOfRowsExtractedOutputs];
	//cout << "perceptron.numberOfInputNeurons = "<< perceptron.numberOfInputNeurons<< endl;
	//cout << "perceptron.numberOfOutputNeurons = "<< perceptron.numberOfOutputNeurons<< endl;
	//cout<<  "numberOfRowsHiddenNeurons = " << perceptron.numberOfHiddenNeurons<<endl;
	//cout << "perceptron.numberOfPatterns = "<< perceptron.numberOfPatterns<< endl;
	//cout<<  "numberOfExtractedInputs = " << numberOfExtractedInputs<<endl;
	//cout<<  "numberOfRowsExtractedOutputs = " << numberOfRowsExtractedOutputs<<endl;
	}
	//////////////////////
	// from line 3 to 2+perceptron.numberOfPatterns+1
	//////////////////////
	else if (i>3 && i<3+perceptron.numberOfPatterns+1){
	// extracting inputs
        int count=0;
        string firstPlace; string secondPlace; string thirdPlace; string fourthPlace;
        string fifthPlace;  string sixthPlace;  string seventhPlace;
        string tempLine=line;
        string temp;
        stringstream iss(tempLine);
        while(iss.good()){
            getline(iss,temp,' ');
            if(count<perceptron.numberOfInputNeurons-1){
                extractedInputs[RowCounter1]=atof(temp.c_str());
                RowCounter1=RowCounter1+1;
            }
            else if(count>=perceptron.numberOfInputNeurons-1){
                extractedOutputs[RowCounter2]=atof(temp.c_str());
                RowCounter2=RowCounter2+1;
            }
            count=count+1;
        }

 	}////////////
 	////End parsing from 3 to 2+perceptron.numberOfPatterns+1
 	/////////////
 	////////////
 	//// line 2+perceptron.numberOfPatterns+1 of the file
 	////////////
 	else if(i== 3+perceptron.numberOfPatterns+1){
        string tempLine1=line;
        string delimeter = " ";
        size_t posDelimeter = 0;
        int count=0;
        // number of perceptron.numberOfPatterns
        string firstPlace;
        //number of output nembers
        string secondPlace;

        std::string temp;
        std::stringstream iss(tempLine1);
        while(iss.good()){
            getline(iss,temp,' ');
            if(count==0){
                firstPlace=temp;
            }
            else if(count==1){
                secondPlace=temp;
            }
            count=count+1;
        }
        // number of perceptron.numberOfPatterns
        perceptron.numberOfTestDataPatterns=atoi(firstPlace.c_str());
        int numberOfRowsExtractedTestDate=perceptron.numberOfTestDataPatterns * perceptron.numberOfInputNeurons;
        extractedTestData= new double[numberOfRowsExtractedTestDate];
	}////////////
	////
 	//// line 2+perceptron.numberOfPatterns+1 of the file
 	////////////
 	////////////
    //// BEGIN EXTRACT TEST DATE
 	//// lines from  2+perceptron.numberOfPatterns+2 to 2+perceptron.numberOfPatterns+2+perceptron.numberOfTestDataPatterns+1
 	////////////
 	else if(i>3+perceptron.numberOfPatterns+1 && i<3+perceptron.numberOfPatterns+1+perceptron.numberOfTestDataPatterns+1 ){
        int count=0;
        string firstPlace; string secondPlace; string thirdPlace; string fourthPlace;
        string tempLine=line;
        std::string temp;
        std::stringstream iss(tempLine);
        while(iss.good()){
            getline(iss,temp,' ');
            if(count<perceptron.numberOfInputNeurons-1){
                firstPlace=temp;
                extractedTestData[RowCounter3]=atof(temp.c_str());
                RowCounter3=RowCounter3+1;
            }
        }

        count=count+1;
    }////////////
 	//// lines from  2+perceptron.numberOfPatterns+2 to 2+perceptron.numberOfPatterns+2+perceptron.numberOfTestDataPatterns+1
 	////////////
 	else{
        break;
	}
        i=i+1;
}/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
//////////// END PARSING OF THE FILE OF THE IRIS DATA  //////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

    // setting the values of the output matrix fro mthe  extractedOutputs from file
    //cout<< "**************** Begin Setting perceptron.outputs[][] ****************"<<endl;
    int counter1=0;
    for(int i=0; i<perceptron.numberOfPatterns;i++){
        for(int j=0; j<perceptron.numberOfOutputNeurons;j++){
            perceptron.outputs[i][j]=extractedOutputs[counter1];
            counter1++;
        }
    }
    //cout<< "**************** end Setting perceptron.outputs[][] ****************"<<endl;

    // setting the values of the input matrix fro mthe  extractedOutputs from file
    //cout<< "**************** Begin Setting perceptron.inputs[][] ****************"<<endl;
    int counter2=0;
    for(int i=0; i<perceptron.numberOfPatterns;i++){
        for(int j=0; j<perceptron.numberOfInputNeurons-1;j++){
            perceptron.inputs[i][j]=extractedInputs[counter2];
            counter2++;
        }
    }
    //cout<< "**************** end Setting perceptron.inputs[][] ****************"<<endl;

    //cout<< "************* Begin Setting perceptron.testData[][] ****************"<<endl;
    // create and initialise inputs of the perceptron
    perceptron.testData= new double*[perceptron.numberOfTestDataPatterns];
    perceptron.CopytestData= new double*[perceptron.numberOfTestDataPatterns];
    for (int i=0;i<perceptron.numberOfTestDataPatterns;i++){
        perceptron.testData[i] = new double[perceptron.numberOfInputNeurons];
        perceptron.CopytestData[i] = new double[perceptron.numberOfInputNeurons];
        for (int j=0;j<perceptron.numberOfInputNeurons;j++){
            // the columns of the bias with -1
            if(j==perceptron.numberOfInputNeurons-1){
                perceptron.testData[i][j]=-1;
            }
            // initialise all the inputs Of TestData to 0 so that we set them later
            else{
                perceptron.testData[i][j]=0;
            }
        }
    } // end initilisation testData matrix
    // setting the values of the Test Data matrix from the  extractedTestData from file
    int counter3=0;
    for(int i=0; i<perceptron.numberOfTestDataPatterns;i++){
        for(int j=0; j<perceptron.numberOfInputNeurons-1;j++){
            // setting the test data values from the extractedData other than the bias part
            perceptron.testData[i][j]=extractedTestData[counter3];
            perceptron.CopytestData[i][j]=extractedTestData[counter3];
            counter3++;
        }
    }
    //cout<< "**************** end Setting perceptron.testData[][] ****************"<<endl;

    //return perceptron;
    //return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////// MAIN PROGRAM //////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
int main (int argc, char **argv)
{
    cout<< "BEGIN TESTING" << endl;
    //cout<< "beginning of MLP CODE" << endl;
	MLP perceptron ;
	readingFileSettingParameters(perceptron);

    // initialise the weights[5][5] with random values between .1 and .5
    perceptron.weights= RandomDoubleWeights(0.1,0.5,perceptron.numberOfInputNeurons,perceptron.numberOfHiddenNeurons);
    // print all the weights for MLP
    //cout<<" ********************** print 1st in initialisation perceptron.weights **********************"<<endl;
    for(int i=0; i<perceptron.numberOfInputNeurons ; i++){
        for (int ii=0;ii<perceptron.numberOfHiddenNeurons;ii++){
            //cout<< perceptron.weights[i][ii] <<" | ";
        }
        //cout<< " " << endl;
    }
    //cout<<" ********************** end print 1st in initialisation perceptron.weights **********************"<<endl;

    // initialise the weightsAfterHidden[5][3] with random values between .1 and .5
    perceptron.weightsAfterHidden= RandomDoubleWeights(0.1,0.5,perceptron.numberOfHiddenNeurons+1,perceptron.numberOfOutputNeurons);
    //cout<<"********************** print weightsAfterHidden in initialisation perceptron.weightsAfterHidden **********************"<<endl;
    for(int i=0; i<perceptron.numberOfHiddenNeurons; i++){
        for (int ii=0;ii<perceptron.numberOfOutputNeurons;ii++){
            //cout<< perceptron.weightsAfterHidden[i][ii] <<" | ";
        }
        //cout<< " " << endl;
    }
    //cout<<" ********************** end print weightsAfterHidden in initialisation  perceptron.weightsAfterHidden **********************"<<endl;

    // initialise learning rate equal to 0.01
    perceptron.eita=0.1;
    // initialise all the 1D array of containing neurons activation status to 0.
    perceptron.HiddenNeuronActivationArray = InitialiseActivationArray(perceptron.numberOfHiddenNeurons);
    // initialise the outputActivationArray status to 0.
    perceptron.OutputActivationArray = InitialiseActivationArray(perceptron.numberOfOutputNeurons);

    //cout<<"************************** Printing of before Normalisation INPUT Data with BIAS ****************************"<<endl;
    for (int k=0;k<perceptron.numberOfPatterns;k++){
        for(int l=0;l<perceptron.numberOfInputNeurons;l++){
            //cout<< " " << perceptron.inputs[k][l] << "|";
        }
        //cout<< endl;
    }
    //cout<<"********************** End Printing before Normalisation INPUT Data with BIAS***************************** " << endl;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////// Normalizing input Data ////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // used to hold the min of training data per input. (for every column)
    double * minArrayInputs = new double [perceptron.numberOfInputNeurons-1]; // -1 because we wont't normalise the bias

    // used to hold the max of training data per input. (for every column)
    double * maxArrayInputs = new double [perceptron.numberOfInputNeurons-1]; // -1 because we wont't normalise the bias

    for(int i=0;i< perceptron.numberOfInputNeurons-1; i++){
        maxArrayInputs[i]=perceptron.inputs[0][i];
        minArrayInputs[i]=perceptron.inputs[0][i];
    }
    //finding the smallest value and the biggest value from all the entries per input
    for (int i=0;i<perceptron.numberOfPatterns;i++) {
        for(int j=0;j<perceptron.numberOfInputNeurons-1;j++){
            if (perceptron.inputs[i][j] > maxArrayInputs[j]){
                maxArrayInputs[j]= perceptron.inputs[i][j];
               //cout<< "maxArrayInputs[j] " << maxArrayInputs[j];
            }
            else if(perceptron.inputs[i][j] < minArrayInputs[j]){
                minArrayInputs[j]= perceptron.inputs[i][j];
                //cout<< "minArrayInputs[j] " << minArrayInputs[j];
            }
        }
    }
    // chiSquareNormalize normalisation of perceptron.input
    // calculating the mean matrix
    double ** meanInput = calculateMean(perceptron.inputs,perceptron.numberOfPatterns, perceptron.numberOfInputNeurons-1);
    // calculate matrix with standard deviation of the chisquare normalization
    double ** standardDeviation = CalculateStandardDeviation(perceptron.inputs,perceptron.numberOfPatterns, perceptron.numberOfInputNeurons-1) ;
    // chiSquareNormalize
    perceptron.inputs= chiSquareNormalize (perceptron.inputs, perceptron.numberOfPatterns,perceptron.numberOfInputNeurons-1,meanInput, standardDeviation);
    //adding the bias because it was removed in the last ligne
    for (int k=0;k<perceptron.numberOfPatterns;k++){
        perceptron.inputs[k][perceptron.numberOfInputNeurons-1]=-1;
    }
    //cout<<"************************** Printing of Normalised INPUT Data with BIAS ****************************"<<endl;
    for (int k=0;k<perceptron.numberOfPatterns;k++){
        for(int l=0;l<perceptron.numberOfInputNeurons;l++){
            //cout<< " " << perceptron.inputs[k][l] << "|";
        }
        //cout<< endl;
    }
    //cout<<"********************** End Printing of Normalised INPUT Data with BIAS***************************** " << endl;


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////// //////////////////////////////////////////////////
    ////////////////////////////////////////////////////////// Learning part ////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    int numberOfIteration=0;
    bool allCorrect=false;
    // error at output neurons errOutput
    double * errOutput = new double[perceptron.numberOfOutputNeurons];
    // error at hidden neurons errHidden
    double * errHidden = new double[perceptron.numberOfHiddenNeurons];
    for(int i=0;i<perceptron.numberOfOutputNeurons;i++){
        errOutput[i] = 0;
    }
    for(int i=0;i<perceptron.numberOfHiddenNeurons;i++){
        errHidden[i] = 0;
    }

    for(int numbIter=0; numbIter<500000; numbIter++){
        bool error=false;
        // Forward Phase
        for(int i=0; i<perceptron.numberOfPatterns ; i++){
            // set output values which are the HiddenNeuronActivationArray for every hidden neuron and OutputActivationArray for every output neuron
            //that will be later are equal to sum Wij*xj applied later with sigmoid function
            perceptron.HiddenNeuronActivationArray=InitialiseActivationArray(perceptron.numberOfHiddenNeurons+1);
            perceptron.OutputActivationArray=InitialiseActivationArray(perceptron.numberOfOutputNeurons);

            //////////////////////////////////
            // feed forward phase from input neurons till hidden neurons
            //////////////////////////////////
            for(int k=0;k<perceptron.numberOfHiddenNeurons;k++){
                for (int ii=0;ii<perceptron.numberOfInputNeurons;ii++){
                    perceptron.HiddenNeuronActivationArray[k]=perceptron.HiddenNeuronActivationArray[k]+perceptron.weights[ii][k]*perceptron.inputs[i][ii];
                }
                perceptron.HiddenNeuronActivationArray[k]=SigmoidFunction(perceptron.HiddenNeuronActivationArray[k]);
            }

            //////////////////////////////////
            // Bias added before outputs
            //////////////////////////////////
            perceptron.HiddenNeuronActivationArray[perceptron.numberOfHiddenNeurons+1]=-1;


            //////////////////////////////////
            // feed forward phase from hidden neurons till output neurons
            //////////////////////////////////
            for(int k=0;k<perceptron.numberOfOutputNeurons;k++){
                for(int ii=0;ii<perceptron.numberOfHiddenNeurons+1;ii++){
                    perceptron.OutputActivationArray[k]=perceptron.OutputActivationArray[k]+perceptron.weightsAfterHidden[ii][k]*perceptron.HiddenNeuronActivationArray[ii];
                }
                perceptron.OutputActivationArray[k]=SigmoidFunction(perceptron.OutputActivationArray[k]);
            }


            //////////////////////////////////
            //Error at the output neurons
            //////////////////////////////////
            for(int j=0;j<perceptron.numberOfOutputNeurons;j++){
                    errOutput[j]= (perceptron.OutputActivationArray[j] - perceptron.outputs[i][j]) * (perceptron.OutputActivationArray[j]) * (1 - perceptron.OutputActivationArray[j]) ;
            }
            double sum=0;


            //////////////////////////////////
            //// Error at the hidden neurons
            //////////////////////////////////
            for(int j=0;j<perceptron.numberOfHiddenNeurons;j++){
                for(int k=0;k<perceptron.numberOfInputNeurons;k++){
                    for(int h=0;h<perceptron.numberOfOutputNeurons;h++){
                        sum=sum + ((errOutput[h]) * perceptron.weightsAfterHidden[j][k] );
                    }
                }
                errHidden[j]= perceptron.HiddenNeuronActivationArray[j] * (1-perceptron.HiddenNeuronActivationArray[j]) * sum;
                sum=0;
            }


            //////////////////////////////////
            //// update Output Layer Weights
            //////////////////////////////////
            for(int j=0;j<perceptron.numberOfOutputNeurons;j++){
                    for (int k=0;k<perceptron.numberOfHiddenNeurons+1;k++){
                    // update each of the weights individually
                        perceptron.weightsAfterHidden[k][j]= perceptron.weightsAfterHidden[k][j] - perceptron.eita * errOutput[j]* perceptron.HiddenNeuronActivationArray[k];
                    }
            }


            //////////////////////////////////
            //// update hidden Layer Weights
            //////////////////////////////////
            for(int j=0;j<perceptron.numberOfHiddenNeurons;j++){
                    for (int k=0;k<perceptron.numberOfInputNeurons;k++){
                        perceptron.weights[k][j]= perceptron.weights[k][j] - perceptron.eita * errHidden[j]* perceptron.inputs[i][k];
                    }
            }
       }
       numberOfIteration = numberOfIteration + 1;
    }
    //cout<< "Total Number Of Iterations = " << numberOfIteration << endl;
    //////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////End of Learning part///////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////


    //cout<< " ////////////////////////////////////////////////// BEGIN TESTING ////////////////////////////////////////////////////////////" << endl;


    //cout<<"**************************** Printing BEFORE Normalising Test DATA with Bias********************************* " << endl;
    for (int k=0;k<perceptron.numberOfTestDataPatterns;k++){
        // not normalizing the bias input
        for(int l=0;l<perceptron.numberOfInputNeurons;l++){
            //cout << perceptron.testData[k][l]<< "|" ;
        }
        //cout<<endl;
    }
    //cout<<"**************************** END Printing BEFORE Normalising Test DATA with Bias********************************* " << endl;

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////// chiSquareNormalize normalisation of perceptron.testData  ///////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // calculating the mean matrix of perceptron.testData
    double** meanInputTest = calculateMean(perceptron.testData, perceptron.numberOfTestDataPatterns, perceptron.numberOfInputNeurons-1);

    // calculate matrix with standard deviation of the chisquare normalization for perceptron.testData
    double ** standardDeviationTest = CalculateStandardDeviation(perceptron.testData, perceptron.numberOfTestDataPatterns, perceptron.numberOfInputNeurons-1) ;

    // chiSquareNormalize perceptron.testData
    perceptron.testData= chiSquareNormalize (perceptron.testData, perceptron.numberOfTestDataPatterns,
                                            perceptron.numberOfInputNeurons-1,meanInput, standardDeviation);
    //adding the bias because it was removed in the last ligne
    for (int k=0;k<perceptron.numberOfTestDataPatterns;k++){
        perceptron.testData[k][perceptron.numberOfInputNeurons-1]=-1;
    }


    //cout<<"**************************** Printing Normalised Test DATA with Bias********************************* " << endl;
    for (int k=0;k<perceptron.numberOfTestDataPatterns;k++){
        // not normalizing the bias input
        for(int l=0;l<perceptron.numberOfInputNeurons;l++){
            //cout << perceptron.testData[k][l]<< "|" ;
        }
        //cout<<endl;
    }
    //cout<<"**************************** END Printing Normalised Test DATA with Bias********************************* " << endl;

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////// END Normalize perceptron.testData //////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////  TESTING THE NEURAL NETWORK WITH THE DEFINED NEURAL NETWORK //////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    double ** OutputTest=new double*[perceptron.numberOfTestDataPatterns];
    double ** FinalOutputTest = new double*[perceptron.numberOfTestDataPatterns];
    for (int i=0;i<perceptron.numberOfTestDataPatterns;i++){
        OutputTest[i] = new double[perceptron.numberOfHiddenNeurons+1];
        for(int j=0;j<perceptron.numberOfHiddenNeurons;j++){
            OutputTest[i][j]=0;
        }
    }

    for (int i=0;i<perceptron.numberOfTestDataPatterns;i++){
        FinalOutputTest[i] = new double[perceptron.numberOfOutputNeurons];
        for(int j=0;j<perceptron.numberOfOutputNeurons;j++){
            FinalOutputTest[i][j]=0;
        }
    }

    // output of the first hidden layer
    for(int r=0; r<perceptron.numberOfTestDataPatterns ; r++){
        for (int l=0;l<perceptron.numberOfHiddenNeurons;l++){
            for (int i=0;i<perceptron.numberOfInputNeurons;i++){
            OutputTest[r][l] = OutputTest[r][l] + perceptron.weights[i][l]*perceptron.testData[r][i];
            }
        }
    }

    for(int r=0; r<perceptron.numberOfTestDataPatterns ; r++){
            OutputTest[r][perceptron.numberOfHiddenNeurons] =-1 ;
    }

    // output of the output lqyer of MLP network
    for(int r=0; r<perceptron.numberOfTestDataPatterns ; r++){
        for (int l=0;l<perceptron.numberOfOutputNeurons;l++){
            for (int i=0;i<perceptron.numberOfHiddenNeurons+1;i++){
                FinalOutputTest[r][l] = FinalOutputTest[r][l] + perceptron.weightsAfterHidden[i][l]*SigmoidFunction(OutputTest[r][i]);
            }
        }
    }


    ///////////////////////////////////////////////////////////////////////////////////////////
    ////// print all the first weights between the input layer and the hidden layer for MLP
    ///////////////////////////////////////////////////////////////////////////////////////////
    //cout<<"+++++++++++++ print 1st weights after +++++++++++"<<endl;
    for(int i=0; i<perceptron.numberOfInputNeurons ; i++){
        for (int ii=0;ii<perceptron.numberOfHiddenNeurons;ii++){
            //cout<< perceptron.weights[i][ii] <<" | ";
        }
        //cout<< " " << endl;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // print all the weights between the Hidden Layer and the output layer
    ///////////////////////////////////////////////////////////////////////////////////////////
    //cout<<"+++++++++++++ print 2nd weights after +++++++++++"<<endl;
    for(int i=0; i<perceptron.numberOfHiddenNeurons+1 ; i++){
        for (int ii=0;ii<perceptron.numberOfOutputNeurons;ii++){
            //cout<< perceptron.weightsAfterHidden[i][ii] <<" | ";
        }
        //cout<< " " << endl;
    }

    cout<<std::setprecision(0)<<std::fixed;
    // print test data then calculated test data in the same line for MLP
    //cout<<"+++++++++++++ print outputs Test+++++++++++"<<endl;
    for(int i=0; i<perceptron.numberOfTestDataPatterns ; i++){
        for(int j=0; j<perceptron.numberOfInputNeurons-1 ; j++){
            cout<< std::setprecision(1) <<perceptron.CopytestData[i][j]<<" ";
        }
        for(int k=0; k<perceptron.numberOfOutputNeurons ; k++){
            cout<< std::setprecision(0) << SigmoidFunction(FinalOutputTest[i][k])<<" ";
        }
        cout<< " " << endl;
    }


    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////// FREEING MEMORY ///////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    deletingMemory(perceptron.weightsAfterHidden, perceptron.numberOfHiddenNeurons+1, perceptron.weights, perceptron.numberOfInputNeurons, perceptron.inputs, perceptron.numberOfPatterns,
                        perceptron.outputs, perceptron.numberOfPatterns, perceptron.testData, perceptron.numberOfTestDataPatterns, perceptron.CopytestData, perceptron.numberOfTestDataPatterns,
                        perceptron.HiddenNeuronActivationArray, perceptron.OutputActivationArray );


    return 0;
}
*/
