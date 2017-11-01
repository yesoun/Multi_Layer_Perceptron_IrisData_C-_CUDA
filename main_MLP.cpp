/* Author: Yassine Maalej (Github: yesoun) 
   email: maalej.yessine@gmail.com && maal4948@vandals.uidaho.edu
   Date: Frebuary 2016
   Class: CS541 Advanced Operating Systems
   Insitution: University Of Idaho
*/
#include <iostream> // for std:: cout and std::cin
#include <fstream> // input/output stream class to operate on files
#include <stdlib.h> 
#include <math.h> 
#include <vector> 
#include <iterator>
#include <string>
#include <sstream> // Stream Class to Operate on strings and objects of this class use a string buffer that contains a sequence of characters.
#include <iomanip> // for the std::setprecision that is used to limitate prinitin to 2 digits in output of tested data
#include "MLP.h"

using namespace std;


int main (int argc, char **argv)
{
    cout<< "BEGIN TESTING" << endl;
    //cout<< "beginning of MLP CODE" << endl;
	MLP perceptron ;
	perceptron.readingFileSettingParameters(perceptron);

    // initialise the weights[5][5] with random values between .1 and .5
    perceptron.weights= perceptron.RandomDoubleWeights(0.1,0.5,perceptron.numberOfInputNeurons,perceptron.numberOfHiddenNeurons);

    // initialise the weightsAfterHidden[5][3] with random values between .1 and .5
    perceptron.weightsAfterHidden= perceptron.RandomDoubleWeights(0.1,0.5,perceptron.numberOfHiddenNeurons+1,perceptron.numberOfOutputNeurons);

    // initialise learning rate equal to 0.01
    perceptron.eita=0.1;
    // initialise all the 1D array of containing neurons activation status to 0.
    perceptron.HiddenNeuronActivationArray = perceptron.InitialiseActivationArray(perceptron.numberOfHiddenNeurons);
    // initialise the outputActivationArray status to 0.
    perceptron.OutputActivationArray = perceptron.InitialiseActivationArray(perceptron.numberOfOutputNeurons);


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
    double ** meanInput = perceptron.calculateMean(perceptron.inputs,perceptron.numberOfPatterns, perceptron.numberOfInputNeurons-1);
    // calculate matrix with standard deviation of the chisquare normalization
    double ** standardDeviation = perceptron.CalculateStandardDeviation(perceptron.inputs,perceptron.numberOfPatterns, perceptron.numberOfInputNeurons-1) ;
    // chiSquareNormalize
    perceptron.inputs= perceptron.chiSquareNormalize (perceptron.inputs, perceptron.numberOfPatterns,perceptron.numberOfInputNeurons-1,meanInput, standardDeviation);
    //adding the bias because it was removed in the last ligne
    for (int k=0;k<perceptron.numberOfPatterns;k++){
        perceptron.inputs[k][perceptron.numberOfInputNeurons-1]=-1;
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////// ////////////////////////////////////////////////
    ////////////////////////////////////////////////////////// Learning part /////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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

    for(int numbIter=0; numbIter<5000000; numbIter++){
        bool error=false;
        // Forward Phase
        for(int i=0; i<perceptron.numberOfPatterns ; i++){
            // set output values which are the HiddenNeuronActivationArray for every hidden neuron and OutputActivationArray for every output neuron
            //that will be later are equal to sum Wij*xj applied later with sigmoid function
            perceptron.HiddenNeuronActivationArray= perceptron.InitialiseActivationArray(perceptron.numberOfHiddenNeurons+1);
            perceptron.OutputActivationArray= perceptron.InitialiseActivationArray(perceptron.numberOfOutputNeurons);

            //////////////////////////////////
            // feed forward phase from input neurons till hidden neurons
            //////////////////////////////////
            for(int k=0;k<perceptron.numberOfHiddenNeurons;k++){
                for (int ii=0;ii<perceptron.numberOfInputNeurons;ii++){
                    perceptron.HiddenNeuronActivationArray[k]=perceptron.HiddenNeuronActivationArray[k]+perceptron.weights[ii][k]*perceptron.inputs[i][ii];
                }
                perceptron.HiddenNeuronActivationArray[k]=perceptron.SigmoidFunction(perceptron.HiddenNeuronActivationArray[k]);
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
                perceptron.OutputActivationArray[k]=perceptron.SigmoidFunction(perceptron.OutputActivationArray[k]);
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
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////// End of Learning part //////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////





    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////// chiSquareNormalize normalisation of perceptron.testData  ///////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // calculating the mean matrix of perceptron.testData
    double** meanInputTest = perceptron.calculateMean(perceptron.testData, perceptron.numberOfTestDataPatterns, perceptron.numberOfInputNeurons-1);

    // calculate matrix with standard deviation of the chisquare normalization for perceptron.testData
    double ** standardDeviationTest = perceptron.CalculateStandardDeviation(perceptron.testData, perceptron.numberOfTestDataPatterns, perceptron.numberOfInputNeurons-1) ;

    // chiSquareNormalize perceptron.testData
    perceptron.testData= perceptron.chiSquareNormalize (perceptron.testData, perceptron.numberOfTestDataPatterns,
                                            perceptron.numberOfInputNeurons-1,meanInput, standardDeviation);
    //adding the bias because it was removed in the last ligne
    for (int k=0;k<perceptron.numberOfTestDataPatterns;k++){
        perceptron.testData[k][perceptron.numberOfInputNeurons-1]=-1;
    }

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
                FinalOutputTest[r][l] = FinalOutputTest[r][l] + perceptron.weightsAfterHidden[i][l]*perceptron.SigmoidFunction(OutputTest[r][i]);
            }
        }
    }


    cout<<std::setprecision(0)<<std::fixed;
    // print test data then calculated test data in the same line for MLP
    //cout<<"+++++++++++++ print outputs Test+++++++++++"<<endl;
    for(int i=0; i<perceptron.numberOfTestDataPatterns ; i++){
        for(int j=0; j<perceptron.numberOfInputNeurons-1 ; j++){
            cout<< std::setprecision(1) <<perceptron.CopytestData[i][j]<<" ";
        }
        for(int k=0; k<perceptron.numberOfOutputNeurons ; k++){
            cout<< std::setprecision(0) << perceptron.SigmoidFunction(FinalOutputTest[i][k])<<" ";
        }
        cout<< " " << endl;
    }


    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////// FREEING MEMORY ///////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    perceptron.deletingMemory(perceptron.weightsAfterHidden, perceptron.numberOfHiddenNeurons+1, perceptron.weights, perceptron.numberOfInputNeurons, perceptron.inputs, perceptron.numberOfPatterns,
                        perceptron.outputs, perceptron.numberOfPatterns, perceptron.testData, perceptron.numberOfTestDataPatterns, perceptron.CopytestData, perceptron.numberOfTestDataPatterns,
                        perceptron.HiddenNeuronActivationArray, perceptron.OutputActivationArray );


    return 0;
}

