#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <eigen3/Eigen/Sparse>
#include <sys/time.h>
#include <pthread.h>
#include <thread>
using namespace std;

//struct for NLR matrix
struct csr_1{
    vector<double> values;
    vector<int> rowPtrs;
    vector<int> cols;

    //The following vector is for testing correctness
    int numRows;
    int numCols;
    vector<int> rows;
    //1-D dense vector
    vector<double> vectorValues;
};

vector<double> resultParallel;
csr_1 *csr = new csr_1;
int NUM_THREADS = thread::hardware_concurrency();

void readFiles(string fileName){
    ifstream inputFile;
    inputFile.open(fileName);

    string line;
    bool gotFirstLine = false;
    int rows;
    int cols;
    int numVals;
    //create the struct that will store the sparse matrix data
    //csr_1 *csr = new csr_1;

    //also create a dictionary to map row to column
    cout << "Reading matrix from " << fileName << "..." << endl;
    map<int, vector<int>> row2col;
    while(getline(inputFile, line)){
        if(line.size() == 0 or line[0] == '%') continue;    //don't read comment lines

        if(!gotFirstLine){  //read the rows, cols, numvals from the first line in the file
            stringstream ss(line);
            ss >> rows;
            ss >> cols;
            ss >> numVals;
            gotFirstLine = true;

            csr->numRows = rows;
            csr->numCols = cols;
        } else {
            int row;
            int col;
            stringstream ss(line);

            ss >> row;
            ss >> col;

            row2col[row-1].push_back(col-1);
        }
    }
    inputFile.close();

    //create the correct rows vector in csr from rowCounts
    cout << "Populating CSR data structure..." << endl;

    csr->rowPtrs.push_back(0);
    for(int i = 0; i < rows; i++){
        double randNum = 0;
        for(int j = 0; j < row2col[i].size(); j++){ //loop through the vector of columns with values in the current row
             randNum = (4.9 - 0.1) * ((double)rand() / (double)RAND_MAX) + 0.1;
            csr->values.push_back(randNum);

            csr->cols.push_back(row2col[i][j]);
            csr->rows.push_back(i);
        }
        randNum = (4.9 - 0.1) * ((double)rand() / (double)RAND_MAX) + 0.1;
        csr->vectorValues.push_back(randNum);

        csr->rowPtrs.push_back(csr->rowPtrs[i] + row2col[i].size());    //rowPtrs[i] = rowPtrs[i-1] + num values in row i
    }
}

void* parallelMultiply(void* threadID){
    int tid = (long)threadID;
    int startRow = (int)((csr->numRows / (double)NUM_THREADS) * tid);
    int endRow = (int)((csr->numRows / (double)NUM_THREADS) * (tid+1));

    for(int i = startRow; i < endRow; i++){
        for (int j = csr->rowPtrs[i]; j < csr->rowPtrs[i+1]; j++){
            resultParallel[i] += csr->values[j] * csr->vectorValues[csr->cols[j]];
        }
    }
}

vector<double> sparseMatrixMultiplication(){

    resultParallel.resize(csr->numRows);

    // declaring four threads
    pthread_t threads[NUM_THREADS];

    timespec start;
    timespec end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // create the threads and have them call the function
    for (int i = 0; i < NUM_THREADS; i++) {
        //cout << (int)((csr->numRows / (double)NUM_THREADS) * i) << " : " << (int)((csr->numRows / (double)NUM_THREADS) * (i+1)) << endl;
        pthread_create(&threads[i], NULL, parallelMultiply, (void*)i);
    }
    //wait for all threads to complete
    for (int i = 0; i < NUM_THREADS; i++)  {
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double diff = (1000000000 * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / (double)1000000000;
    cout << endl << "My matrix multiplication: " << diff << " seconds" << endl;


    //2
    clock_gettime(CLOCK_MONOTONIC, &start);
    vector<double> resultParallel2;
    resultParallel2.resize(csr->numRows);
    #pragma omp parallel for
    for(int i = 0; i < csr->numRows; i++){
        for (int j = csr->rowPtrs[i]; j < csr->rowPtrs[i+1]; j++){
            resultParallel2[i] += csr->values[j] * csr->vectorValues[csr->cols[j]];
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    diff = (1000000000 * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / (double)1000000000;
    cout << "OpenMP 'parallel for' directive matrix multiplication: " << diff << " seconds" << endl;

    // ofstream outFile("NREmultiplicationTest.txt", std::ios_base::app);
    // for(int i = 0; i < resultParallel.size(); i++){
    //     outFile << resultParallel[i] << endl;
    // }
    // outFile.close();

    return resultParallel;
}


vector<double> testCorrectness(){

    vector<double> resultTest;

    Eigen::SparseMatrix<double> sparseMatrix(csr->numRows, csr->numCols);
    Eigen::VectorXd denseVector(csr->vectorValues.size());

    vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(csr->values.size());
    //populate Eigen sparse matrix and vector
    for (int i = 0; i < csr->values.size(); i++){
        if(i < csr->vectorValues.size()){
            denseVector(i) = csr->vectorValues[i];
        }
        //sparseMatrix.insert(csr->rows[i], csr->cols[i]) = csr->values[i];
        tripletList.push_back(Eigen::Triplet<double>(csr->rows[i], csr->cols[i], csr->values[i]));
    }
    sparseMatrix.setFromTriplets(tripletList.begin(), tripletList.end());

    //take time of matrix multiplication
    timespec start;
    timespec end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    denseVector = sparseMatrix * denseVector;       //matrix multiplication
    clock_gettime(CLOCK_MONOTONIC, &end);

    for(int i = 0; i < denseVector.rows(); i++){
        resultTest.push_back(denseVector(i));
    }

    double diff = (1000000000 * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / (double)1000000000;
    cout << "Eigen library matrix multiplication: " << diff << " seconds" << endl;

    // const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
    // ofstream outFile("NREmultiplicationCorrect.txt", std::ios_base::app);
    // outFile << denseVector.format(CSVFormat);
    // outFile.close();

    return resultTest;
}

void compareResults(vector<double> test, vector<double> correct){

    cout << endl << "Size of my output matrix: " << test.size() << endl;
    cout << "Size of Eigen output matrix: " << correct.size() << endl;

    int wrongCount = 0;
    for(int i = 0; i < test.size(); i++){
        if(test[i] != correct[i]) wrongCount++;
    }

    cout << "Number of differences between output 1-D vectors: " << wrongCount << endl;
}

int main(){

    vector<double> test;
    vector<double> correct;
    string names[3] = {{"NLR.mtx"}, {"delaunay_n19.mtx"}, {"channel-500x100x100-b050.mtx"}};

    for(int i = 0; i < 3; i++){
        readFiles(names[i]);

        test = sparseMatrixMultiplication();
        correct = testCorrectness();

        compareResults(test, correct);
        cout << endl << "++++++++++++++++++++++++++++++" << endl;

        //free CSR data structure
        csr->values.clear();
        csr->rowPtrs.clear();
        csr->cols.clear();
        csr->rows.clear();
        csr->vectorValues.clear();
        resultParallel.clear();
    }
}
