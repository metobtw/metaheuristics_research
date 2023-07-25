#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
#include <random>
#include <vector>

using namespace std;

vector <int> generate_blocks(int size){
    vector <int> permutation;
    for (int i = 0; i < size; i++) {
        permutation.push_back(i);
    }   
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(permutation.begin(), permutation.end(), gen);
    return permutation;
}

int main() {
    //opening file 
    ifstream inputFile("/home/meto/cpp_tests/to_embed.txt");
    int ind_information = 0;
    string information;
    getline(inputFile,information);
    inputFile.close();
    
    //opening image
    cv::Mat image = cv::imread("/home/meto/cpp_tests/lena512.png", cv::IMREAD_GRAYSCALE);
    int rows = image.rows;
    int cols = image.cols;
    std::vector<std::vector<int>> img(rows, std::vector<int>(cols));

    // Copy pixel values from the OpenCV image to the 2D vector
    for (int i = 0; i < rows; ++i) 
        for (int j = 0; j < cols; ++j) 
            img[i][j] = static_cast<int>(image.at<uchar>(i, j));

    //generate blocks, saving to txt
    vector <int> blocks = generate_blocks(rows*cols);
    std::ofstream outputFile("/home/meto/cpp_tests/blocks.txt");
    for (int num: blocks)
        outputFile << num << ' ';
    outputFile.close();

    int cnt = 0;
    for (int i : blocks){
        int block_w = i % (rows / 8);
        int block_h = (i - block_w) / (rows / 8);
        vector <vector<int>> pixel_matrix;
        for (int i1=block_h * 8 ;i<block_h * 8 +8;i1++){
            pixel_matrix.push_back(vector<int>());
            for (int i2=block_w * 8 ;i2<block_w * 8 +8;i2++){
                pixel_matrix[i1].push_back(img[i1][i2]);
            }
        }
        for (int i1= 0;i1<8;i1++)
        for (int j1=0;j1<8;j1++)
        cout << pixel_matrix[i1][j1];
        break;
    }
}

