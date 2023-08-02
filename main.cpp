#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
#include <random>
#include <vector>
#include <cmath>

using namespace std;

vector <vector<int>> ret_idx(){
    vector<vector<int>> IDX = {
        {3, 4}, {2, 5}, {1, 6}, {0, 7},
        {1, 7}, {2, 6}, {3, 5}, {4, 4},
        {5, 3}, {6, 2}, {7, 1}, {7, 2},
        {6, 3}, {5, 4}, {4, 5}, {3, 6},
        {2, 7}, {3, 7}, {4, 6}, {5, 5},
        {6, 4}, {7, 3}, {7, 4}, {6, 5},
        {5, 6}, {4, 7}, {5, 7}, {6, 6},
        {7, 5}, {7, 6}, {6, 7}, {7, 7}
    };
    return IDX;
}

int sign(int x){
    return (x > 0) - (x < 0);
}

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

std::vector<std::vector<double>> do_dct(const std::vector<std::vector<int>>& input) {
    cv::Mat inputMat(input.size(), input[0].size(), CV_32S);
    for (size_t i = 0; i < input.size(); ++i) {
        for (size_t j = 0; j < input[0].size(); ++j) {
            inputMat.at<int>(i, j) = input[i][j];
        }
    }

    // Convert input matrix to CV_32F (float) for DCT computation
    cv::Mat floatInput;
    inputMat.convertTo(floatInput, CV_32F);

    // Perform 2D DCT without normalization
    cv::Mat dctResult;
    cv::dct(floatInput, dctResult);

    // Manual normalization (equivalent to 'norm="ortho"' in SciPy)

    std::vector<std::vector<double>> output(dctResult.rows, std::vector<double>(dctResult.cols));
    for (int i = 0; i < dctResult.rows; ++i) {
        for (int j = 0; j < dctResult.cols; ++j) {
            output[i][j] = dctResult.at<float>(i, j);
        }
    }

    return output;
}

// Function to compute the 2D IDCT on DCT coefficients to recover the original data
std::vector<std::vector<int>> undo_dct(const std::vector<std::vector<double>>& dctCoefficients) {
    cv::Mat dctResult(dctCoefficients.size(), dctCoefficients[0].size(), CV_64FC1);
    for (size_t i = 0; i < dctCoefficients.size(); ++i) {
        for (size_t j = 0; j < dctCoefficients[0].size(); ++j) {
            dctResult.at<double>(i, j) = dctCoefficients[i][j];
        }
    }

    // Perform 2D IDCT without normalization
    cv::Mat idctResult;
    cv::idct(dctResult, idctResult);

    // Convert back to integer data
    std::vector<std::vector<int>> output(idctResult.rows, std::vector<int>(idctResult.cols));
    for (int i = 0; i < idctResult.rows; ++i) {
        for (int j = 0; j < idctResult.cols; ++j) {
            output[i][j] = static_cast<int>(idctResult.at<double>(i, j));
        }
    }

    return output;
}

std::vector<std::vector<double>> embed_to_dct(std::vector<std::vector<double>> dct_matrix, const string bit_string, float q = 20.0){
    int ind = 0;
    std::vector<std::vector<int>> IDX = ret_idx();
    for (const auto& coord : IDX) {
        int i = coord[0], j = coord[1];
        dct_matrix[i][j] = sign(dct_matrix[i][j]) * (q * int(abs(dct_matrix[i][j]) / q) + (q/2) * (int(bit_string[ind]) - int('0')));
        ind += 1;
    }
    return dct_matrix;
}

std::vector<std::vector<float>> generate_population(vector<vector<int>> original,vector<vector<int>> notorig, int population_size = 128,float beta = 0.9,int search_space = 10){
    std::random_device rd;
    std::mt19937 gen(rd());
    double lower_bound = 0.0;
    double upper_bound = 1.0;
    std::uniform_real_distribution<double> uniform_dist(lower_bound, upper_bound);
    std::uniform_int_distribution<int> search_dist(-search_space, search_space);

    vector<float> diff; //flatten
    for (int i = 0; i < original.size(); i++)
        for (int j = 0; j < original[0].size();j++)
            diff.push_back(original[i][j]-notorig[i][j]);
    
    vector<vector<float>> population(population_size-1,vector<float>(diff.size()));
    population.push_back(diff);

    for (int i = 0; i < population_size; i++){
        for (int j = 0; j < diff.size(); j++){
            double random_value = uniform_dist(gen);
            if (random_value > beta){
                int random_search = search_dist(gen); 
                population[i][j] = random_search;
            }
            else{
                population[i][j] = diff[j];
            }
        }
    }
    return population;
}

std::vector<float> meanAlongAxis(const std::vector<std::vector<float>>& array) {
    std::vector<float> meanValues(array[0].size(), 0.0);

    for (int j = 0; j < array[0].size(); j++) {
        double sum = 0.0;
        for (int i = 0; i < array.size(); i++) {
            sum += array[i][j];
        }
        meanValues[j] = sum / array.size();
    }

    return meanValues;
}

class Metric{
    private:
    std::vector<std::vector<int>> block_matrix;
    std::string bit_string;

    public: 
    Metric(const std::vector<std::vector<int>>& block_matrix, const std::string& bit_string)
        : block_matrix(block_matrix), bit_string(bit_string) {}

    std::pair<float, vector<float>> metric(const std::vector<float>& block, int q = 20) {//block float?

        std::vector<vector<int>> new_block = block_matrix;
        vector<float> block_flatten = block; 
        for (int i = 0; i < block_flatten.size(); i++)
            block_flatten[i] = floor(block_flatten[i]);

        int ind_fl = 0;
        for (int i = 0; i < 8; i++){
            for (int j = 0; j < 8; j++){
                new_block[i][j] += block_flatten[ind_fl];
                if (new_block[i][j] > 255){
                    int diff = abs(new_block[i][j]-255);
                    block_flatten[ind_fl] -= diff;
                    new_block[i][j] = 255;
                }
                if (new_block[i][j] < 0){
                    int diff = abs(new_block[i][j]);
                    block_flatten[ind_fl] += diff;
                    new_block[i][j] = 0;
                }
                ind_fl++;
            }
        }
        int sum_elem = 0;
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++)
                sum_elem += pow(block_matrix[i][j] - new_block[i][j],2);
        float psnr = 10 * log10((pow(8,2) * pow(255,2)) / float(sum_elem));
    
        //extraction
        vector <vector<double>> dct_block = do_dct(new_block);
        string s;
        vector<vector<int>> IDX = ret_idx();
        for (const auto& coord : IDX) {
            int i = coord[0], j = coord[1];
            int c0 = sign(dct_block[i][j]) * (q * int(abs(dct_block[i][j]) / q) + (q/2) * (0));
            int c1 = sign(dct_block[i][j]) * (q * int(abs(dct_block[i][j]) / q) + (q/2) * (1));
            if (abs(dct_block[i][j] - c0) < abs(dct_block[i][j] - c1))
                s += '0';
            else
                s += '1';
            if (s[0] != bit_string[0]){
                std::pair<float, vector<float>> to_ret = std::make_pair(0.0, block_flatten);
                return to_ret;
            }
        }
        int cnt = 0;
        for (int i = 0; i < s.length(); i++)
            if (s[i] == bit_string[i])
                cnt += 1;
        
        std::pair<float, vector<float>> to_ret = std::make_pair(psnr/10000 + float(cnt)/float(s.length()), block_flatten);
        return to_ret;
    }
};

std::vector<float> getRandomArray(size_t size, float low, float high) {
    std::vector<float> randomArray;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distribution(low, high);

    for (size_t i = 0; i < size; ++i) {
        randomArray.push_back(distribution(gen));
    }

    return randomArray;
}

std::vector<float> calculateDifference(const std::vector<float>& teacher, const std::vector<float>& population_mean) {
    size_t num_features = teacher.size();
    std::vector<float> difference;
    
    std::vector<float> randomUniform1 = getRandomArray(num_features, 0.0, 1.0);
    std::vector<float> randomUniform2 = getRandomArray(num_features, 1.0, 2.0);
    
    for (size_t i = 0; i < num_features; ++i) {
        difference.push_back(randomUniform1[i] * (teacher[i] - randomUniform2[i] * population_mean[i]));
    }

    return difference;
}

std::vector<float> calculateDifferenceRand(const std::vector<float>& rand1,const std::vector<float>& rand2, const std::vector<float>& population_cur) {
    size_t num_features = population_cur.size();
    std::vector<float> new_population;
    
    std::vector<float> randomUniform1 = getRandomArray(num_features, 0.0, 1.0);
    
    for (size_t i = 0; i < num_features; ++i) {
        new_population.push_back(population_cur[i] + randomUniform1[i] * (rand1[i] - rand2[i]));
    }

    return new_population;
}


int getRandomIndex(int population_size) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<int> uniform_dist(0, population_size - 1);

    return uniform_dist(gen);
}

class TLBO{
    private:
    int population_size;
    int num_iterations;
    int num_features;
    std::vector<std::vector<float>> population; // Reference to the population used for Metric

    public:
    TLBO(const std::vector<std::vector<float>>& initial_population, int population_size, int num_iterations, int num_features)
        : population_size(population_size), num_iterations(num_iterations), num_features(num_features), population(initial_population) {
    }
    std::pair<float, vector<float>> optimize(Metric& obj) { //vector float -> int
        vector<vector<int>> best_solution;
        float best_fitness = 0.0;
        vector<float> fitness;
        for (int i = 0; i < population.size(); i++){
            std::pair<float, vector<float>> pr = obj.metric(population[i]);
            fitness.push_back(pr.first);
        }
        for (int h = 0; h < num_iterations; h++){
            //Teaching phase
            int best_index = 0;
            for (int i = 0; i < fitness.size();i++){
                if (fitness[i] > fitness[best_index])
                    best_index = i;
            }
            vector<float> teacher = population[best_index];
            cout << fitness[best_index];
            vector <float> population_mean = meanAlongAxis(population);

            for (int i = 0; i < population_size;i++){
                if (i != best_index){
                    vector<float> difference = calculateDifference(teacher,population_mean);
                    for (int j = 0; j < difference.size(); j++){
                        difference[j] += population[i][j];
                    }
                    float old_score = fitness[i];
                    std::pair<float,vector<float>> new_sc_d = obj.metric(difference);
                    if (new_sc_d.first > old_score){
                        population[i] = new_sc_d.second;
                        fitness[i] = new_sc_d.first;
                    }
                }
            }

            //learning phase
            for (int i = 0; i < population_size;i++){
                int random_index_1 = 0;
                int random_index_2 = 0;
                while (random_index_1 == random_index_2){
                    random_index_1 = getRandomIndex(population_size);
                    random_index_2 = getRandomIndex(population_size);
                }
                float rand1_sc = fitness[random_index_1],rand2_sc = fitness[random_index_2];
                vector<float> rand1_bl = population[random_index_1], rand2_bl = population[random_index_2];
                vector <float> new_population ;
                if (rand1_sc > rand2_sc){
                    new_population = calculateDifferenceRand(rand1_bl,rand2_bl,population[i]);
                }
                else{
                    new_population = calculateDifferenceRand(rand2_bl,rand1_bl,population[i]);
                }
                float old_score = fitness[i];
                std::pair<float,vector<float>> new_sc_d = obj.metric(new_population);
                if (new_sc_d.first > old_score){
                    population[i] = new_sc_d.second;
                    fitness[i] = new_sc_d.first;
                }  
            }
        }
        


        vector<float> a(10);
        std::pair<float,vector<float>> to_ret = std::make_pair(0.0,a);
        return to_ret;
    }
};

int main() {
    //opening file 
    ifstream inputFile("to_embed.txt");
    int ind_information = 0;
    string information;
    getline(inputFile,information);
    inputFile.close();
    
    //opening image
    cv::Mat image = cv::imread("lena512.png", cv::IMREAD_GRAYSCALE);
    int rows = image.rows;
    int cols = image.cols;
    std::vector<std::vector<int>> img(rows, std::vector<int>(cols));

    // Copy pixel values from the OpenCV image to the 2D vector
    for (int i = 0; i < rows; ++i) 
        for (int j = 0; j < cols; ++j) 
            img[i][j] = static_cast<int>(image.at<uchar>(i, j));

    //generate blocks, saving to txt
    vector <int> blocks = generate_blocks(rows*cols/64);
    std::ofstream outputFile("blocks.txt");
    for (int num: blocks)
        outputFile << num << ' ';
    outputFile.close();

    int cnt = 0;
    //for all blocks
    for (int i : blocks){
        //getting image block
        int block_w = i % (rows / 8);
        int block_h = (i - block_w) / (rows / 8);
        vector <vector<int>> pixel_matrix(8,vector<int>(8));
        for (int i1 = block_h * 8 ;i1 < block_h * 8 + 8; i1++)
            for (int i2 = block_w * 8;i2 < block_w * 8 + 8; i2++)
                pixel_matrix[i1-block_h * 8][i2-block_w*8] = img[i1][i2];
            
        
        //dct transform
        vector<vector<double>> dct_matrix;
        dct_matrix = do_dct(pixel_matrix);
        //embedding info может тут быть проблема с дкт-преобразованием? слишком ровные числа и маленькие
        dct_matrix = embed_to_dct(dct_matrix, string(1, '1') + information.substr(ind_information, 31));
        //undo dct matrix
        vector<vector<int>> new_pixel_matrix = undo_dct(dct_matrix);
        //generating population
        vector<vector<float>> population = generate_population(pixel_matrix,new_pixel_matrix);
        //metric for block and info
        Metric metric(pixel_matrix,string(1, '1') + information.substr(ind_information, 31));
        //tlbo size
        TLBO tlbo(population,128,128,64);

        pair <float,vector<float>> a = tlbo.optimize(metric);
        cout << a.first;
        break;
    }
}

