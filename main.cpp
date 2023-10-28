#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
#include <random>
#include <vector>
#include <cmath>
#include <filesystem>
#include <cstdlib>
#include <ctime>
using namespace std;

int sign(double x){
    // Функция определяет знак числа
    return (x > 0) - (x < 0);
}

vector <int> generate_blocks(int size){
    // Функция генерирует рандомную перестановку блоков с заданным размером
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
    /*
        Функция преобразует значения пикселей в DCT-coef
        На вход принимается блок изображения
        На выходе блок dct-коэффициентов 
    */

    // Перевод из формата <vector> в формат <Mat>
    cv::Mat inputMat(input.size(), input[0].size(), CV_32S);
    for (size_t i = 0; i < input.size(); i++) {
        for (size_t j = 0; j < input[0].size(); j++) {
            inputMat.at<int>(i, j) = input[i][j];
        }
    }

    // Конвертируем матрицу <Mat> в float
    cv::Mat floatInput;
    inputMat.convertTo(floatInput, CV_32F);

    // Преобразование матрицы в DCT-coef
    cv::Mat dctResult;
    cv::dct(floatInput, dctResult);

    // Сохранение матрицы в виде вектора, вывод вектора
    std::vector<std::vector<double>> output(dctResult.rows, std::vector<double>(dctResult.cols));
    for (int i = 0; i < dctResult.rows; i++) {
        for (int j = 0; j < dctResult.cols; j++) {
            output[i][j] = dctResult.at<float>(i, j);
        }
    }

    return output;
}

std::vector<std::vector<int>> undo_dct(const std::vector<std::vector<double>>& dctCoefficients) {
    /*
        Функция преобразует DCT-coef обратно в пиксели
        На вход получаем блок DCT-coef
        На выходе получаем блок значений пикселей изображения
    */

    // Перевод из формата <vector> в формат <Mat>
    cv::Mat dctResult(dctCoefficients.size(), dctCoefficients[0].size(), CV_64FC1);
    for (size_t i = 0; i < dctCoefficients.size(); i++) {
        for (size_t j = 0; j < dctCoefficients[0].size(); j++) {
            dctResult.at<double>(i, j) = dctCoefficients[i][j];
        }
    }

    // Обратное DCT-преобразование, получаем значения пикселей
    cv::Mat idctResult;
    cv::idct(dctResult, idctResult);

    // Конвертируем из формата <Mat> в <vector> и выводим его
    std::vector<std::vector<int>> output(idctResult.rows, std::vector<int>(idctResult.cols));
    for (int i = 0; i < idctResult.rows; i++) {
        for (int j = 0; j < idctResult.cols; j++) {
            output[i][j] = static_cast<int>(idctResult.at<double>(i, j));
        }
    }

    return output;
}

std::vector<std::vector<double>> embed_to_dct(std::vector<std::vector<double>> dct_matrix, const string bit_string, const char mode = 'A', double q = 8.0){
    /*
    *   Функция реализует встраивание в блок с DCT-coef
    *   На входе:
        dct_matrix - блок dct-coef
        bit_string - встраиваемая строка
        mode - выбранный тип работы, "A" - встроить 32 бита, иначе только 1 бит
        q - шаг квантования
    *   Функция выводит блок dct-coef со встроенными значениями бит
    */
    int ind = 0;
    int cntj = 6;
    for (int i = 0; i < dct_matrix.size(); i++){
        for (int j = dct_matrix[0].size() - 1; j > cntj; j--){
            dct_matrix[i][j] = sign(dct_matrix[i][j]) * (q * int(abs(dct_matrix[i][j]) / q) + (q/2) * (int(bit_string[ind]) - int('0')));
            if (mode != 'A') return dct_matrix; // возвращаем со встроенным одним битом
            ind++;
        }
        if (i == 3) continue; // для обхода только нужных элементов для встраивания
        cntj--;
    }
    return dct_matrix;
}

std::vector<std::vector<double>> generate_population(vector<vector<int>> original,vector<vector<int>> notorig, int population_size = 128,double beta = 0.9,int search_space = 10){
    /*
    *   Функция генерирует популяцию для работы метаэвристик
    *   На входе:
        original - блок изначального изображения
        notorig - блок изображения после встраивание в его DCT-coef (поменялся из-за смены DCT-coef)
        mode - выбранный тип работы, "A" - встроить 32 бита, иначе только 1 бит
        population_size - размер популяции, сколько особей будет в ней
        beta - с какой вероятностью берется значение разности между изначальным блоком и измененным (иначе берется рандом из пространства поиска)
        search_space - пространство поиска, какому интервалу должны соотвествовать значения матрицы изменений 
    *   Функция возвращает популяцию, которая состоит из заданного числа особей
    */

    std::random_device rd;
    std::mt19937 gen(rd());
    double lower_bound = 0.0;
    double upper_bound = 1.0;
    std::uniform_real_distribution<double> uniform_dist(lower_bound, upper_bound);
    std::uniform_int_distribution<int> search_dist(-search_space, search_space);

    // подсчитываем матрицу изменений
    vector<double> diff;
    for (int i = 0; i < original.size(); i++)
        for (int j = 0; j < original[0].size();j++)
            diff.push_back(original[i][j]-notorig[i][j]);
    
    vector<vector<double>> population(population_size,vector<double>(diff.size()));

    // первая особь - начальная матрица изменений
    for (int j = 0; j < diff.size(); j++)
        population[0][j] = diff[j];

    // генерируем популяцию c 0-го индекса,т.к. первая особь - начальная матрица изменений
    for (int i = 1; i < population_size; i++){
        for (int j = 0; j < diff.size(); j++){
            double random_value = uniform_dist(gen);
            if (random_value > beta){ // если рандом больше вероятности, то заместо значения из матрицы изменений выбираем рандомное
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

std::vector<double> meanAlongAxis(const std::vector<std::vector<double>>& array) {
    /*
        Функция считает среднее по всем столбцам популяции
        На входе - популяция
        На выходе - вектор средних значения по столбцам
    */
    std::vector<double> meanValues(array[0].size(), 0.0);

    for (int j = 0; j < array[0].size(); j++) {
        double sum = 0.0;
        for (int i = 0; i < array.size(); i++) {
            sum += array[i][j];
        }
        meanValues[j] = sum / array.size();
    }

    return meanValues;
}

double getRandomInteger(int search_space) {
    // Функция генерирует случайное целое число
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> distribution(-search_space, search_space);
    int randomInteger = distribution(rng);
    return double(randomInteger);
}

std::string extracting_dct(std::vector<std::vector<int>> pixel_block, double q = 8.0){
    /*
    *   Функция реализует извлечение встроенной информации из блока 
    *   На входе:
        pixel_block - блок изображения, из которого необходимо извлечь информацию
        q - заданный шаг квантования еще при встраивании, такой же при извлечении
    *   Функция возвращает строку - извлеченная информация
    */

    vector <vector<double>> dct_block = do_dct(pixel_block); // перевод блока в DCT-coef
    string s;
    int cntj = 6;

    for (int i = 0; i < dct_block.size(); i++){
        for (int j = dct_block[0].size() - 1; j > cntj; j--){
            double c0 = sign(dct_block[i][j]) * (q * int(abs(dct_block[i][j]) / q) + (q/2) * (0));
            double c1 = sign(dct_block[i][j]) * (q * int(abs(dct_block[i][j]) / q) + (q/2) * (1));
            if (abs(dct_block[i][j] - c0) < abs(dct_block[i][j] - c1)){
                s += '0';
                if (s == "0") // если 1ый выстроенный бит - 0, то в такой блок информацию не встроили
                    return "0"; // возвращаем флаг, что информации в этом блоке нет
            }
            else
                s += '1';
        }
        if (i == 3) continue; // для правильного обхода по элементам блока
        cntj--;
    }

    return s;
}

class Metric{
    /*
    *   Класс оценки качества встраивания для данной особи
        Задается параметрами:
        block_matrix - блок из начальной картинки
        bit_string - строка, которую собираемся встраивать
        search_space - пространство поиска, дальше которого значения особи выходить не могут
        mode - встраиваем 1 или несколько бит
    */
    private:
    std::vector<std::vector<int>> block_matrix;
    std::string bit_string;
    int search_space;
    char mode;

    public: 
    Metric(const std::vector<std::vector<int>>& block_matrix, const std::string& bit_string, const int& search_space, const char& mode)
        : block_matrix(block_matrix), bit_string(bit_string), search_space(search_space), mode(mode) {}

    std::pair<double, vector<double>> metric(const std::vector<double>& block, int q = 8) {
        /*
        *   Функция реализует подсчет значения качества для данного блока
        *   На входе:
            block - особь популяции, которая нуждается в проверке качества
            q - заданный шаг квантования еще при встраивании, такой же при извлечении
        *   Функция возвращает пару - преобразованную особь и значение качества для нее
        */

        // New_block - блок после добавлениня к нему матрицы изменений
        // block_flatten - особь, у которой отбросили остаток и проверили на выход за пространство поиска
        std::vector<vector<int>> new_block = block_matrix;
        vector<double> block_flatten = block; 
        for (int i = 0; i < block_flatten.size(); i++){
            block_flatten[i] = floor(block_flatten[i]); // отброс остатка у особи
            if ((block_flatten[i] < -search_space) || (block_flatten[i] > search_space))
                block_flatten[i] = getRandomInteger(search_space); // если значение в особи вышло за пространство - генерируем заместо него новое
        }
        //ind_f1 - индекс, идущий поэлементно в осооби
        int ind_fl = 0;
        for (int i = 0; i < 8; i++){
            for (int j = 0; j < 8; j++){
                new_block[i][j] -= block_flatten[ind_fl];
                if (new_block[i][j] > 255){ // выход за предел 255 в изображении, увеличиваем значение особи на разность выхода и 255
                    int diff = abs(new_block[i][j]-255);
                    block_flatten[ind_fl] += diff;
                    new_block[i][j] = 255;
                }
                if (new_block[i][j] < 0){ // выход за предел 0, уменьшаем значение особи на значение выхода по модулю
                    int diff = abs(new_block[i][j]);
                    block_flatten[ind_fl] -= diff;
                    new_block[i][j] = 0;
                }
                ind_fl++;
            }
        }
        // считаем метрику качества psnr
        int sum_elem = 0;
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++)
                sum_elem += pow(block_matrix[i][j] - new_block[i][j],2);
        double psnr = 0;
        if (sum_elem != 0)
            psnr = 10 * log10((pow(8,2) * pow(255,2)) / double(sum_elem));
        else
            psnr = 42;
            
        string s = extracting_dct(new_block);
        if (s[0] != bit_string[0]){ // несовпадение первого извлеченного бита - нет смысла дальше проверять, возвращаем 0
            std::pair<double, vector<double>> to_ret = std::make_pair(0.0, block_flatten);
            return to_ret;
        }
        // подсчитываем кол-во бит, извлеченных правильно
        int cnt = 0;
        for (int i = 0; i < s.length(); i++)
            if (s[i] == bit_string[i])
                cnt += 1;

        //выводим в кач-ве метрики сумму psnr*10^-4 + ber
        std::pair<double, vector<double>> to_ret = std::make_pair(psnr/10000 + double(cnt)/double(s.length()), block_flatten);
        return to_ret;
    }
};


std::vector<double> getRandomArray(size_t size, double low, double high) {
    // Функция генерирует рандомный вектор, состоящий из нецелых чисел
    std::vector<double> randomArray;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distribution(low, high);

    for (size_t i = 0; i < size; i++) {
        randomArray.push_back(distribution(gen));
    }

    return randomArray;
}

std::vector<double> calculateDifference(const std::vector<double>& teacher, const std::vector<double>& population_mean) {
    // Функция подсчитывает разность между лучшей особью(учитель) и средним для популяции (метаэвристика TLBO)
    size_t num_features = teacher.size();
    std::vector<double> difference;
    
    std::vector<double> randomUniform1 = getRandomArray(num_features, 0.0, 1.0);
    std::vector<double> randomUniform2 = getRandomArray(num_features, 1.0, 2.0);
    
    for (size_t i = 0; i < num_features; i++) {
        difference.push_back(randomUniform1[i] * (teacher[i] - randomUniform2[i] * population_mean[i]));
    }

    return difference;
}

std::vector<double> calculateDifferenceRand(const std::vector<double>& rand1,const std::vector<double>& rand2, const std::vector<double>& population_cur) {
    // Подсчет разности для рандомных особей (метаэвристика TLBO)
    size_t num_features = population_cur.size();
    std::vector<double> new_population;
    
    std::vector<double> randomUniform1 = getRandomArray(num_features, 0.0, 1.0);
    
    for (size_t i = 0; i < num_features; i++) {
        new_population.push_back(population_cur[i] + randomUniform1[i] * (rand1[i] - rand2[i]));
    }

    return new_population;
}

std::vector<double> calculateDifferenceSCA(const std::vector<double>& random_agent,const std::vector<double>& agent, const double C) {
    // Подсчет разности между агентами в популяции (метаэвристика SCA)
    size_t num_features = agent.size();
    std::vector<double> D;
        
    for (size_t i = 0; i < num_features; i++) {
        D.push_back(abs(C * random_agent[i] - agent[i]));
    }

    return D;
}

std::vector<double> calculateDifferenceRandomPositonSCA(const std::vector<double>& random_agent,const std::vector<double>& D, const double A) {
    // Подсчет разности для рандомных позиций (метаэвристика SCA)
    size_t num_features = D.size();
    std::vector<double> new_position;
        
    for (size_t i = 0; i < num_features; i++) {
        new_position.push_back(random_agent[i] - D[i]*A);
    }

    return new_position;
}

int getRandomIndex(int population_size) {
    // Функция генерирует рандомное целое число - индекс для особи в популяции
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<int> uniform_dist(0, population_size - 1);

    return uniform_dist(gen);
}

double getRandomValue(double low, double high) {
    // Функция генерирует случайное нецелое число в интервале (low,high)
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> distribution(low, high);
    double randomInteger = distribution(rng);
    return double(randomInteger);
}

std::vector<double> updatePosition(const std::vector<double>& agent, double t, std::pair<double,double> search) {
    std::vector<double> new_position(agent.size());
    for(size_t i = 0; i < agent.size(); i++) {
        double amplitude = (search.second - search.first) / 2.0;
        new_position[i] = agent[i] + amplitude * std::sin(2 * M_PI * t);
        // Обеспечиваем, что новая позиция находится в пределах пространства поиска
        new_position[i] = std::clamp(new_position[i], search.first, search.second);
    }
    return new_position;
}

class TLBO{
    /*
    *   Класс метаэвристики TLBO
        Задается параметрами:
        population_size - размер популяции, по которой нужно итерироваться
        num_iterations - количество поколений, в течение которых выполняется оптимизация метрики
        num_features - количество значений в каждой особи
        population - популяция, с которой нужно будет работать
    */
    private:
    int population_size;
    int num_iterations;
    int num_features;
    std::vector<std::vector<double>> population;

    public:
    TLBO(const std::vector<std::vector<double>>& initial_population, int population_size, int num_iterations, int num_features)
        : population_size(population_size), num_iterations(num_iterations), num_features(num_features), population(initial_population) {
    }

    std::pair<double, vector<double>> optimize(Metric& obj) {
        /*
            Функция реализует оптимизацию метрики с помощью метаэвристики TLBO
            На входе - объект класса метрики
            На выходе - лучшее значение метрики для всех особей в популяции, особь, показывающая лучшее значение метрики
        */

    
        vector<double> fitness; // вектор, содержащий значения кач-ва для каждой особи
        for (int i = 0; i < population.size(); i++){
            std::pair<double, vector<double>> pr = obj.metric(population[i]);
            population[i] = pr.second; // обновление особи после метрики, с учетом ограничений
            fitness.push_back(pr.first);
        }

        for (int h = 0; h < num_iterations; h++){
            // Стадия учителя
            int best_index = 0;
            for (int i = 0; i < fitness.size();i++)
                if (fitness[i] > fitness[best_index]) // поиск учителя
                    best_index = i;
            
            vector<double> teacher = population[best_index]; // учитель
            vector <double> population_mean = meanAlongAxis(population);

            for (int i = 0; i < population_size;i++){
                if (i != best_index){ // если это не учитель 
                    vector<double> difference = calculateDifference(teacher,population_mean);
                    for (int j = 0; j < difference.size(); j++)
                        difference[j] += population[i][j];
                    
                    double old_score = fitness[i];
                    std::pair<double,vector<double>> new_sc_d = obj.metric(difference);
                    if (new_sc_d.first > old_score){ // проверка, обучил ли учитель ученика 
                        population[i] = new_sc_d.second; // если да - обновляем значение особи и значение метрики для нее 
                        fitness[i] = new_sc_d.first;
                    }
                }
            }

            //стадия ученика 
            for (int i = 0; i < population_size;i++){
                int random_index_1 = 0;
                int random_index_2 = 0;
                while (random_index_1 == random_index_2){ // ищем 2 рандомных учеников
                    random_index_1 = getRandomIndex(population_size);
                    random_index_2 = getRandomIndex(population_size);
                }
                double rand1_sc = fitness[random_index_1],rand2_sc = fitness[random_index_2];
                vector<double> rand1_bl = population[random_index_1], rand2_bl = population[random_index_2];

                vector <double> new_population;
                if (rand1_sc > rand2_sc){ // сравниваем их значения метрики
                    new_population = calculateDifferenceRand(rand1_bl,rand2_bl,population[i]);
                }
                else{
                    new_population = calculateDifferenceRand(rand2_bl,rand1_bl,population[i]);
                }

                double old_score = fitness[i];
                std::pair<double,vector<double>> new_sc_d = obj.metric(new_population);
                if (new_sc_d.first > old_score){ // проверка - лучше ли стало, по сравнению с изначальным
                    population[i] = new_sc_d.second; // если да - обновляем особь, меняем значения метрики
                    fitness[i] = new_sc_d.first;
                }  
            }
        }
        
        // поиск лучшей особи с большим значением метрики
        double max_fitness = 0;
        vector <double> best_agent;
        for (int i = 0; i < fitness.size(); i++){
            if (fitness[i] > max_fitness){
                max_fitness = fitness[i];
                best_agent = population[i];
            }
        }

        std::pair<double,vector<double>> to_ret = std::make_pair(max_fitness,best_agent);
        return to_ret;
    }
};

class SCA{
    /*
    *   Класс метаэвристики SCA
        Задается параметрами:
        population_size - размер популяции, по которой нужно итерироваться
        num_iterations - количество поколений, в течение которых выполняется оптимизация метрики
        num_features - количество значений в каждой особи
        population - популяция, с которой нужно будет работать
    */
    private:
    int population_size;
    int num_iterations;
    int num_features;
    std::vector<std::vector<double>> agents; // Reference to the population used for Metric

    public:
    SCA(const std::vector<std::vector<double>>& initial_population, int population_size, int num_iterations, int num_features)
        : population_size(population_size), num_iterations(num_iterations), num_features(num_features), agents(initial_population) {
    }

    std::pair<double, vector<double>> optimize(Metric& obj, int flag = 0, double a_linear_component = 2.0) {
        /*
            Функция реализует оптимизацию метрики с помощью метаэвристики SCA
            На входе - объект класса метрики, flag - встраиваем 1 бит или несколько (для встраивания бит-флага 0)
            На выходе - лучшее значение метрики для всех особей в популяции, особь, показывающая лучшее значение метрики
        */

        // значения метрики для каждой особи
        double best_fitness = 0.0;
        vector<double> fitness;
        for (int i = 0; i < agents.size(); i++){
            std::pair<double, vector<double>> pr = obj.metric(agents[i]);
            agents[i] = pr.second;
            fitness.push_back(pr.first);
        }
        int best_agent_index  = 0;
        for (int i = 0; i < fitness.size();i++){
            if (fitness[i] > fitness[best_agent_index])
                best_agent_index = i;
        }
        // поиск лучшего агента и лучшего значения метрики
        double best_agent_fitness = fitness[best_agent_index];
        vector <double> best_agent = agents[best_agent_index];
        // оптимизация метаэвристикой
        for (int t = 0; t < num_iterations; t++){
           for (int i = 0; i < agents.size(); i++){
                double a_t = a_linear_component - double(t) * (a_linear_component / double(num_iterations));
                double r1 = getRandomValue(0,1);
                double r2 = getRandomValue(0,1);
                double A = 2 * a_t * r1 - a_t;
                double C = 2 * r2;
                int random_agent_index = getRandomIndex(population_size);
                while (random_agent_index == i)
                   random_agent_index = getRandomIndex(population_size); 
                
                vector <double> random_agent = agents[random_agent_index];
                vector <double> D = calculateDifferenceSCA(random_agent,agents[i],C);
                vector <double> new_position = calculateDifferenceRandomPositonSCA(random_agent,D,A);

                std::pair<double,vector<double>> now_func_bl = obj.metric(new_position);
                if (now_func_bl.first > fitness[i]){
                    agents[i] = now_func_bl.second;
                    fitness[i] = now_func_bl.first;
                    if (fitness[i] > best_agent_fitness){
                        best_agent_fitness = fitness[i];
                        best_agent = agents[i];
                    }
                }

                if (flag){ // при встраивании одного бита
                    if (best_agent_fitness > 0){
                        std::pair<double,vector<double>> to_ret = std::make_pair(best_agent_fitness,best_agent);
                        return to_ret;
                    }
                }
            }
        }
        std::pair<double,vector<double>> to_ret = std::make_pair(best_agent_fitness,best_agent);
        return to_ret;
    }
};

class DE{
    /*
    *   Класс метаэвристики SCA
        Задается параметрами:
        population_size - размер популяции, по которой нужно итерироваться
        num_iterations - количество поколений, в течение которых выполняется оптимизация метрики
        num_features - количество значений в каждой особи
        population - популяция, с которой нужно будет работать
    */
    private:
    int population_size;
    int num_iterations;
    int num_features;
    std::vector<std::vector<double>> agents; // Reference to the population used for Metric

    public:
    DE(const std::vector<std::vector<double>>& initial_population, int population_size, int num_iterations, int num_features)
        : population_size(population_size), num_iterations(num_iterations), num_features(num_features), agents(initial_population) {
    }

    std::pair<double, vector<double>> optimize(Metric& obj, double cr = 0.3, double f = 0.1) { 
        /*  
            Функция реализует оптимизацию метрики с помощью метаэвристики DE
            На входе - объект класса метрики, flag - встраиваем 1 бит или несколько (для встраивания бит-флага 0)
            На выходе - лучшее значение метрики для всех особей в популяции, особь, показывающая лучшее значение метрики
        */
        // значения метрики для каждой особи
        double best_fitness = 0.0;
        vector<double> fitness;
        for (int i = 0; i < agents.size(); i++){
            std::pair<double, vector<double>> pr = obj.metric(agents[i]);
            agents[i] = pr.second;
            fitness.push_back(pr.first);
        }
        double best_agent_fitness = fitness[0];
        vector <double> best_agent = agents[0];
        vector<double> y(agents[0].size());
        // оптимизация метаэвристикой
        for (int t = 0; t < num_iterations; t++){
           for (int i = 0; i < agents.size(); i++){
                //cout << rand() % 4096;;
                // выбор рандомных индексов, отличных от друг друга(a, b, c) и от i-го
                int a_ind,b_ind,c_ind;
                //a_ind = getRandomIndex(population_size);
                //b_ind = getRandomIndex(population_size);
                //c_ind = getRandomIndex(population_size);
                a_ind = rand() % agents.size();
                b_ind = rand() % agents.size();
                c_ind = rand() % agents.size();
//                while (a_ind == i) a_ind = getRandomIndex(population_size);
//                while (b_ind == i || b_ind == a_ind) b_ind = getRandomIndex(population_size);
//                while (c_ind == i || c_ind == a_ind || c_ind == b_ind) c_ind = getRandomIndex(population_size);
               while (a_ind == i) a_ind = rand() % agents.size();
               while (b_ind == i || b_ind == a_ind) b_ind = rand() % agents.size();
               while (c_ind == i || c_ind == a_ind || c_ind == b_ind) c_ind = rand() % agents.size();
                // генерация возможной новой особи

                
                for (int pos = 0; pos < agents[0].size(); pos++){
                    double r = getRandomValue(0,1);
                    if (r < cr)
                        y[pos] = agents[a_ind][pos] + f * (agents[b_ind][pos] - agents[c_ind][pos]);
                    else
                        y[pos] = agents[i][pos];
                }
                // проверка новой особи
                std::pair<double, vector<double>> pr = obj.metric(y);
                if (pr.first > fitness[i]){
                    fitness[i] = pr.first;
                    agents[i] = pr.second;
                    if (fitness[i] > best_agent_fitness){
                        best_agent_fitness = fitness[i];
                        best_agent = agents[i];
                    }
                }
            }
        }

//        // поиск лучшего агента
//        double best_agent_fitness = 0;
//        vector <double> best_agent;
//        for (int i = 0; i < agents.size(); i++){
//            std::pair<double, vector<double>> pr = obj.metric(agents[i]);
//            agents[i] = pr.second;
//            fitness[i] = pr.first;
//            if (fitness[i] > best_agent_fitness){
//                best_agent_fitness = fitness[i];
//                best_agent = agents[i];
//            }
//        }
        std::pair<double,vector<double>> to_ret = std::make_pair(best_agent_fitness,best_agent);
        return to_ret;
    }
};

class SSA{
    /*
    *   Класс метаэвристики SSA
        Задается параметрами:
        population_size - размер популяции, по которой нужно итерироваться
        num_iterations - количество поколений, в течение которых выполняется оптимизация метрики
        num_features - количество значений в каждой особи
        population - популяция, с которой нужно будет работать
    */
private:
    int searching;
    int num_salps; // population_size
    int num_iterations;
    int num_dimensions; // num_features
    std::vector<std::vector<double>> salps; // Reference to the population used for Metric

public:
    SSA(const std::vector<std::vector<double>>& initial_population,int searching, int num_salps, int num_iterations, int num_dimensions)
            : searching(searching), num_salps(num_salps), num_iterations(num_iterations), num_dimensions(num_dimensions),salps(initial_population) {
    }

    std::pair<double, vector<double>> optimize(Metric& obj) {
        /*
            Функция реализует оптимизацию метрики с помощью метаэвристики SCA
            На входе - объект класса метрики, flag - встраиваем 1 бит или несколько (для встраивания бит-флага 0)
            На выходе - лучшее значение метрики для всех особей в популяции, особь, показывающая лучшее значение метрики
        */
        const std::pair<double, double> search_space(static_cast<double>(-searching),static_cast<double>(searching));
        // Calculate fitness for each salp
        std::vector<double> fitness(num_salps);
        for (int i = 0; i < num_salps; ++i) {
            std::pair<double, vector<double>> pr = obj.metric(salps[i]);
            salps[i] = pr.second;
            fitness[i] = pr.first;
        }

        for (int t = 0; t < num_iterations; ++t) {
            // Get the best salp
            int best_index = std::distance(fitness.begin(), std::max_element(fitness.begin(), fitness.end()));
            std::vector<double> best_salp = salps[best_index];

            // Update positions with adaptive parameter
            double w = 1.0 - (static_cast<double>(t) / num_iterations);

            for (int i = 0; i < num_salps; ++i) {
                for (int j = 0; j < num_dimensions; ++j) {
                    if (i == 0) {
                        salps[i][j] = best_salp[j]; // The first salp follows the lead
                    } else {
                        // Subsequent salps follow their predecessor
                        salps[i][j] = (salps[i][j] + salps[i - 1][j]) / 2;

                        // Introduce randomization for the latter half of iterations
                        if (t > num_iterations / 2) {
                            salps[i][j] += w * (2.0 * static_cast<double>(std::rand()) / RAND_MAX - 1.0); // random value in [-1,1]
                        }
                        // Boundary check
                        if (salps[i][j] < search_space.first) {
                            salps[i][j] = search_space.first;
                        } else if (salps[i][j] > search_space.second) {
                            salps[i][j] = search_space.second;
                        }
                    }
                }
            }

            // Update fitness values
            for (int i = 0; i < num_salps; ++i) {
                std::pair<double, vector<double>> pr = obj.metric(salps[i]);
                salps[i] = pr.second;
                fitness[i] = pr.first;
            }
        }

        int best_index = std::distance(fitness.begin(), std::max_element(fitness.begin(), fitness.end()));
        std::pair<double,vector<double>> to_ret = std::make_pair(fitness[best_index],salps[best_index]);
        return to_ret;
    }
};

class WOA{
    /*
    *   Класс метаэвристики WOA
        Задается параметрами:
        population_size - размер популяции, по которой нужно итерироваться
        num_iterations - количество поколений, в течение которых выполняется оптимизация метрики
        num_features - количество значений в каждой особи
        population - популяция, с которой нужно будет работать
    */
private:
    int num_agents;
    int num_iterations;
    int num_features;
    std::vector<std::vector<double>> agents;
    int searching;
public:
    WOA(const std::vector<std::vector<double>>& initial_population, int num_agents, int num_iterations, int num_features,int searching)
            : num_agents(num_agents), num_iterations(num_iterations), num_features(num_features), agents(initial_population),searching(searching) {
    }

    std::pair<double, vector<double>> optimize(Metric& obj) {
        /*
            Функция реализует оптимизацию метрики с помощью метаэвристики WOA
            На входе - объект класса метрики
            На выходе - лучшее значение метрики для всех особей в популяции, особь, показывающая лучшее значение метрики
        */

        std::vector<double> fitness(num_agents);
        const std::pair<double, double> search_space(static_cast<double>(-searching),static_cast<double>(searching));
        for(int i = 0; i < num_agents; i++) {
            std::pair<double, vector<double>> pr = obj.metric(agents[i]);
            agents[i] = pr.second; // обновление особи после метрики, с учетом ограничений
            fitness.push_back(pr.first);
        }

        for (int t = 0; t < num_iterations; t++) {
            double a = 2.0 - t * ((2.0) / num_iterations);

            for(int i = 0; i < num_agents; i++) {
                double r1 = getRandomValue(0, 1);
                double r2 = getRandomValue(0, 1);

                double A = 2.0 * a * r1 - a;
                double C = 2.0 * r2;

                double b = 1;
                double l = (getRandomValue(0, 1) * 2) - 1;

                double p = getRandomValue(0, 1);

                std::vector<double> X_rand = agents[getRandomValue(0, num_agents - 1)];

                std::vector<double> D_X_rand(num_features);
                std::vector<double> X_new(num_features);

                if(p < 0.5) {
                    if(std::fabs(A) < 1) {
                        for(int j = 0; j < num_features; j++) {
                            D_X_rand[j] = std::fabs(C * X_rand[j] - agents[i][j]);
                            X_new[j] = X_rand[j] - A * D_X_rand[j];
                        }
                    } else {
                        for(int j = 0; j < num_features; j++) {
                            X_new[j] = X_rand[j] - A * std::fabs(C * X_rand[j] - agents[i][j]);
                        }
                    }
                } else {
                    for(int j = 0; j < num_features; j++) {
                        D_X_rand[j] = std::fabs(X_rand[j] - agents[i][j]);
                        X_new[j] = D_X_rand[j] * std::exp(b * l) * std::cos(2 * M_PI * l) + X_rand[j];
                    }
                }

                for(int j = 0; j < num_features; j++) {
                    X_new[j] = std::clamp(X_new[j], search_space.first, search_space.second);
                }

                std::pair<double, vector<double>> pr  = obj.metric(X_new);
                if(pr.first > fitness[i]) {
                    agents[i] = pr.second;
                    fitness[i] = pr.first;
                }
            }
        }

        auto max_element_iter = std::max_element(fitness.begin(), fitness.end());
        int best_index = std::distance(fitness.begin(), max_element_iter);
        std::pair<double,vector<double>> to_ret = std::make_pair(fitness[best_index],agents[best_index]);
        return to_ret;
    }
};

class ICA{
    /*
    *   Класс метаэвристики ICA
        Задается параметрами:
        population_size - размер популяции, по которой нужно итерироваться
        num_iterations - количество поколений, в течение которых выполняется оптимизация метрики
        num_features - количество значений в каждой особи
        population - популяция, с которой нужно будет работать
    */
private:
    int num_agents;
    int num_iterations;
    int num_features;
    std::vector<std::vector<double>> agents;
    int searching;
    int num_empires;
public:
    ICA(const std::vector<std::vector<double>>& initial_population, int num_agents, int num_iterations, int num_features,int searching, int num_empires)
            : num_agents(num_agents), num_iterations(num_iterations), num_features(num_features), agents(initial_population),searching(searching),num_empires(num_empires) {
    }

    std::pair<double, vector<double>> optimize(Metric& obj) {
        /*
            Функция реализует оптимизацию метрики с помощью метаэвристики ICA
            На входе - объект класса метрики
            На выходе - лучшее значение метрики для всех особей в популяции, особь, показывающая лучшее значение метрики
        */
        std::vector<double> fitness(num_agents);
        for (int i = 0; i < num_agents; ++i) {
            std::pair<double, vector<double>> pr = obj.metric(agents[i]);
            agents[i] = pr.second;
            fitness[i] = pr.first;
        }

        std::vector<int> sorted_indices(num_agents);
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(), [&fitness](int i1, int i2) { return fitness[i1] > fitness[i2]; });

        std::vector<std::vector<double>> empires(num_empires);
        std::vector<std::vector<double>> colonies(num_agents - num_empires);
        std::vector<double> empire_fitness(num_empires);
        std::vector<double> colony_fitness(num_agents - num_empires);

        for (int i = 0; i < num_empires; ++i) {
            empires[i] = agents[sorted_indices[i]];
            empire_fitness[i] = fitness[sorted_indices[i]];
        }
        for (int i = 0; i < colonies.size(); ++i) {
            colonies[i] = agents[sorted_indices[i + num_empires]];
            colony_fitness[i] = fitness[sorted_indices[i + num_empires]];
        }

        // Инициализация адаптивных скоростей обучения и коэффициентов ассимиляции
        double learning_rate_init = 0.5;
        double learning_rate_final = 0.01;
        double assimilation_coeff_init = 0.5;
        double assimilation_coeff_final = 0.1;

        for (int t = 0; t < num_iterations; ++t) {
            double assimilation_coeff = assimilation_coeff_init - (assimilation_coeff_init - assimilation_coeff_final) * static_cast<double>(t) / num_iterations;
            double learning_rate = learning_rate_init - (learning_rate_init - learning_rate_final) * static_cast<double>(t) / num_iterations;

            // Осуществляем скрещивание между империями
            for (int i = 0; i < num_empires; ++i) {
                if (static_cast<double>(std::rand()) / RAND_MAX < 0.5) {
                    int other = std::rand() % num_empires;
                    std::vector<double> child(num_features);
                    for (int j = 0; j < num_features; ++j) {
                        child[j] = 0.5 * (empires[i][j] + empires[other][j]);
                    }
                    std::pair<double, vector<double>> pr  = obj.metric(child);
                    if (pr.first > empire_fitness[i]) {
                        empires[i] = pr.second;
                        empire_fitness[i] = pr.first;
                    }
                }
            }

            // Обновление позиций колоний на основе их соответствующих империй
            for (int i = 0; i < colonies.size(); ++i) {
                for (int j = 0; j < num_features; ++j) {
                    colonies[i][j] -= learning_rate * assimilation_coeff * (colonies[i][j] - empires[i % num_empires][j]);
                }
            }

            // Осуществляем революцию, внося случайные возмущения
            for (int i = 0; i < colonies.size(); ++i) {
                for (int j = 0; j < num_features; ++j) {
                    colonies[i][j] += 0.2 * static_cast<double>(rand()) / RAND_MAX;
                }
            }

            // Обновление приспособленности всех агентов
            for (int i = 0; i < num_empires; ++i) {
                std::pair<double, vector<double>> pr  = obj.metric(empires[i]);
                empire_fitness[i] = pr.first;
                empires[i] = pr.second;
            }
            for (int i = 0; i < colonies.size(); ++i) {
                std::pair<double, vector<double>> pr  = obj.metric(colonies[i]);
                colony_fitness[i] = pr.first;
                colonies[i] = pr.second;
            }

            // Поиск лучшей империи
            int best_index = std::distance(empire_fitness.begin(), std::max_element(empire_fitness.begin(), empire_fitness.end()));
            auto best_agent = empires[best_index];
            double best_fitness = empire_fitness[best_index];
                                                   }

        int best_index = std::distance(empire_fitness.begin(), std::max_element(empire_fitness.begin(), empire_fitness.end()));;
        std::pair<double,vector<double>> to_ret = std::make_pair(empire_fitness[best_index],empires[best_index]);
        return to_ret;
    }
}; // num_empires??

class AOA{
    /*
    *   Класс метаэвристики AOA
        Задается параметрами:
        population_size - размер популяции, по которой нужно итерироваться
        num_iterations - количество поколений, в течение которых выполняется оптимизация метрики
        num_features - количество значений в каждой особи
        population - популяция, с которой нужно будет работать
    */
private:
    int num_agents;
    int num_iterations;
    int num_features;
    std::vector<std::vector<double>> agents;
    int searching;
public:
    AOA(const std::vector<std::vector<double>>& initial_population, int num_agents, int num_iterations, int num_features,int searching)
            : num_agents(num_agents), num_iterations(num_iterations), num_features(num_features), agents(initial_population),searching(searching) {
    }

    std::pair<double, vector<double>> optimize(Metric& obj) {
        /*
            Функция реализует оптимизацию метрики с помощью метаэвристики AOA
            На входе - объект класса метрики
            На выходе - лучшее значение метрики для всех особей в популяции, особь, показывающая лучшее значение метрики
        */
        std::pair<double,double> search(static_cast<double>(-searching),static_cast<double>(searching));
        std::vector<double> fitness(num_agents);
        for (int i = 0; i < num_agents; ++i) {
            std::pair<double, vector<double>> pr = obj.metric(agents[i]);
            agents[i] = pr.second;
            fitness[i] = pr.first;
        }
        // Основной цикл оптимизации
        for (int t = 0; t < num_iterations; t++) {
            double time_ratio = static_cast<double>(t) / num_iterations;

            for (int i = 0; i < num_agents; i++) {
                std::vector<double> new_position = updatePosition(agents[i], time_ratio, search);
                std::pair<double, vector<double>> pr  = obj.metric(new_position);
                if (pr.first > fitness[i]) {
                    agents[i] = pr.second;
                    fitness[i] = pr.first;
                }
            }
        }

        auto max_element_iter = std::max_element(fitness.begin(), fitness.end());
        int best_index = std::distance(fitness.begin(), max_element_iter);
        std::pair<double,vector<double>> to_ret = std::make_pair(fitness[best_index],agents[best_index]);
        return to_ret;
    }
};

double psnr(std::vector<std::vector<int>> original_img,std::vector<std::vector<int>> saved_img){
    /*
        Функция принимает на вход оригинальное изображение и изображение после вставки
        На выходе - значение метрики psnr
    */
    int sum_elem = 0;
    for (int i = 0; i < original_img.size(); i++)
        for (int j = 0; j < original_img[0].size(); j++)
            sum_elem += pow(original_img[i][j] - saved_img[i][j],2);
        
    double psnr = 10 * log10(pow(255,4) / double(sum_elem));
    return psnr;
}

double ssim(std::vector<std::vector<int>> original_img,std::vector<std::vector<int>> saved_img){
    /*
        Функция принимает на вход оригинальное изображение и изображение после вставки
        На выходе - значение метрики ssim
    */
    double mean1 = 0, mean2 = 0;
    for (int i = 0; i < original_img.size(); i++){
        for (int j = 0; j < original_img[0].size(); j++){
            mean1 += original_img[i][j];
            mean2 += saved_img[i][j];
        }
    }
    mean1 /= ((original_img.size()*original_img.size())*3);
    mean2 /= ((original_img.size()*original_img.size())*3);

    double sd1 = 0, sd2 = 0, cov = 0;


    for (int i = 0; i < original_img.size(); i++){
        for (int j = 0; j < original_img[0].size(); j++){
            sd1 += (original_img[i][j]/3 - mean1) * (original_img[i][j]/3 - mean1);
            sd2 += (saved_img[i][j]/3 - mean2) * (saved_img[i][j]/3 - mean2);
            cov += (original_img[i][j] / 3 - mean1) * (saved_img[i][j] / 3 - mean2);
        }
    }
    
    cov /= (original_img.size()*original_img.size());
    sd1 = pow((sd1 / (original_img.size()*original_img.size())),0.5);
    sd2 = pow((sd2 / (original_img.size()*original_img.size())),0.5);

    double c1 = pow(0.01*255,2), c2 = pow(0.03*255,2);
    return ((2 * mean1 * mean2 + c1) * (2 * cov + c2)) / (
            (pow(mean1,2) + pow(mean2,2) + c1) * (pow(sd1,2) + pow(sd2,2) + c2));
}


int main() {

    vector <string> pictures{
    "peppers512.png","lena512.png",
    "airplane512.png","baboon512.png","barbara512.png",
    "boat512.png","goldhill512.png","stream_and_bridge512.png"
    };

    vector <string> metaheu{
        "sca","tlbo","ica","aoa","ssa","woa","de"
    };

    for (int m4 = 0; m4 < metaheu.size(); m4++) {
        string METAHEURISTIC = metaheu[m4];
        cout << METAHEURISTIC << '\n';
        for (int i = 0; i < pictures.size(); i++) {

            string picture = pictures[i];
            cout << picture << ' ';
            const string directoryPath = picture + METAHEURISTIC;
            // Create the directory using mkdir
            int status = std::system(("mkdir " + std::string(directoryPath)).c_str());
            int mode = 1;
            if (mode == 1) { // встраивание
                const int SEARCH_SPACE = 10; // пространство поиска

                //открытие файла, что нужно встроить
                ifstream inputFile("to_embed.txt");
                int ind_information = 0;
                string information;
                getline(inputFile, information);
                inputFile.close();
                //открытие картинки
                cv::Mat image = cv::imread(picture, cv::IMREAD_GRAYSCALE);
                int rows = image.rows;
                int cols = image.cols;
                std::vector<std::vector<int>> img(rows, std::vector<int>(cols));

                // трансформация из формата <Mat> в <vector>
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        img[i][j] = static_cast<int>(image.at<uchar>(i, j));

                //генерация порядка блоков, сохранение их в файл
                vector<int> blocks = generate_blocks(rows * cols / 64);
                std::ofstream outputFile(picture + METAHEURISTIC + "/blocks.txt");
                for (int num: blocks)
                    outputFile << num << ' ';
                outputFile.close();

                vector<vector<int>> copy_img = img;
                int cnt1 = 0;
                int cnt_blocks = 0;

                for (int i: blocks) {
                    cout << endl << cnt_blocks++ << ' ' << endl;
                    //получаем блок изображения по известному номера блока
                    int block_w = i % (rows / 8);
                    int block_h = (i - block_w) / (rows / 8);
                    vector<vector<int>> pixel_matrix(8, vector<int>(8));
                    for (int i1 = block_h * 8; i1 < block_h * 8 + 8; i1++)
                        for (int i2 = block_w * 8; i2 < block_w * 8 + 8; i2++)
                            pixel_matrix[i1 - block_h * 8][i2 - block_w * 8] = img[i1][i2];

                    //трансформация из пикселей в DCT-coef
                    vector<vector<double>> dct_matrix;
                    dct_matrix = do_dct(pixel_matrix);

                    //встраивание информации в DCT-coef блок
                    dct_matrix = embed_to_dct(dct_matrix, string(1, '1') + information.substr(ind_information, 31));

                    //перевод блока из DCT-coef в пиксельный формат
                    vector<vector<int>> new_pixel_matrix = undo_dct(dct_matrix);

                    //генерация популяции на основе блока после встраивания и изначального
                    vector<vector<double>> population = generate_population(pixel_matrix, new_pixel_matrix, 128,
                                                                            double(0.9), SEARCH_SPACE);

                    //задаем объект метрики для данного блока и информации для встраивания
                    Metric metric(pixel_matrix, string(1, '1') + information.substr(ind_information, 31), SEARCH_SPACE,
                                  'A');
                    //выбор метаэвристики и оптимизации с помощью нее
                    pair<double, vector<double>> solution;
                    if (METAHEURISTIC == "tlbo") {
                        TLBO meta(population, 128, 128, 64);
                        solution = meta.optimize(metric);
                    }
                    else if(METAHEURISTIC == "sca") {
                        SCA meta(population, 128, 128, 64);
                        solution = meta.optimize(metric);
                    }
                    else if(METAHEURISTIC == "de") {
                        DE meta(population, 128, 128, 64);
                        solution = meta.optimize(metric);
                    }
                    else if (METAHEURISTIC == "ssa") {
                        SSA meta(population, SEARCH_SPACE, 128, 128, 64);
                        solution = meta.optimize(metric);
                    }
                    else if (METAHEURISTIC == "woa") {
                        WOA meta(population, 128, 128, 64, SEARCH_SPACE);
                        solution = meta.optimize(metric);
                    }
                    else if (METAHEURISTIC == "aoa") {
                        AOA meta(population, 128, 128, 64, SEARCH_SPACE);
                        solution = meta.optimize(metric);
                    }
                    else if (METAHEURISTIC == "ica") {
                        ICA meta(population, 128, 128, 64, SEARCH_SPACE, 10);
                        solution = meta.optimize(metric);
                    }
                    //pair<double, vector<double>> solution = meta.optimize(metric);

                    if (solution.first > 1) { // значение кач-ва метрики >1 => информация встроена идеально, сохраняем новый блок, добавляя к нему матрицу изменений
                        cnt1 += 1;
                        int ind = 0;
                        for (int i1 = block_h * 8; i1 < block_h * 8 + 8; i1++) {
                            for (int i2 = block_w * 8; i2 < block_w * 8 + 8; i2++) {
                                copy_img[i1][i2] -= solution.second[ind];
                                ind++;
                            }
                        }

                        vector<vector<int>> new_pixel_matrix_1(8, vector<int>(8));
                        for (int i1 = block_h * 8; i1 < block_h * 8 + 8; i1++)
                            for (int i2 = block_w * 8; i2 < block_w * 8 + 8; i2++)
                                new_pixel_matrix_1[i1 - block_h * 8][i2 - block_w * 8] = copy_img[i1][i2];

                        ind_information += 31; // переход к следующей части информации
                    } else { // информация встроена неидеально
                        cout << solution.first;
                        int searching = 5;
                        // встраиваем 1 бит - 0
                        dct_matrix = embed_to_dct(do_dct(pixel_matrix), string(1, '0'), 'Z');

                        // перевод из DCT-coef в пиксели
                        vector<vector<int>> new_pixel_matrix = undo_dct(dct_matrix);

                        //генерация популяции
                        vector<vector<double>> population = generate_population(pixel_matrix, new_pixel_matrix, 128,
                                                                                double(0.9), searching);

                        //создание объекта метрики, с учетом встраивание 1 бита
                        Metric metric(pixel_matrix, string(1, '0'), searching, 'Z');

                        // оптимизация с помощью метаэвристики SCA
                        SCA sca(population, 128, 128, 64);
                        pair<double, vector<double>> solution = sca.optimize(metric, 1);

                        for (int i1 = 0; i1 < 64; i1++)
                            cout << solution.second[i1] << ' ';
                        //сохраняем блок, в который не встраивалась информация
                        int ind = 0;
                        for (int i1 = block_h * 8; i1 < block_h * 8 + 8; i1++)
                            for (int i2 = block_w * 8; i2 < block_w * 8 + 8; i2++)
                                copy_img[i1][i2] -= solution.second[ind++];
                    }

                }

                //сохраняем изображение
                cv::Mat imageMat(rows, cols, CV_8UC1);
                for (int row = 0; row < rows; row++)
                    for (int col = 0; col < cols; col++)
                        imageMat.at<uchar>(row, col) = static_cast<uchar>(copy_img[row][col]);
                std::string outputFilePath = picture + METAHEURISTIC + "/saved.png";
                bool success = cv::imwrite(outputFilePath, imageMat);

                cout << cnt1;
            }
            mode = 2;
            if (mode == 2) { // извлечение
                string bit_string = "";

                //открываем изображение
                cv::Mat image = cv::imread(picture + METAHEURISTIC + "/saved.png", cv::IMREAD_GRAYSCALE);
                int rows = image.rows;
                int cols = image.cols;
                std::vector<std::vector<int>> img(rows, std::vector<int>(cols));

                // переводим из формата <Mat> в <vector>
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        img[i][j] = static_cast<int>(image.at<uchar>(i, j));

                // открываем файл с порядком блоков
                std::ifstream inputFile(picture + METAHEURISTIC + "/blocks.txt");
                std::vector<int> blocks;
                int num;
                while (inputFile >> num) {
                    blocks.push_back(num);
                }
                inputFile.close();
                for (int i: blocks) {
                    // получаем значения блока изображения по прочитанному номеру блоку
                    int block_w = i % (rows / 8);
                    int block_h = (i - block_w) / (rows / 8);
                    vector<vector<int>> pixel_matrix(8, vector<int>(8));
                    for (int i1 = block_h * 8; i1 < block_h * 8 + 8; i1++)
                        for (int i2 = block_w * 8; i2 < block_w * 8 + 8; i2++)
                            pixel_matrix[i1 - block_h * 8][i2 - block_w * 8] = img[i1][i2];

                    // извлекаем информацию из блока
                    string s = extracting_dct(pixel_matrix);
                    if (s != "0") // информация должна быть извлечена
                        bit_string += s.substr(1);
                }
                // сохраняем извлеченную информацию в файл
                std::ofstream outputFile(picture + METAHEURISTIC + "/saved.txt");
                outputFile << bit_string;
                outputFile.close();

                // октрываем изначальное изображение и считаем метрику psnr между изначальным и получившимся
                cv::Mat image_base = cv::imread(picture, cv::IMREAD_GRAYSCALE);
                rows = image_base.rows;
                cols = image_base.cols;
                std::vector<std::vector<int>> img_base(rows, std::vector<int>(cols));
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        img_base[i][j] = static_cast<int>(image_base.at<uchar>(i, j));
                cout << psnr(img_base, img) << ' ' << ssim(img_base, img) << '\n';
            }
            else {
                //открываем изображение
                string route;
                //cin >> route;
                route = directoryPath + "/saved.png";
                cv::Mat image = cv::imread(route, cv::IMREAD_GRAYSCALE);
                int rows = image.rows;
                int cols = image.cols;
                std::vector<std::vector<int>> img(rows, std::vector<int>(cols));

                // переводим из формата <Mat> в <vector>
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        img[i][j] = static_cast<int>(image.at<uchar>(i, j));

                cv::Mat image_base = cv::imread(picture, cv::IMREAD_GRAYSCALE);
                rows = image_base.rows;
                cols = image_base.cols;
                std::vector<std::vector<int>> img_base(rows, std::vector<int>(cols));
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        img_base[i][j] = static_cast<int>(image_base.at<uchar>(i, j));

                std::ifstream inputFile(directoryPath + "/saved.txt");
                std::string line;
                std::getline(inputFile, line);
                cout << psnr(img_base, img) << ' ' << ssim(img_base, img) << ' ' << line.length() << '\n';
            }
        }
    }
}