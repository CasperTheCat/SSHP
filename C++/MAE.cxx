#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

int main(int argc, char **argv)
{
    // Catch no input
    if(argc != 2) return EXIT_FAILURE;
    
    // Open the file, it's a CSV (or should be)
    std::ifstream masterFile (argv[1]);
    std::vector<double> rowPack;
    std::string line;

    // For MAE, we only need Vactual and Vpredict
    // These are 

    std::getline(masterFile,line); // Drop Header

    while(std::getline(masterFile, line))
    {   
        // Per line of CSV
        // Skip 0,1,2 and 4
        std::stringstream ls(line);
        std::string cell;
        std::getline(ls, cell, ','); // DumpRow0
        std::getline(ls, cell, ','); // DumpRow1
        std::getline(ls, cell, ','); // DumpRow2

        std::getline(ls, cell, ',');
        rowPack.push_back(std::stod(cell));

        std::getline(ls, cell, ','); // DumpRow4 

        std::getline(ls, cell, ','); 
        rowPack.push_back(std::stod(cell));
    }

    if(rowPack.size() % 2 != 0) return EXIT_FAILURE;

    std::cout << rowPack.size() << std::endl;    

    double total = 0;
    uint32_t rowPackUni = rowPack.size() / 2;
    // Process
#pragma omp parallel for reduction(+:total)
    for(uint64_t i = 0; i < rowPackUni; ++i)
    {
        total += abs(rowPack[i * 2] - rowPack[i * 2 + 1]);
    }
    std::cout << total << std::endl;
    std::cout << total / rowPackUni << std::endl;

    return 0;
}
