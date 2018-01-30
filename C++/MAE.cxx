#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

int main(int argc, char **argv)
{
    // Catch no input
    if(argc < 2) return EXIT_FAILURE;
    
    std::vector<double> rowPack;

    for(uint32_t i = 1; i < argc; ++i)
    {
        //std::cerr << "Loading file " << i << std::endl;
        // Open the file, it's a CSV (or should be)
        std::ifstream masterFile (argv[i], std::ifstream::in);
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
            if(cell.empty()) {std::cout << i << std::endl; return 1;}
            rowPack.push_back(std::stod(cell));
    
            std::getline(ls, cell, ','); // DumpRow4 
    
            std::getline(ls, cell, ','); 
            if(cell.empty()) {std::cout << i << std::endl; return 1;}
            rowPack.push_back(std::stod(cell));
        }
    }

    if(rowPack.size() % 2 != 0) return EXIT_FAILURE;

    std::cout << "Elements: " << rowPack.size() << std::endl << "Files: " << (argc - 1) << std::endl;    

    double total = 0;
    uint32_t rowPackUni = rowPack.size() / 2;
    // Process
#pragma omp parallel for reduction(+:total)
    for(uint64_t i = 0; i < rowPackUni; ++i)
    {
        total += abs(rowPack[i * 2] - rowPack[i * 2 + 1]);
    }
    //std::cout << total << std::endl;
    std::cout << "Global MAE:" << total / rowPackUni << std::endl;

    return 0;
}
