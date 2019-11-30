#include "TestTask.h"

int main(int argc, char** argv) {  
    
    TestTask::Task task(argc, argv);
    task();
    
    return 0;
} 
