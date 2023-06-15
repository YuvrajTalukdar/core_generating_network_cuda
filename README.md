# core_generating_network_cuda
This is the next version of the project non backpropagation neural network which includes considerable changes and improvements.

This is a novel ML algorithm which uses LP for generating a trained neural network.
Version 1 (non backpropagation neural network) had certain shortcommings like poor accuracy in data which has large no os attributes. Also it did not had 
any mechanism to to fine tune the hyper parameters. 
This version solves all such problems.

Copyrighte Yuvraj Talukdar SW-16114/2023

We implemented the program in cuda. It works well and manage to have high gpu utilization. We tested the program in Nvidia 30 series (Ampere Architecture) and compiled using CUDA 12. CUDA 11 too should work. For older or newer Nvidia gpu the make file may need some changes.  

Compilation from source (Linux)-
Required Library
1. G++ Compiler any version which supports c++ std 17.
2. Make
3. CUDA 11 or later
For compilation-
1. Git clone https://github.com/YuvrajTalukdar/core_generating_network_cuda.git
2. cd core_generating_network
3. make all
4. To run- ./a.out
