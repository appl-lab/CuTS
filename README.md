# AE cuTS: Scaling Subgraph Isomorphism on Distributed Multi-GPU Systems Using Trie Based Data Structure

This repository contains the code for the "cuTS: Scaling Subgraph Isomorphism on Distributed Multi-GPU Systems Using Trie Based Data Structure" framework. The cuTS framework is an efficient subgraph isomorphism solver for GPUs. 

## Package requirements:
* cmake(>=3.10)
* OpenMP
* CUDA(>=10.0)
* MPI (openmpi/3.0.1)

We used `cmake 3.17.2`, `openmpi/3.0.1` and `CUDA 11.0`

## Build instructions:

We provide automated scripts to build the framework. Running `build.sh` will build the entire framework. (Before build, user should set the cuda arch properly based on his/her platform. In our experiments, we used Nvidia V100 and A100 GPU cards. To build entire cuts project on V100 machine, usr should enable target_compile_options(cuts/cuts_mpi PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_70,code=sm_70>) in the CMakeLists.txt file. For A100 machine, user should set target_compile_options(cuts/cuts_mpi PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_80,code=sm_80>) in the CMakeLists.txt file and comment out target_compile_options for V100.) To set the CUDA binary and library path, edit `build.sh`. The compilation will produce the following executable `cuts` `cuts_mpi` in the build directory,`cuts` is the subgraph isomorphisms solver for single node,`cuts_mpi` is the solver for multiple nodes.

## Dataset preparation:

    bash download.sh

## Graph format:
See examples in dataset/query/ours_format (first line is the number of vertices and other lines represent edges).

### Software requirements for downloading script:
* gunzip
* wget
* python3
    
## Single node experiments:
For single node expriment, the machine should have at least`300GB` memory. If the machine doesn't have that much memory, cuts still be able to run some experiment 
cases, however it will throw out of memroy (illegal memory) error for some experiment cases.

The provided script will execute the code and report the data graph name, query graph name, execution time, and number of subgraph isomorphisms.

If cuts failed to run, it will throw an `illegal memory` error (`out of memory`) and terminate (`same for multi nodes`).

    python cuts.py

Sample output:

    start loading graph file from disk to memory...
    graph loading complete...
    start copying graph to gpu...
    end copying graph to gpu...
    Enron.g,1.g,5469.28ms,697122720
    start writting matching results to disk,ans.txt
    finish writting matching results to disk,ans.txt
    
By default, cuts doesn't write matching results to disk. Inorder to save matching results to disk, users have to set write_to_disk be true in main.cu/main_mpi.cu (matching results will be saved to ans.txt).

Sample ans.txt:

    (0,959800)(2,961042)(4,961098)(1,961040)(3,961039)
    node 0 in query graph match node 959800 in data graph...
    
    
### Custom query graph and data graph

The users can run custom query graphs and data graphs using the following command, user should make the graph in proper format (see graph format).

    ./build/cuts <data_graph_path> <query_graph_path> <trunks num>(optional for big data graphs, recommended (2-8))
    
## Multi node experiments:
For multi node expriment, each machine should have (>=`400GB`) memory.

The provided script will execute the code and report the node id, data graph name, query graph name, execution time and number of subgraph isomorphisms on two nodes. Make sure that the MPI and CUDA libraries are loaded. Update `2nodes_exe.sh` to set the paths (prefix is the prefix path of cuts project). Execute the following commands to obtain the output.

    sbatch 2nodes_exe.sh
    
The provided script will execute the code and report the node id, data graph name, query graph name, execution time and number of subgraph isomorphisms on four nodes. Make sure that the MPI libraries and CUDA libraries are loaded. Update `4nodes_exe.sh` to set the paths (prefix is the prefix path of cuts project). Execute the following commands to obtain the output.

    sbatch 4nodes_exe.sh

The scripts will produce an output file with the execution time and number of subgraph isomorphisms per node.

### Custom query graph and data graph for multi nodes

The users can run custom query graphs and data graphs using the following command

    mpirun -N <number of nodes> ./build/cuts_mpi <data_graph_path> <query_graph_path>
    
# Benchmarking platform and Dataset 

## Machine 1: 
* GPU: Nvidia Ampere A100 (108 SMs, 40 GB)
* OS:  Ubuntu 20.04 LTS
* CUDA: 11.0

## Machine 2: 
* GPU: Nvidia Volta V100(84 SMs, 32GB)
* OS:   Ubuntu 18.04.4 LTS
* CUDA: 10.2

## Dataset:
* See the dataset preparation.
# External Links
* https://github.com/pkumod/GSI

# LICENSE

Refer LICENSE in the root directory
