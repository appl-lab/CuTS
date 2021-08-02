/*
 * cuTS:  Scaling Subgraph Isomorphism on Distributed Multi-GPU Systems Using
 *        Trie Based Data Structure
 *
 * Copyright (C) 2021 APPL Laboratories (aravind_sr@outlook.com)
 *
 * This software is available under the MIT license, a copy of which can be
 * found in the file 'LICENSE' in the top-level directory.
 *
 * For further information contact:
 *   (1) Lizhi Xiang (lizhi.xiang@wsu.edu)
 *   (2) Aravind Sukumaran-Rajam (aravind_sr@outlook.com)
 *
 * The citation information is provided in the 'README' in the top-level
 * directory.
 */
#include "../inc/free_memories.h"
inline void chkerr(cudaError_t code)
{
    if (code != cudaSuccess)
    {
        std::cout<<cudaGetErrorString(code)<<std::endl;
        exit(-1);
    }
}
void free_graph_gpu_memory(G_pointers &p){
    chkerr(cudaFree(p.neighbors));
    chkerr(cudaFree(p.r_neighbors));
    chkerr(cudaFree(p.neighbors_offset));
    chkerr(cudaFree(p.r_neighbors_offset));
    chkerr(cudaFree(p.signatures));
}
void free_query_constraints_gpu_memory(C_pointers &p){
    chkerr(cudaFree(p.order_sqeuence));
    chkerr(cudaFree(p.children));
    chkerr(cudaFree(p.children_offset));
    chkerr(cudaFree(p.parents));
    chkerr(cudaFree(p.parents_offset));
}
void free_other_searching_gpu_memory(S_pointers &p){
    chkerr(cudaFree(p.indexes_table));
    chkerr(cudaFree(p.results_table));
    chkerr(cudaFree(p.helper_buffer1));
    chkerr(cudaFree(p.helper_buffer2));
    chkerr(cudaFree(p.lengths));
}