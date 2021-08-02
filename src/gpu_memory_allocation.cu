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
#include "../inc/gpu_memory_allocation.h"
inline void chkerr(cudaError_t code)
{
    if (code != cudaSuccess)
    {
        std::cout<<cudaGetErrorString(code)<<std::endl;
        exit(-1);
    }
}
void malloc_graph_gpu_memory(Graph &g,G_pointers &p){
    chkerr(cudaMalloc(&(p.neighbors),g.neighbors_offset[g.V]*sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.neighbors,g.neighbors,g.neighbors_offset[g.V]*sizeof(unsigned int),cudaMemcpyHostToDevice));
    chkerr(cudaMalloc(&(p.neighbors_offset),(g.V+1)*sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.neighbors_offset,g.neighbors_offset,(g.V+1)*sizeof(unsigned int),cudaMemcpyHostToDevice));
    chkerr(cudaMalloc(&(p.signatures),(g.V)*sizeof(unsigned int)*Signature_Properties));
    chkerr(cudaMemcpy(p.signatures,g.signatures,(g.V)*sizeof(unsigned int)*Signature_Properties,cudaMemcpyHostToDevice));
    chkerr(cudaMalloc(&(p.r_neighbors),g.r_neighbors_offset[g.V]*sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.r_neighbors,g.r_neighbors,g.r_neighbors_offset[g.V]*sizeof(unsigned int),cudaMemcpyHostToDevice));
    chkerr(cudaMalloc(&(p.r_neighbors_offset),(g.V+1)*sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.r_neighbors_offset,g.r_neighbors_offset,(g.V+1)*sizeof(unsigned int),cudaMemcpyHostToDevice));
    p.V = g.V;
}
void malloc_query_constraints_gpu_memory(Graph &g,C_pointers &p){
    chkerr(cudaMalloc(&(p.parents),g.parents_offset[g.V]*sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.parents,g.parents,g.parents_offset[g.V]*sizeof(unsigned int),cudaMemcpyHostToDevice));
    chkerr(cudaMalloc(&(p.parents_offset),(g.V+1)*sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.parents_offset,g.parents_offset,(g.V+1)*sizeof(unsigned int),cudaMemcpyHostToDevice));
    chkerr(cudaMalloc(&(p.children),g.children_offset[g.V]*sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.children,g.children,g.children_offset[g.V]*sizeof(unsigned int),cudaMemcpyHostToDevice));
    chkerr(cudaMalloc(&(p.children_offset),(g.V+1)*sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.children_offset,g.children_offset,(g.V+1)*sizeof(unsigned int),cudaMemcpyHostToDevice));
    chkerr(cudaMalloc(&(p.order_sqeuence),(g.V)*sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.order_sqeuence,g.order_sequence,(g.V)*sizeof(unsigned int),cudaMemcpyHostToDevice));
}
void malloc_other_searching_gpu_memory(S_pointers &p,unsigned int workers,unsigned int max_nodes){
    chkerr(cudaMallocManaged(&(p.lengths),(max_nodes+1)*sizeof(unsigned int)));
    chkerr(cudaMemset(p.lengths,0,(max_nodes+1)*sizeof(unsigned int)));
    chkerr(cudaMalloc(&(p.helper_buffer1),workers*HelperSize*sizeof(unsigned int)));
    chkerr(cudaMalloc(&(p.helper_buffer2),workers*HelperSize*sizeof(unsigned int)));
    unsigned int remaining_words = GPU_TABLE_LIMIT - workers*HelperSize;
    unsigned long long int table_size = remaining_words * sizeof(unsigned int);
    chkerr(cudaMalloc(&(p.results_table),table_size));
    chkerr(cudaMalloc(&(p.indexes_table),table_size));
    unsigned long long int cpu_table_size = CPU_FINAL_TABLE_SIZE * sizeof(unsigned int);
    chkerr(cudaMallocManaged(&(p.final_results_table),cpu_table_size));
    chkerr(cudaMallocManaged(&(p.final_count),sizeof(unsigned long long int)));
    p.final_count[0] = 0;
    chkerr(cudaMalloc(&(p.write_pos),sizeof(unsigned long long int)));
    chkerr(cudaMemset(p.write_pos,0,sizeof(unsigned long long int)));
    unsigned long long int row_ptrs_size = FINAL_RESULTS_ROW_PTR_SIZE * sizeof(unsigned long long int);
    chkerr(cudaMallocManaged(&(p.final_results_row_ptrs),row_ptrs_size));
    chkerr(cudaMallocManaged(&(p.indexes_pos),sizeof(unsigned long long int)));
    chkerr(cudaMemset(p.indexes_pos,0,sizeof(unsigned long long int)));
}
