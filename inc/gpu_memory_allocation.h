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

#ifndef CUTS_GPU_MEMORY_ALLOCATION_H
#define CUTS_GPU_MEMORY_ALLOCATION_H
#include "./graph.h"
void malloc_graph_gpu_memory(Graph &g,G_pointers &p);
void malloc_query_constraints_gpu_memory(Graph &g,C_pointers &p);
void malloc_other_searching_gpu_memory(S_pointers &p,unsigned int workers,unsigned int max_nodes);
#endif //CUTS_GPU_MEMORY_ALLOCATION_H
