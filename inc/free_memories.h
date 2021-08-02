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
#ifndef CUTS_FREE_MEMORIES_H
#define CUTS_FREE_MEMORIES_H
#include "./common.h"
void free_graph_gpu_memory(G_pointers &p);
void free_query_constraints_gpu_memory(C_pointers &p);
void free_other_searching_gpu_memory(S_pointers &p);
#endif //CUTS_FREE_MEMORIES_H
