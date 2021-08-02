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
#ifndef CUTS_HOST_FUNCS_H
#define CUTS_HOST_FUNCS_H
#include "./gpu_memory_allocation.h"
#include "./device_funcs.h"
unsigned long long int search(string query_file,string data_file,bool write_to_disk);
void copy_graph_to_gpu(Graph query_graph,Graph data_graph,G_pointers &query_pointers,G_pointers &data_pointers,
                       C_pointers &c_pointers,S_pointers &s_pointers);
unsigned long long int search_dfs_bfs_strategy(string query_file,string data_file,bool write_to_disk,
                                               unsigned int trunks);
#endif //CUTS_HOST_FUNCS_H
