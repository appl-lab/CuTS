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
#ifndef CUTS_COMMON_H
#define CUTS_COMMON_H
#define Signature_Properties 2
#define In_degree_offset 0
#define Out_degree_offset 1
#define BLK_NUMS 108
#define BLK_DIM 1024
#define WARPS_EACH_BLK (BLK_DIM/32)
#define WORK_UNITS (BLK_NUMS*WARPS_EACH_BLK)
#define GPU_TABLE_LIMIT 3750000000
#define CPU_FINAL_TABLE_SIZE 50000000000
#define FINAL_RESULTS_ROW_PTR_SIZE 10000000000
#define HelperSize 15000
#define QUERY_NODES 12
#define MAX_NE 112
#define MAX_NE_V 40
#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <fstream>
#include <map>
#include <utility>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <unordered_map>
#include <stack>
#include <deque>
#include <random>
#include <cuda.h>
#include <bits/stdc++.h>
#include "omp.h"

using namespace std;

bool binary_search(unsigned int v,const unsigned int *array, unsigned int start, unsigned int end);
bool compare_signature(unsigned int *sig1, unsigned int *sig2);
void shuffle_array(unsigned int arr[], unsigned int n);
string base_name(string s);
typedef struct G_pointers {
    unsigned int* neighbors;
    unsigned int* neighbors_offset;
    unsigned int* signatures;
    unsigned int* r_neighbors;
    unsigned int* r_neighbors_offset;
    unsigned int V;
} G_pointers;//graph related
typedef struct C_pointers{
    unsigned int* parents;
    unsigned int* parents_offset;
    unsigned int* children;
    unsigned int* children_offset;
    unsigned int* order_sqeuence;
} C_pointers;//construct related
typedef struct S_pointers{
    unsigned int* lengths;
    unsigned int* results_table;
    unsigned int* indexes_table;
    unsigned int* helper_buffer1;
    unsigned int* helper_buffer2;
    unsigned int* final_results_table;
    unsigned long long int *final_count;
    unsigned long long int *write_pos;
    unsigned long long int *final_results_row_ptrs;
    unsigned long long int *indexes_pos;
} S_pointers;//searching kernel required
#endif //CUTS_COMMON_H
