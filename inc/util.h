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

#ifndef CUTS_UTIL_H
#define CUTS_UTIL_H
#include "./common.h"
unsigned int file_reader(string input_file,vector<set<unsigned int> > &neighbors,vector<set<unsigned int> > &r_neighbors);
void write_match_to_disk(unsigned long long int count,unsigned long long int *row_ptrs,unsigned int V,
                         unsigned int *order_sequence,unsigned int *results);
#endif //CUTS_UTIL_H
