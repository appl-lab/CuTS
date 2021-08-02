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

#ifndef CUTS_GRAPH_H
#define CUTS_GRAPH_H
#include "./score.h"
#include "./util.h"
class Graph{
public:
    unsigned int V;
    unsigned int E;
    unsigned int AVG_DEGREE = 0;
    unsigned int * neighbors;
    unsigned int * r_neighbors;
    unsigned int * neighbors_offset;
    unsigned int * r_neighbors_offset;
    unsigned int * parents_offset;
    unsigned int * parents;
    unsigned int * children;
    unsigned int * children_offset;
    unsigned int * order_sequence;
    unsigned int * signatures;
    void sort_search_order(vector< set<unsigned int> > ns,vector< set<unsigned int> > r_ns);
    Graph(unsigned int mode,std::string input_file);
    //mode 0 for query graph, mode 1 for data graph
};
#endif //CUTS_GRAPH_H
