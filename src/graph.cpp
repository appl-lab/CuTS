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
#include "../inc/graph.h"
Graph::Graph(unsigned int mode,std::string input_file){
    vector< set<unsigned int> > ns;
    vector< set<unsigned int> > r_ns;
    V = file_reader(input_file,ns,r_ns);
    signatures = new unsigned int[V*Signature_Properties];
    order_sequence = new unsigned int[V];
    #pragma omp parallel for
    for(int i=0;i<V;++i){
        signatures[i*Signature_Properties+In_degree_offset] = r_ns[i].size();
        signatures[i*Signature_Properties+Out_degree_offset] = ns[i].size();
    }
    neighbors_offset = new unsigned int[V+1];
    r_neighbors_offset = new unsigned int[V+1];
    #pragma omp parallel for
    for(unsigned int i=0;i<V+1;++i){
        neighbors_offset[i] = 0;
        r_neighbors_offset[i] = 0;
    }
    for(unsigned int i=1;i<V+1;++i){
        neighbors_offset[i] += neighbors_offset[i-1] + ns[i-1].size();
        r_neighbors_offset[i] += r_neighbors_offset[i-1] + r_ns[i-1].size();
    }
    E = neighbors_offset[V];
    neighbors = new unsigned int[neighbors_offset[V]];
    r_neighbors = new unsigned int[r_neighbors_offset[V]];
    AVG_DEGREE = E/V + 2;
    unsigned int j = 0;
    unsigned int k = 0;
    for(unsigned int i=0;i<V;++i){
        std::set<unsigned int> s = ns[i];
        for(std::set<unsigned int>::iterator p = s.begin();p!=s.end();p++){
            neighbors[j] = *p;
            j++;
        }
        s = r_ns[i];
        for(std::set<unsigned int>::iterator p = s.begin();p!=s.end();p++){
            r_neighbors[k] = *p;
            k++;
        }
    }
    if(!mode){
        sort_search_order(ns,r_ns);
    }
}
void Graph::sort_search_order(vector< set<unsigned int> > ns,vector< set<unsigned int> > r_ns){
    unsigned int max_out_degree = 0;
    unsigned int idx;
    parents_offset = new unsigned int[V+1];
    children_offset = new unsigned int[V+1];
    parents_offset[0] = children_offset[0] = 0;
    for(unsigned int v=0;v<V;++v){
        parents_offset[v+1] = children_offset[v+1] = 0;
        if(max_out_degree < signatures[v*Signature_Properties+Out_degree_offset]){
            max_out_degree = signatures[v*Signature_Properties+Out_degree_offset];
            idx = v;
        }
    }
    order_sequence[0] = idx;
    unsigned int inserted_vertexes = 1;
    while(inserted_vertexes < V){
        Score max_score(0,0,0);
        for(unsigned int v = 0;v<V;++v){
            if(binary_search(v,order_sequence,0,inserted_vertexes)){
                continue;
            }
            unsigned int score1 = get_score1(order_sequence,inserted_vertexes,neighbors,neighbors_offset,v);
            unsigned int score2 = get_score2(neighbors,order_sequence,inserted_vertexes,neighbors_offset,v);
            unsigned int score3 = get_score3(order_sequence,inserted_vertexes,neighbors,neighbors_offset,v,V);
            Score temp_score(score1,score2,score3);
            if(compare_score(temp_score,max_score)){
                max_score = temp_score;
                idx = v;
            }
        }
        order_sequence[inserted_vertexes++] = idx;
    }
    vector<set<unsigned int> > P;
    vector<set<unsigned int> > C;
    for(unsigned int i=0;i<V;++i){
        set<unsigned int> temp1;
        set<unsigned int> temp2;
        P.push_back(temp1);
        C.push_back(temp2);
    }
    for(unsigned int i=1;i<V;++i){
        unsigned int v = order_sequence[i];
        for(unsigned int j=0;j<i;++j){
            unsigned int t_v = order_sequence[j];
            if(ns[v].find(t_v) != ns[v].end()){
                C[i].insert(j);
            }
            if(r_ns[v].find(t_v)!=r_ns[v].end()){
                P[i].insert(j);
            }
        }
    }
    for(unsigned int i=1;i<V+1;++i){
        parents_offset[i] += parents_offset[i-1] + P[i-1].size();
        children_offset[i] += children_offset[i-1] + C[i-1].size();
    }
    parents = new unsigned int[parents_offset[V]];
    children = new unsigned int[children_offset[V]];
    unsigned int j = 0;
    unsigned int k = 0;
    for(unsigned int i=0;i<V;++i){
        std::set<unsigned int> s = P[i];
        for(std::set<unsigned int>::iterator p = s.begin();p!=s.end();p++){
            parents[j] = *p;
            j++;
        }
        s = C[i];
        for(std::set<unsigned int>::iterator p = s.begin();p!=s.end();p++){
            children[k] = *p;
            k++;
        }
    }
}
