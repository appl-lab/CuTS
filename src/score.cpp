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
#include "../inc/score.h"
unsigned int get_score1(unsigned int *selected_nodes,unsigned int orders_len,unsigned int *neighbors,
                        unsigned int *offset,unsigned int v){
    unsigned int dout = 0;
    unsigned int din = 0;
    for(unsigned int idx = 0;idx < orders_len; ++idx){
        unsigned int node = selected_nodes[idx];
        if(binary_search(v,neighbors,offset[node],offset[node+1])){
            dout++;
        }
    }
    for(unsigned int idx = offset[v];idx<offset[v+1];idx++){
        unsigned int adj = neighbors[idx];
        if(binary_search(adj,selected_nodes,0,orders_len)){
            din++;
        }
    }
    return dout + din;
}
unsigned int get_score2(unsigned int *neighbors,unsigned int *selected_vertexes,unsigned int ordered_len,
                        unsigned int *offset, unsigned int v){
    unsigned int score = 0;
    std::set<unsigned int> non_visited_adjs;
    for(unsigned int idx=0;idx<ordered_len;++idx){
        unsigned int vertex = selected_vertexes[idx];
        for(unsigned int adj_idx = offset[vertex];adj_idx<offset[vertex+1];adj_idx++){
            unsigned int adj = neighbors[adj_idx];
            if(!binary_search(adj,selected_vertexes,0,ordered_len)){
                non_visited_adjs.insert(adj);
            }
        }
    }
    for(auto adj:non_visited_adjs){
        if(binary_search(adj,neighbors,offset[v],offset[v+1])){
            score++;
        }
    }
    return score;
}

unsigned int get_score3(const unsigned int *selected_vertexes,unsigned int sequence_len,
                        unsigned int *neighbors,unsigned int *offset, unsigned int v,unsigned int V){
    unsigned int score = 0;
    std::set<unsigned int> reachable_adjs;
    for(unsigned int idx=0;idx<sequence_len;++idx){
        unsigned int vertex = selected_vertexes[idx];
        reachable_adjs.insert(vertex);
        for(unsigned int adj_idx=offset[vertex];adj_idx<offset[vertex+1];adj_idx++){
            reachable_adjs.insert(neighbors[adj_idx]);
        }
    }
    std::set<unsigned int> non_reachable_vertex;
    for(unsigned int vertex=0;vertex<V;vertex++){
        if(reachable_adjs.find(vertex)==reachable_adjs.end()){
            non_reachable_vertex.insert(vertex);
        }
    }
    for(auto vertex: non_reachable_vertex){
        if(binary_search(vertex,neighbors,offset[v],offset[v+1])){
            score++;
        }
    }
    return score;
}
bool compare_score(Score S1, Score S2){
    if(S1.score1 > S2.score1){
        return true;
    }else if(S1.score1 == S2.score1){
        if(S1.score2>S2.score2){
            return true;
        }else if(S1.score2 == S2.score2){
            if(S1.score3>S2.score3){
                return true;
            }else{
                return false;
            }
        }else{
            return false;
        }
    }else{
        return false;
    }
}
