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

#include "../inc/util.h"
unsigned int file_reader(string input_file,vector<set<unsigned int> > &neighbors,vector<set<unsigned int> > &r_neighbors){
    double load_start = omp_get_wtime();
    ifstream infile;
    infile.open(input_file);
    if(!infile){
        cout<<"load graph file failed "<<endl;
        exit(-1);
    }
    unsigned int V  = 0;
    string line;
    const std::string delimter = "\t";
    unsigned int line_index = 0;
    getline(infile,line);
    V = stoi(line);
    for(unsigned int i=0;i<V;++i){
        set<unsigned int> temp1;
        set<unsigned int> temp2;
        neighbors.push_back(temp1);
        r_neighbors.push_back(temp2);
    }
    while(getline(infile,line)){
        auto pos = line.find(delimter);
        if(pos == std::string::npos){
            continue;
        }
        int s = stoi(line.substr(0, pos));
        int t = stoi(line.substr(pos + 1, line.size() - pos - 1));
        neighbors[s].insert(t);
        r_neighbors[t].insert(s);
    }
    infile.close();
    double load_end = omp_get_wtime();
    return V;
}
void write_match_to_disk(unsigned long long int count,unsigned long long int *row_ptrs,unsigned int V,
                         unsigned int *order_sequence,unsigned int *results){
    std::ofstream out("ans.txt");
    for(unsigned long long int i=0;i<count;++i){
        string prefix_line = "";
        unsigned long long int start_pos = row_ptrs[2*i];
        unsigned long long int end_pos = row_ptrs[2*i+1];
        for(unsigned long long int j = start_pos;j<start_pos+V-1;++j){
            unsigned int query_v = order_sequence[j - start_pos];
            unsigned int matched_v = results[j];
            string match_map = "("+to_string(query_v)+","+ to_string(matched_v)+")";
            prefix_line +=match_map;
        }
        for(unsigned long long int j = start_pos+V-1;j<end_pos;++j){
            unsigned int query_v = order_sequence[V - 1];
            unsigned int matched_v = results[j];
            string match_map = "("+to_string(query_v)+","+ to_string(matched_v)+")";
            string result_line = prefix_line + match_map;
            out<<result_line<<endl;
        }
    }
}
