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
/*
 * For processing large data graphs, set the third argument first depth trunks to some unsigned integer number
 * (recommended 2-8).
 * The trunks is helpful to reduce GPU memory expansion for the intermediate results produced during matching.
 * For small data graphs, there is no need to set the third argument.
 */
#include "./inc/host_funcs.h"
int main(int argc, char *argv[]){
    if (argc < 3) {
        cout<<"args data_graph path,query_graph_path,first depth trunks(optional)"<<endl;
        exit(-1);
    }
    std::string query_graph_file = argv[2];
    std::string data_graph_file = argv[1];
    bool write_to_disk = false;
    if(argc == 3){
        unsigned long long int result_len = search(query_graph_file,data_graph_file,write_to_disk);
    }else{
        unsigned int trunks;
        try {
            trunks = atoi(argv[3]);
        }catch(int e){
            cout<<"invalid trunks, set trunks = 4"<<endl;
            trunks = 4;
        }
        unsigned long long int result_len = search_dfs_bfs_strategy(query_graph_file,data_graph_file,
                                                                    write_to_disk,trunks);
    }
    return 0;
}
