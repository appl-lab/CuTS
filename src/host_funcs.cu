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
#include "../inc/host_funcs.h"
#include "../inc/free_memories.h"
#include "../inc/gpu_memory_allocation.h"
inline void chkerr(cudaError_t code)
{
    if (code != cudaSuccess)
    {
        std::cout<<cudaGetErrorString(code)<<std::endl;
        exit(-1);
    }
}
void copy_graph_to_gpu(Graph query_graph,Graph data_graph,G_pointers &query_pointers,G_pointers &data_pointers,
                        C_pointers &c_pointers,S_pointers &s_pointers){
    malloc_graph_gpu_memory(query_graph,query_pointers);
    malloc_graph_gpu_memory(data_graph,data_pointers);
    malloc_query_constraints_gpu_memory(query_graph,c_pointers);
    if(data_graph.AVG_DEGREE <= 3){
        malloc_other_searching_gpu_memory(s_pointers,BLK_NUMS*WARPS_EACH_BLK*4,query_graph.V);
    }else{
        malloc_other_searching_gpu_memory(s_pointers,BLK_NUMS*WARPS_EACH_BLK,query_graph.V);
    }
}
unsigned long long int search_dfs_bfs_strategy(string query_file,string data_file,bool write_to_disk,unsigned int trunks){
    cout<<"start loading graph file from disk to memory..."<<endl;
    Graph query_graph(0,query_file);
    Graph data_graph(1,data_file);
    cout<<"graph loading complete..."<<endl;
    string q_base_name = base_name(query_file);
    string d_base_name = base_name(data_file);
    G_pointers query_pointers;
    G_pointers data_pointers;
    C_pointers c_pointers;
    S_pointers s_pointers;
    cudaEvent_t event_start;
    cudaEvent_t event_stop;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);
    cout<<"start copying graph to gpu..."<<endl;
    copy_graph_to_gpu(query_graph,data_graph,query_pointers,data_pointers,c_pointers,s_pointers);
    cout<<"finish copying graph to gpu..."<<endl;
    unsigned int iters = query_graph.V;
    unsigned int *global_count;
    cudaMalloc(&global_count,sizeof(unsigned int));
    cudaEventRecord(event_start);
    initialize_searching<<<108,512>>>(query_pointers.signatures,
                                      data_pointers.signatures,s_pointers.results_table,c_pointers.order_sqeuence,data_graph.V,
                                      s_pointers.lengths,1,0);
    chkerr(cudaDeviceSynchronize());
    if(s_pointers.lengths[1]>0&&iters>1){
        unsigned int trunk_size = (s_pointers.lengths[1] -1)/trunks + 1;
        unsigned int t_size = trunk_size;
        for(unsigned int i=0;i<trunks;++i){
            s_pointers.lengths[2] = s_pointers.lengths[1];
            if(i == trunks - 1){
                t_size = s_pointers.lengths[1] - i * trunk_size;
            }
            cudaMemset(global_count,0,sizeof(unsigned int));
            if(data_graph.AVG_DEGREE <= 3){
                search_kernel_virtual_warp<<<BLK_NUMS,BLK_DIM>>>(query_pointers,data_pointers,c_pointers,s_pointers,
                                                                 data_graph.V,1,t_size,i*trunk_size,
                                                                 global_count);
            }else{
                search_kernel<<<BLK_NUMS,BLK_DIM>>>(query_pointers,data_pointers,c_pointers,s_pointers,
                                                    data_graph.V,1,t_size,i*trunk_size,global_count);
            }
            chkerr(cudaDeviceSynchronize());
            for(unsigned int iter=2;iter<iters;iter++){
                s_pointers.lengths[iter+1] = s_pointers.lengths[iter];
                cudaMemset(global_count,0,sizeof(unsigned int));
                unsigned int jobs_count = s_pointers.lengths[iter] - s_pointers.lengths[iter-1];
                if(data_graph.AVG_DEGREE <= 3){
                    search_kernel_virtual_warp<<<BLK_NUMS,BLK_DIM>>>(query_pointers,data_pointers,c_pointers,s_pointers,
                                                                     data_graph.V,iter,jobs_count,0,global_count);
                }else{
                    search_kernel<<<BLK_NUMS,BLK_DIM>>>(query_pointers,data_pointers,c_pointers,s_pointers,data_graph.V,
                                                        iter,jobs_count,0,global_count);
                }
                chkerr(cudaDeviceSynchronize());
                unsigned int results_count = s_pointers.lengths[iter+1] - s_pointers.lengths[iter];
                if(results_count == 0){
                    break;
                }
            }
        }
    }
    cudaEventRecord(event_stop);
    cudaEventSynchronize(event_stop);
    float time_milli_sec = 0;
    cudaEventElapsedTime(&time_milli_sec, event_start, event_stop);
    cout<<d_base_name<<","<<q_base_name<<","<<time_milli_sec<<"ms,"<<s_pointers.final_count[0]<<endl;
    if(write_to_disk){
        cout<<"start writting matching results to disk,ans.txt"<<endl;
        write_match_to_disk(s_pointers.indexes_pos[0],s_pointers.final_results_row_ptrs,query_graph.V,
                                 query_graph.order_sequence,s_pointers.final_results_table);
        cout<<"finish writting matching results to disk,ans.txt"<<endl;
    }
    return s_pointers.final_count[0];
}
unsigned long long int search(string query_file,string data_file,bool write_to_disk){
    cout<<"start loading graph file from disk to memory..."<<endl;
    Graph query_graph(0,query_file);
    Graph data_graph(1,data_file);
    cout<<"graph loading complete..."<<endl;
    string q_base_name = base_name(query_file);
    string d_base_name = base_name(data_file);
    G_pointers query_pointers;
    G_pointers data_pointers;
    C_pointers c_pointers;
    S_pointers s_pointers;
    cudaEvent_t event_start;
    cudaEvent_t event_stop;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);
    cout<<"start copying graph to gpu..."<<endl;
    copy_graph_to_gpu(query_graph,data_graph,query_pointers,data_pointers,c_pointers,s_pointers);
    cout<<"end copying graph to gpu..."<<endl;
    unsigned int iters = query_graph.V;
    unsigned int *global_count;
    cudaMalloc(&global_count,sizeof(unsigned int));
    cudaEventRecord(event_start);
    initialize_searching<<<108,512>>>(query_pointers.signatures,data_pointers.signatures,s_pointers.results_table,
                                      c_pointers.order_sqeuence,data_graph.V,s_pointers.lengths,1,0);
    chkerr(cudaDeviceSynchronize());
    if(s_pointers.lengths[1]>0){
        for(unsigned int iter=1;iter<iters;iter++){
            s_pointers.lengths[iter+1] = s_pointers.lengths[iter];
            cudaMemset(global_count,0,sizeof(unsigned int));
            unsigned int jobs_count = s_pointers.lengths[iter] - s_pointers.lengths[iter-1];
            if(data_graph.AVG_DEGREE <= 3){
                search_kernel_virtual_warp<<<BLK_NUMS,BLK_DIM>>>(query_pointers,data_pointers,c_pointers,s_pointers,
                                                                 data_graph.V,iter,jobs_count,0,global_count);
            }else{
                search_kernel<<<BLK_NUMS,BLK_DIM>>>(query_pointers,data_pointers,c_pointers,s_pointers,data_graph.V,
                                                    iter,jobs_count,0,global_count);
            }
            chkerr(cudaDeviceSynchronize());
            unsigned int results_count = s_pointers.lengths[iter+1] - s_pointers.lengths[iter];
            if(results_count == 0){
                break;
            }
        }
    }
    cudaEventRecord(event_stop);
    cudaEventSynchronize(event_stop);
    float time_milli_sec = 0;
    cudaEventElapsedTime(&time_milli_sec, event_start, event_stop);
    cout<<d_base_name<<","<<q_base_name<<","<<time_milli_sec<<"ms,"<<s_pointers.final_count[0]<<endl;
    if(write_to_disk){
        cout<<"start writting matching results to disk,ans.txt"<<endl;
        write_match_to_disk(s_pointers.indexes_pos[0],s_pointers.final_results_row_ptrs,query_graph.V,
                            query_graph.order_sequence,s_pointers.final_results_table);
        cout<<"finish writting matching results to disk,ans.txt"<<endl;
    }
    return s_pointers.final_count[0];
}
