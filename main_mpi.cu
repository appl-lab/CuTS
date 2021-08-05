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
#include "./inc/device_funcs.h"
#include "./inc/gpu_memory_allocation.h"
#include "mpi.h"
#define PATHLEN 10
#define MAXNODES 8
#define MAXGB 1000000000
struct Point {
    unsigned int a, b, c, d, e;
};
int wsize;
int grank;
char msg_buffer[MAXNODES][100];
MPI_Request rq_send_msg[MAXNODES];
MPI_Request rq_recv_msg[MAXNODES];
unsigned int iter = 0;
bool global_free_list[MAXNODES];
inline void chkerr(cudaError_t code) {
    if (code != cudaSuccess) {
        std::cerr << "ERROR!!!:" << cudaGetErrorString(code) <<std::endl;
        exit(-1);
    }
}
int count_free_list() {
    int cnt = 0;
    for (int i = 0; i < wsize; ++i) {
        if (global_free_list[i]) {
            cnt++;
        }
    }
    return cnt;
}

void mpi_irecv(int src) {
    MPI_Irecv(msg_buffer[src], 1, MPI_CHAR, src, 0, MPI_COMM_WORLD,
              &rq_recv_msg[src]);

}
void mpi_isend(int dest, char *msg) {
    //MPI_Isend(msg, strlen(msg) + 1, MPI_CHAR, dest, 0, MPI_COMM_WORLD,
    MPI_Isend(msg, 1, MPI_CHAR, dest, 0, MPI_COMM_WORLD,
              &rq_send_msg[dest]);

}

void mpi_irecv_all(int rank) {
    for (int i = 0; i < wsize; i++) {
        if (i != rank) {
            mpi_irecv(i);
        }
    }
}

void mpi_isend_all(int rank, char *msg) {
    for (int i = 0; i < wsize; i++) {
        if (i != rank) {
            mpi_isend(i, msg);
        }
    }
}
bool take_work(int from, int rank, unsigned int *buffer) {
    /// first ask the other node to confirm that it has pending work
    /// it might have finished it by the time we received the processing request or
    /// someone else might have offered it help

    mpi_isend(from, "c"); //ask for confirmation
    MPI_Status status;
    char last_msg = 'r';
    while (last_msg == 'r') // the while loop ensures that multiple `r' requests are removed
    {
        MPI_Wait(&rq_recv_msg[from], &status); //blocking wait till we get a proper response
        last_msg = msg_buffer[from][0];
        mpi_irecv(from); ///initiate a request
    }
    if (last_msg == 'C') {
        mpi_isend_all(rank, "t");
        MPI_Datatype dt_point;
        MPI_Type_contiguous(5, MPI_UNSIGNED, &dt_point);
        MPI_Type_commit(&dt_point);
        MPI_Recv((Point *) buffer, MAXGB, dt_point, from, 1, MPI_COMM_WORLD, &status);
        global_free_list[rank] = false;
        return true;
    } else if (last_msg == 'f') {
        global_free_list[from] = true;
    }
    return false;
}
int take_work_wrap(int rank, unsigned int *buffer) {
    bool took_work = false;
    mpi_isend_all(rank, "f"); // see top of file for meaning
    global_free_list[rank] = true;
    while (!took_work && count_free_list() < wsize) //wait till we get someone request or till everyone completed job
    {
        for (int i = 0; i < wsize; i++) {
            if (i == rank) {
                continue;
            }
            MPI_Status status;
            int flag = 1;
            char last_msg = 'z'; //invalid message

            while (flag == 1) /// move forward till we find the last message
            {
                MPI_Test(&rq_recv_msg[i], &flag, &status); //check if we recvd a msg

                if (flag) {
                    last_msg = msg_buffer[i][0];
                    mpi_irecv(i); /// initiate new recv request again

                    if (last_msg == 'f') {
                        global_free_list[i] = true;
                    } else if (last_msg == 't') {
                        global_free_list[i] = false;
                    }
                }
            }
            if (last_msg == 'r')//someone is asking us to help to process their request...
            {
                if (!took_work) {
                    took_work = take_work(i, rank, buffer);
                }
            }
        }
    }
    return count_free_list();
}
void give_work(int rank, int taker, unsigned int *buffer) {
    MPI_Status status;
    mpi_isend(taker, "C"); /// send confirmation
    MPI_Wait(&rq_send_msg[taker], &status); //blocking wait till we send response
    /// At this point we know that the taker is waiting to recv data
    /// TODO WRITE CODE HERE to initiate data transfer
    /// USE TAG 1 for sync
    unsigned int giveSize;

    giveSize = ((buffer[0] + buffer[1])*2 + 2) / 5 + 1;
    MPI_Datatype dt_point;
    MPI_Type_contiguous(5, MPI_UNSIGNED, &dt_point);
    MPI_Type_commit(&dt_point);

    MPI_Send((Point *) buffer, giveSize, dt_point, taker, 1, MPI_COMM_WORLD);
}
bool check_for_confirmation(int rank, int &taker, unsigned int *buffer) {
    bool agreed_to_split_work = false;
    /// first try to respond all nodes which has send a confirmation request as all of them will be waiting
    for (int i = 0; i < wsize; i++) {
        if (i == rank) {
            continue;
        }
        MPI_Status status;
        int flag = true;
        char last_msg = 'z'; //invalid message

        while (flag) /// move forward till we find the last message
        {
            MPI_Test(&rq_recv_msg[i], &flag, &status); //check if we recvd a msg
            if (flag) {
                last_msg = msg_buffer[i][0];
                if (last_msg == 'f') {
                    global_free_list[i] = true;
                } else if (last_msg == 't') {
                    global_free_list[i] = false;
                }
                mpi_irecv(i); /// initiate new recv request again
            }
        }
        if (last_msg == 'c') //we found someone waiting for confirmation
        {
            if (!agreed_to_split_work) {
                give_work(rank, i, buffer); //give work to this node
                agreed_to_split_work = true;
                taker = i;
            } else {
                ///send decline
                mpi_isend(i, "D");
            }
        }
    }
    return agreed_to_split_work;
}
bool give_work_wrapper(int rank, int &taker, unsigned int *buffer) {
    bool agreed_to_split_work = check_for_confirmation(rank, taker, buffer);
    /// no one send confirmation
    if (!agreed_to_split_work) {
        for (int i = 0; i < wsize; i++) /// send a process request to all free nodes
        {
            if (i != rank && global_free_list[i]) {
                mpi_isend(i, "r");
            }
        }
        // retry to see someone send confirmation
        agreed_to_split_work = check_for_confirmation(rank, taker, buffer);
    }
    return agreed_to_split_work;
}

void encode_com_buffer(unsigned int *mpi_buffer,S_pointers s,unsigned iter,unsigned int buf_len){
    unsigned int pre_len = s.lengths[iter - 1];
    mpi_buffer[0] = pre_len;
    mpi_buffer[1] = buf_len;
    mpi_buffer[2] = iter;
    unsigned int copy_offset = 3;
    chkerr(cudaMemcpy(&mpi_buffer[copy_offset], s.results_table,pre_len * sizeof(unsigned int),
                      cudaMemcpyDeviceToHost));
    copy_offset+=(pre_len);
    chkerr(cudaMemcpy(&mpi_buffer[copy_offset], &s.results_table[pre_len+buf_len],
                      buf_len * sizeof(unsigned int),cudaMemcpyDeviceToHost));
    copy_offset+=buf_len;
    chkerr(cudaMemcpy(&mpi_buffer[copy_offset],s.indexes_table,pre_len * sizeof(unsigned int),
                      cudaMemcpyDeviceToHost));
    copy_offset+=pre_len;
    chkerr(cudaMemcpy(&mpi_buffer[copy_offset],&s.indexes_table[pre_len+buf_len],
                      buf_len * sizeof(unsigned int),cudaMemcpyDeviceToHost));
}
unsigned int decode_com_buffer(unsigned int *mpi_buffer,S_pointers &s){
    unsigned int pre_len = mpi_buffer[0];
    unsigned int buf_len = mpi_buffer[1];
    unsigned int iter = mpi_buffer[2];
    unsigned int copy_offset = 3;
    chkerr(cudaMemcpy(s.results_table,&mpi_buffer[copy_offset],(pre_len+buf_len) * sizeof(unsigned int),
                      cudaMemcpyHostToDevice));
    copy_offset+=(pre_len+buf_len);
    chkerr(cudaMemcpy(s.indexes_table,&mpi_buffer[copy_offset],(pre_len+buf_len) * sizeof(unsigned int),
                      cudaMemcpyHostToDevice));
    s.lengths[iter - 1] = pre_len;
    s.lengths[iter] = s.lengths[iter - 1] + buf_len;
    return iter;
}
void kernel_launch(G_pointers query_pointers,G_pointers data_pointers,C_pointers c_pointers,S_pointers &s_pointers,
                   unsigned int U,unsigned int iter,unsigned int jobs_count,
                   unsigned int jobs_offset,unsigned int *global_count,unsigned int avg_degree){
    if(avg_degree <= 3){
        search_kernel_virtual_warp<<<BLK_NUMS,BLK_DIM>>>(query_pointers,data_pointers,c_pointers,
                                                         s_pointers,U,iter,jobs_count,
                                                         jobs_offset,global_count);
    }else{
        search_kernel<<<BLK_NUMS,BLK_DIM>>>(query_pointers,data_pointers,c_pointers,
                                            s_pointers,U,iter,jobs_count,
                                            jobs_offset,global_count);
    }
}
unsigned long long int search_mpi(string query_file,string data_file,int world_size, int rank,bool write_to_disk) {
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

    unsigned int iters = query_graph.V;
    malloc_graph_gpu_memory(query_graph,query_pointers);
    malloc_graph_gpu_memory(data_graph,data_pointers);
    malloc_query_constraints_gpu_memory(query_graph,c_pointers);
    if(data_graph.AVG_DEGREE <= 3){
        malloc_other_searching_gpu_memory(s_pointers,BLK_NUMS*WARPS_EACH_BLK*4,query_graph.V);
    }else{
        malloc_other_searching_gpu_memory(s_pointers,BLK_NUMS*WARPS_EACH_BLK,query_graph.V);
    }
    unsigned int *global_count;
    unsigned int results_count;
    chkerr(cudaMalloc(&global_count,sizeof(unsigned int)));
    unsigned int *mpiCommBuffer = new unsigned int[8000000000];
    mpi_irecv_all(rank); // open communication channels
    for (int i = 0; i < wsize; ++i) {
        global_free_list[i] = false;
    }
    cudaEventRecord(event_start);
    initialize_searching<<<108,512>>>(query_pointers.signatures,data_pointers.signatures,s_pointers.results_table,
                                     c_pointers.order_sqeuence,data_graph.V,s_pointers.lengths,world_size,rank);
    chkerr(cudaDeviceSynchronize());
    unsigned int *cans_array = new unsigned int[s_pointers.lengths[1]];
    unsigned int ini_count = s_pointers.lengths[1];
    if(ini_count == 0){
        return 0;
    }
    chkerr(cudaMemcpy(cans_array,s_pointers.results_table,ini_count*sizeof(unsigned int),cudaMemcpyDeviceToHost));
    shuffle_array(cans_array,ini_count);
    bool helpOthers = false;
    cudaMemset(s_pointers.lengths,0,(iters+1)*sizeof(unsigned int));
    unsigned int trunk_size = 512;
    unsigned int num_trunks = (ini_count - 1)/trunk_size + 1;
    for(unsigned int l=0;l<num_trunks;++l){
        iter = 1;
        unsigned int t_size = trunk_size;
        if(l == num_trunks - 1){
            t_size = ini_count - l*trunk_size;
        }
        cudaMemset(s_pointers.lengths,0,(iters+1)*sizeof(unsigned int));
        s_pointers.lengths[1] = t_size;
        chkerr(cudaMemcpy(s_pointers.results_table,&cans_array[l*trunk_size],
                          t_size*sizeof(unsigned int),cudaMemcpyHostToDevice));
        helpOthers = false;
        do {
            int taker;
            bool divided_work;
            if (helpOthers && iter < iters) {
                cudaMemset(s_pointers.lengths,0,(iters+1)*sizeof(unsigned int));
                iter = decode_com_buffer(mpiCommBuffer,s_pointers);
            }
            for (;iter < iters; ++iter) {
                s_pointers.lengths[iter+1] = s_pointers.lengths[iter];
                cudaMemset(global_count,0,sizeof(unsigned int));
                unsigned int preCandidates = s_pointers.lengths[iter] - s_pointers.lengths[iter-1];
                if(preCandidates > 100000){
                    unsigned int miniBatchSize = preCandidates / 3;
                    cudaMemset(global_count,0,1*sizeof(unsigned int));
                    kernel_launch(query_pointers,data_pointers,c_pointers,
                                  s_pointers,data_graph.V,iter,miniBatchSize,
                                  0,global_count,data_graph.AVG_DEGREE);
                    chkerr(cudaDeviceSynchronize());
                    encode_com_buffer(mpiCommBuffer,s_pointers,iter,miniBatchSize);
                    divided_work = give_work_wrapper(grank, taker, mpiCommBuffer);
                    cudaMemset(global_count,0,1*sizeof(unsigned int));
                    kernel_launch(query_pointers,data_pointers,c_pointers,
                                  s_pointers,data_graph.V,iter,
                                  preCandidates-2*miniBatchSize,
                                  2*miniBatchSize,global_count,data_graph.AVG_DEGREE);
                    chkerr(cudaDeviceSynchronize());
                    if(!divided_work){
                        cudaMemset(global_count,0,1*sizeof(unsigned int));
                        kernel_launch(query_pointers,data_pointers,c_pointers,s_pointers,data_graph.V,iter,
                                      miniBatchSize,miniBatchSize,global_count,data_graph.AVG_DEGREE);
                    }
                }else{
                    cudaMemset(global_count,0,1*sizeof(unsigned int));
                    kernel_launch(query_pointers,data_pointers,c_pointers,
                                  s_pointers,data_graph.V,iter,preCandidates,
                                  0,global_count,data_graph.AVG_DEGREE);
                }
                chkerr(cudaDeviceSynchronize());
                results_count = s_pointers.lengths[iter+1] - s_pointers.lengths[iter];
                if (results_count == 0) {
                    iter = iters;
                    break;
                }
            }
            helpOthers = true;
        } while (wsize != take_work_wrap(rank, mpiCommBuffer));
    }
    cudaEventRecord(event_stop);
    cudaEventSynchronize(event_stop);
    float time_milli_sec = 0;
    cudaEventElapsedTime(&time_milli_sec, event_start, event_stop);
    cout<<rank<<","<<d_base_name<<","<<q_base_name<<","<<time_milli_sec<<"ms,"<<s_pointers.final_count[0]<<endl;
    if(write_to_disk){
        cout<<"start writting matching results to disk,ans.txt"<<endl;
        write_match_to_disk(s_pointers.indexes_pos[0],s_pointers.final_results_row_ptrs,query_graph.V,
                            query_graph.order_sequence,s_pointers.final_results_table);
        cout<<"finish writting matching results to disk,ans.txt"<<endl;
    }
    return s_pointers.final_count[0];
}
int main(int argc, char *argv[]) {
    MPI_Init(&argc,&argv);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    std::string query_graph_file = argv[2];
    std::string data_graph_file = argv[1];
    bool write_to_disk = false;
    unsigned long long int result_len = search_mpi(query_graph_file,data_graph_file,world_size,world_rank,write_to_disk);
    MPI_Finalize();
    return 0;
}
