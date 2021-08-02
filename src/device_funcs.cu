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
#include "../inc/device_funcs.h"
__device__ bool compare_signature_gpu(unsigned int *sig1, unsigned int *sig2){
    for(unsigned int i=0;i<Signature_Properties;++i){
        if(sig2[i]<sig1[i]){
            return false;
        }
    }
    return true;
}
__device__ bool basic_search(unsigned int node, unsigned int *buffer, unsigned int len){
    for(unsigned int idx = 0;idx<len;idx++){
        if(node == buffer[idx]){
            return true;
        }
    }
    return false;
}
__device__ void memory_copy(unsigned int *from,unsigned int *to,unsigned int lane_id,unsigned int copy_len){
    for(unsigned int idx=lane_id;idx<copy_len;idx+=32){
        to[idx] = from[idx];
    }
}
__device__ void construct_pre_path(unsigned int *pre_copy,unsigned int *counter,unsigned int warp_id,unsigned int iter,
                                   unsigned int *lengths,unsigned int *result_table,unsigned int *index_table,
                                   unsigned int pre_idx,unsigned int lane_id){
    if(lane_id ==0){
        unsigned int vertexLocation;
        counter[warp_id] = 0;
        for (int i = iter - 1; i >= 0; i--) {
            if (i == iter - 1) {
                vertexLocation = lengths[i] + pre_idx;
            } else {
                vertexLocation = index_table[vertexLocation];
            }
            pre_copy[warp_id * QUERY_NODES + i] = result_table[vertexLocation];
        }
    }
    __syncwarp();
}

__device__ void initialize_intersection(unsigned int *pre_copy,unsigned int first_index,bool join_type,
                                        unsigned int * intersection,unsigned int *helper_buffer,
                                        unsigned int *neighbors_offset,unsigned int *r_neighbors_offset,
                                        unsigned int *neighbors,unsigned int *r_neighbors,unsigned int *q_sig,
                                        unsigned int *d_sig,unsigned int iter,unsigned int *counter,
                                        unsigned int warp_id,unsigned int lane_id,unsigned int p_count,unsigned int c_count,unsigned int *p_words){
    unsigned int *ns;
    unsigned int v = pre_copy[first_index];
    unsigned int start;
    unsigned int end;
    unsigned int words = 0;
    if(join_type){
        ns = neighbors;
        start = neighbors_offset[v];
        end = neighbors_offset[v+1];
    }else{
        ns = r_neighbors;
        start = r_neighbors_offset[v];
        end = r_neighbors_offset[v+1];
    }
    for(unsigned int i=lane_id+start;i<end;i+=32){
        unsigned int temp_node = ns[i];
        if(!basic_search(temp_node,pre_copy,iter)&&compare_signature_gpu(q_sig,&d_sig[temp_node*Signature_Properties])){
            unsigned int index = atomicAdd(&counter[warp_id],1);
            if(p_count > 0){
                words += d_sig[temp_node*Signature_Properties+In_degree_offset];
            }
            if(c_count >0){
                words += d_sig[temp_node*Signature_Properties+Out_degree_offset];
            }
            if(index>=MAX_NE){
                helper_buffer[index-MAX_NE] = temp_node;
            }else{
                intersection[index] = temp_node;
            }
        }
    }
    __syncwarp();
    for(int j=16;j>0;j=j/2){
        words += __shfl_down_sync(0xFFFFFFFF,words,j);
    }
    words = __shfl_sync(0xFFFFFFFF,words,0);
    (*p_words) = words;
}


__device__ void initialize_joins(unsigned int *parents_offset,unsigned int *children_offset,
                                 unsigned int *parents,unsigned int * children,unsigned int v,
                                 unsigned int *joins,bool * joins_type,unsigned int *join_len,
                                 unsigned int *parents_count,unsigned int lane_id){
    unsigned int joins_count = 0;
    unsigned int start = parents_offset[v];
    unsigned int end = parents_offset[v+1];
    joins_count += (end - start);
    if(lane_id == 0){
        *parents_count = (end - start);
    }
    for(unsigned int i=lane_id;i<end - start;i+=32){
        joins[i] = parents[start + i];
        joins_type[i] = true; // true parent type join
    }
    start = children_offset[v];
    end = children_offset[v+1];
    for(unsigned int i=lane_id;i<end - start;i+=32){
        joins[joins_count+i] = children[start + i];
        joins_type[joins_count+i] = false; // false children joins
    }
    joins_count += (end - start);
    join_len[0] = joins_count;

}
__device__ unsigned int C_intersect(unsigned int *joins,bool * joins_type,unsigned int join_len,
                                    unsigned int *pre_copy,unsigned int *neighbors,unsigned int *neighbors_offset,unsigned int *r_neighbors,
                                    unsigned int *r_neighbors_offset,unsigned int *intersect_buffer1,unsigned int *intersect_helper1,
                                    unsigned int *intersect_buffer2,unsigned int *intersect_helper2,unsigned int *counter,unsigned int lane_id,
                                    unsigned int least_index,unsigned int ini_inter_len){
    unsigned int *n_offset;
    unsigned int *ns;
    unsigned int intersection_len = ini_inter_len;
    unsigned int *intersect_result;
    unsigned int *intersect_result_helper;
    unsigned int *intersect_scratch;
    unsigned int *intersect_scratch_helper;
    unsigned int intersects = 0;
    for(unsigned int i=0;i<join_len;++i){
        if(i == least_index){
            continue;
        }
        if(intersects%2 == 0){
            intersect_result = intersect_buffer2;
            intersect_result_helper = intersect_helper2;
            intersect_scratch = intersect_buffer1;
            intersect_scratch_helper = intersect_helper1;
        }else{
            intersect_result = intersect_buffer1;
            intersect_result_helper = intersect_helper1;
            intersect_scratch = intersect_buffer2;
            intersect_scratch_helper = intersect_helper2;
        }
        intersects++;
        unsigned int v = pre_copy[joins[i]];
        if(joins_type[i]){
            n_offset = neighbors_offset;
            ns = neighbors;
        }else{
            n_offset = r_neighbors_offset;
            ns = r_neighbors;
        }
        unsigned int s = n_offset[v];
        unsigned int e = n_offset[v+1];
        for(unsigned int j=lane_id;j<(e-s);j+=32){
            unsigned int candidate = ns[s+j];
            if(intersection_len <=MAX_NE){
                if(basic_search(candidate,intersect_scratch,intersection_len)){
                    unsigned index = atomicAdd(&counter[0],1);
                    intersect_result[index] = candidate;
                }
            }else{
                if(basic_search(candidate,intersect_scratch,MAX_NE)|| basic_search(candidate,intersect_scratch_helper,
                                                                                   intersection_len - MAX_NE)){
                    unsigned index = atomicAdd(&counter[0],1);
                    if(index>=MAX_NE)
                        intersect_result_helper[index-MAX_NE] = candidate;
                    else
                        intersect_result[index] = candidate;
                }
            }
        }
        __syncwarp();
        if(lane_id == 0){
            intersection_len = counter[0];
            counter[0] = 0;
        }
        intersection_len = __shfl_sync(0xFFFFFFFF,intersection_len,0);
    }
    return intersection_len;
}
__device__ void P_intersect(unsigned int *pre_copy,unsigned int *joins,unsigned int lane_id,
                            unsigned int count,unsigned int * intersection_len,unsigned int *intersections,
                            unsigned int *helper_buffer,unsigned int *neighbors,unsigned int *offset){
    unsigned int g_index = 0;
    for(unsigned int i=0;i<(*intersection_len);i++){
        unsigned int sum = 0;
        unsigned int candidate;
        if(i<MAX_NE){
            candidate = intersections[i];
        }else{
            candidate = helper_buffer[i - MAX_NE];
        }
        unsigned int start = offset[candidate];
        unsigned int end = offset[candidate+1];
        for(unsigned int id=lane_id + start;id<end;id+=32){
            for(unsigned int j=0;j<count;++j){
                if(pre_copy[joins[j]] == neighbors[id]){
                    sum++;
                }
            }
        }
        for(int j=16;j>0;j=j/2){
            sum += __shfl_down_sync(0xFFFFFFFF,sum,j);
        }
        sum = __shfl_sync(0xFFFFFFFF,sum,0);
        if(sum == count){
            if(lane_id==0){
                if(g_index<MAX_NE){
                    intersections[g_index] = candidate;
                }else{
                    helper_buffer[g_index-MAX_NE] = candidate;
                }
            }
            g_index++;
        }
    }
    (*intersection_len) = g_index;
    (*intersection_len) = __shfl_sync(0xFFFFFFFF,*intersection_len, 0);
}
__device__ unsigned int find_vertex_has_least_degree(unsigned int *sigs,unsigned int *pre_copy,
                                                     unsigned int *joins,unsigned int join_len,bool *joins_type,unsigned int *ds,unsigned int lane_id){
    unsigned int least = 0;
    unsigned int least_count = 1000000000;
    for(unsigned int i=lane_id;i<join_len;i+=32){
        bool type = joins_type[i];
        unsigned int v = pre_copy[joins[i]];
        unsigned int neighbors_count;
        if(type){
            neighbors_count = sigs[v*Signature_Properties+Out_degree_offset];
        }else{
            neighbors_count = sigs[v*Signature_Properties+In_degree_offset];
        }
        (*ds)+= neighbors_count;
        if(neighbors_count < least_count){
            least_count = neighbors_count;
            least = i;
        }
    }
    for(int i = 16;i >= 1;i /= 2) {
        unsigned int temp_least_count = __shfl_xor_sync(0xFFFFFFFF, least_count, i, 32);
        unsigned int temp_least = __shfl_xor_sync(0xFFFFFFFF, least, i, 32);
        if (temp_least_count < least_count) {
            least_count = temp_least_count;
            least = temp_least;
        }
    }
    for(int j=16;j>0;j=j/2){
        (*ds) += __shfl_down_sync(0xFFFFFFFF,(*ds),j);
    }
    (*ds) = __shfl_sync(0xFFFFFFFF,(*ds), 0);
    least = __shfl_sync(0xFFFFFFFF,least, 0);
    return least;
}
__global__ void initialize_searching(unsigned int *qSignatures,unsigned int *dSignatures,
                                     unsigned int *result_table,unsigned int *orderSequence,unsigned int U,
                                     unsigned int *lengths,int world_size, int rank){
    unsigned int v = orderSequence[0];
    unsigned int globalIndex = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int totalThreads = 108*blockDim.x;
    for(unsigned int u = globalIndex * world_size + rank; u < U; u += totalThreads * world_size){
        if(compare_signature_gpu(&qSignatures[v*Signature_Properties],&dSignatures[u*Signature_Properties])){
            unsigned int index = atomicAdd(&lengths[1],1);
            result_table[index] = u;
        }
    }
}
__global__ void search_kernel(G_pointers q_p,G_pointers d_p,C_pointers c_p,S_pointers s_p,unsigned int U,unsigned int iter,
                              unsigned int jobs_count,unsigned int jobs_offset,unsigned int *global_count){
    __shared__ unsigned int signature[Signature_Properties];
    __shared__ unsigned int joins[2*QUERY_NODES];
    __shared__ bool joins_type[2*QUERY_NODES];
    __shared__ unsigned int intersect1[WARPS_EACH_BLK*MAX_NE];
    __shared__ unsigned int intersect2[WARPS_EACH_BLK*MAX_NE];
    __shared__ unsigned int pre_copy[WARPS_EACH_BLK*QUERY_NODES];
    __shared__ unsigned int counter[WARPS_EACH_BLK];
    __shared__ unsigned int parents_count;
    __shared__ unsigned int join_len;
    unsigned int warp_id = threadIdx.x/32;
    unsigned int lane_id = threadIdx.x%32;
    unsigned int v = c_p.order_sqeuence[iter];
    unsigned int intersection_len = 0;
    unsigned int global_idx = (blockIdx.x)*WARPS_EACH_BLK+warp_id;
    unsigned int helperOffset = global_idx * HelperSize;
    unsigned int mask = 0xFFFFFFFF;
    unsigned int *intersection_result;
    unsigned int *intersection_helper_result;
    if (warp_id == 0) {
        initialize_joins(c_p.parents_offset,c_p.children_offset,c_p.parents,c_p.children,iter,joins,joins_type,&join_len,&parents_count,lane_id);
    }
    if (warp_id == 0) {
        memory_copy(&q_p.signatures[v*Signature_Properties],signature,lane_id,Signature_Properties);
    }
    __syncthreads();
    while(true){
        unsigned int pre_idx;
        if(lane_id == 0){
            pre_idx = atomicAdd(&global_count[0],1);
        }
        pre_idx = __shfl_sync(mask,pre_idx,0);
        if(pre_idx>=jobs_count){
            break ;
        }
        pre_idx += jobs_offset;
        intersection_len = 0;
        unsigned int c_words = 0;
        unsigned int p_words = 0;
        unsigned int least_index = 0;
        construct_pre_path(pre_copy,counter,warp_id,iter,s_p.lengths,s_p.results_table,s_p.indexes_table,pre_idx,lane_id);
        least_index = find_vertex_has_least_degree(d_p.signatures,&pre_copy[warp_id*QUERY_NODES],joins,join_len,joins_type,&c_words,lane_id);
        initialize_intersection(&pre_copy[warp_id * QUERY_NODES],joins[least_index],joins_type[least_index],
                                &intersect1[warp_id*MAX_NE],&s_p.helper_buffer1[helperOffset],
                                d_p.neighbors_offset,d_p.r_neighbors_offset,d_p.neighbors,
                                d_p.r_neighbors,signature,d_p.signatures,
                                iter,counter,warp_id,lane_id,parents_count,join_len - parents_count,&p_words);
        intersection_len = counter[warp_id];
        __syncwarp();
        if(intersection_len == 0){
            continue;
        }
        //dynamic intersection method selection
        if(c_words < p_words/30){
            counter[warp_id] = 0;
            __syncwarp();
            intersection_len = C_intersect(joins,joins_type,join_len,&pre_copy[warp_id*QUERY_NODES],d_p.neighbors,d_p.neighbors_offset,d_p.r_neighbors,
                                           d_p.r_neighbors_offset,&intersect1[warp_id*MAX_NE],&s_p.helper_buffer1[helperOffset],
                                           &intersect2[warp_id*MAX_NE],&s_p.helper_buffer2[helperOffset],&counter[warp_id],lane_id,least_index,
                                           intersection_len);
            if(join_len%2 == 1){
                intersection_result = intersect1;
                intersection_helper_result = s_p.helper_buffer1;
            }else{
                intersection_result = intersect2;
                intersection_helper_result = s_p.helper_buffer2;
            }
        }else{
            if(parents_count>0){
                P_intersect(&pre_copy[warp_id*QUERY_NODES],joins,lane_id,parents_count,&intersection_len,&intersect1[warp_id*MAX_NE],
                            &s_p.helper_buffer1[helperOffset],d_p.r_neighbors,d_p.r_neighbors_offset);
            }
            if((join_len - parents_count)>0){
                P_intersect(&pre_copy[warp_id*QUERY_NODES],&joins[parents_count],lane_id,join_len - parents_count,
                            &intersection_len,&intersect1[warp_id*MAX_NE],&s_p.helper_buffer1[helperOffset],d_p.neighbors,d_p.neighbors_offset);
            }
            intersection_result = intersect1;
            intersection_helper_result = s_p.helper_buffer1;
        }
        if(intersection_len == 0){
            continue;
        }
        if(iter == q_p.V - 1){
            unsigned long long int write_pos;
            if(lane_id == 0){
                write_pos = atomicAdd(&s_p.write_pos[0],intersection_len+iter);
                unsigned long long int index_pos = atomicAdd(&s_p.indexes_pos[0],1);
                index_pos = 2*index_pos;
                s_p.final_results_row_ptrs[index_pos] = write_pos;
                s_p.final_results_row_ptrs[index_pos+1] = write_pos + intersection_len+iter;
            }
            if(lane_id == 1){
                atomicAdd(&s_p.final_count[0],intersection_len);
            }
            write_pos = __shfl_sync(0xFFFFFFFF,write_pos, 0);
            if(lane_id < 5){
                for(unsigned int i=lane_id;i<iter;i+=5){
                    s_p.final_results_table[write_pos+i] = pre_copy[warp_id*QUERY_NODES+i];
                }
            }else{
                unsigned int j = lane_id - 5;
                for(unsigned int i=j;i<intersection_len;i+=27){
                    unsigned int candidate;
                    if(i<MAX_NE){
                        candidate = intersection_result[warp_id*MAX_NE+i];
                    }else{
                        candidate = intersection_helper_result[helperOffset+i-MAX_NE];
                    }
                    s_p.final_results_table[write_pos+i+iter] = candidate;
                }
            }
        }else{
            unsigned int write_offset;
            for(unsigned int i=lane_id;i<intersection_len;i+=32){
                unsigned int candidate;
                if(i<MAX_NE){
                    candidate = intersection_result[warp_id*MAX_NE+i];
                }else{
                    candidate = intersection_helper_result[helperOffset+i-MAX_NE];
                }
                write_offset = atomicAdd(&s_p.lengths[iter+1],1);
                s_p.results_table[write_offset] = candidate;
                s_p.indexes_table[write_offset] = pre_idx + s_p.lengths[iter - 1];
            }
        }
    }
}
__device__ void construct_pre_path_virtual_warp(unsigned int *pre_copy,unsigned int *counter,unsigned int warp_id,
                                                unsigned int iter,unsigned int *lengths,unsigned int *result_table,
                                                unsigned int *index_table,unsigned int pre_idx,unsigned int lane_id,
                                                unsigned int mask){
    unsigned int useless = lane_id;
    if(lane_id ==0){
        unsigned int vertexLocation;
        counter[warp_id] = 0;
        for (int i = iter - 1; i >= 0; i--) {
            if (i == iter - 1) {
                vertexLocation = lengths[i] + pre_idx;
            } else {
                vertexLocation = index_table[vertexLocation];
            }
            pre_copy[warp_id * QUERY_NODES + i] = result_table[vertexLocation];
        }
    }
    useless = __shfl_sync(mask,useless,(warp_id%4)*8);
}
__device__ void initialize_intersection_virtual_warp(unsigned int *pre_copy,unsigned int first_index,bool join_type,
                                                     unsigned int *intersection,unsigned int *helper_buffer,
                                                     unsigned int *neighbors_offset,unsigned int *r_neighbors_offset,
                                                     unsigned int *neighbors,unsigned int *r_neighbors,unsigned int *q_sig,
                                                     unsigned int *d_sig,unsigned int iter,unsigned int *counter,
                                                     unsigned int warp_id,unsigned int lane_id,unsigned int mask){
    unsigned int *ns;
    unsigned int v = pre_copy[first_index];
    unsigned int start;
    unsigned int end;
    unsigned int useless = lane_id;
    if(join_type){
        ns = neighbors;
        start = neighbors_offset[v];
        end = neighbors_offset[v+1];
    }else{
        ns = r_neighbors;
        start = r_neighbors_offset[v];
        end = r_neighbors_offset[v+1];
    }
    for(unsigned int i=lane_id+start;i<end;i+=8){
        unsigned int temp_node = ns[i];
        if(!basic_search(temp_node,pre_copy,iter)&&compare_signature_gpu(q_sig,&d_sig[temp_node*Signature_Properties])){
            unsigned int index = atomicAdd(&counter[warp_id],1);
            if(index>=MAX_NE_V){
                helper_buffer[index-MAX_NE_V] = temp_node;
            }else{
                intersection[index] = temp_node;
            }
        }
    }
    useless = __shfl_sync(mask,useless,(warp_id%4)*8);
}
__device__ unsigned int C_intersect_virtual_warp(unsigned int *joins,bool * joins_type,unsigned int join_len,
                                                 unsigned int *pre_copy,unsigned int *neighbors,unsigned int *neighbors_offset,
                                                 unsigned int *r_neighbors,unsigned int *r_neighbors_offset,
                                                 unsigned int *intersect_buffer1,unsigned int *intersect_helper1,
                                                 unsigned int *intersect_buffer2,unsigned int *intersect_helper2,
                                                 unsigned int *counter,unsigned int lane_id,
                                                 unsigned int least_index,unsigned int ini_inter_len,unsigned int warp_id,
                                                 unsigned int mask){
    unsigned int *n_offset;
    unsigned int *ns;
    unsigned int intersection_len = ini_inter_len;
    unsigned int *intersect_result;
    unsigned int *intersect_result_helper;
    unsigned int *intersect_scratch;
    unsigned int *intersect_scratch_helper;
    unsigned int intersects = 0;
    for(unsigned int i=0;i<join_len;++i){
        if(i == least_index){
            continue;
        }
        if(intersects%2 == 0){
            intersect_result = intersect_buffer2;
            intersect_result_helper = intersect_helper2;
            intersect_scratch = intersect_buffer1;
            intersect_scratch_helper = intersect_helper1;
        }else{
            intersect_result = intersect_buffer1;
            intersect_result_helper = intersect_helper1;
            intersect_scratch = intersect_buffer2;
            intersect_scratch_helper = intersect_helper2;
        }
        intersects++;
        unsigned int v = pre_copy[joins[i]];
        if(joins_type[i]){
            n_offset = neighbors_offset;
            ns = neighbors;
        }else{
            n_offset = r_neighbors_offset;
            ns = r_neighbors;
        }
        unsigned int s = n_offset[v];
        unsigned int e = n_offset[v+1];
        for(unsigned int j=lane_id;j<(e-s);j+=8){
            unsigned int candidate = ns[s+j];
            if(intersection_len <=MAX_NE_V){
                if(basic_search(candidate,intersect_scratch,intersection_len)){
                    unsigned index = atomicAdd(&counter[0],1);
                    intersect_result[index] = candidate;
                }
            }else{
                if(basic_search(candidate,intersect_scratch,MAX_NE_V)|| basic_search(candidate,intersect_scratch_helper,
                                                                                     intersection_len - MAX_NE_V)){
                    unsigned index = atomicAdd(&counter[0],1);
                    if(index>=MAX_NE_V)
                        intersect_result_helper[index-MAX_NE_V] = candidate;
                    else
                        intersect_result[index] = candidate;
                }
            }
        }
        intersection_len = __shfl_sync(mask,intersection_len, (warp_id%4)*8);
        if(lane_id == 0){
            intersection_len = counter[0];
            counter[0] = 0;
        }
        intersection_len = __shfl_sync(mask,intersection_len, (warp_id%4)*8);
    }
    return intersection_len;
}
__device__ void P_intersect_virtual_warp(unsigned int *pre_copy,unsigned int *joins,unsigned int lane_id,
                                         unsigned int count,unsigned int * intersection_len,unsigned int *intersections,
                                         unsigned int *helper_buffer,unsigned int *neighbors,unsigned int *offset,
                                         unsigned int warp_id,unsigned int mask){
    unsigned int g_index = 0;
    for(unsigned int i=0;i<(*intersection_len);i++){
        unsigned int sum = 0;
        unsigned int candidate;
        if(i<MAX_NE_V){
            candidate = intersections[i];
        }else{
            candidate = helper_buffer[i - MAX_NE_V];
        }
        unsigned int start = offset[candidate];
        unsigned int end = offset[candidate+1];
        for(unsigned int id=lane_id + start;id<end;id+=8){
            for(unsigned int j=0;j<count;++j){
                if(pre_copy[joins[j]] == neighbors[id]){
                    sum++;
                }
            }
        }
        for(int j=4;j>0;j=j/2){
            sum += __shfl_down_sync(mask,sum,j);
        }
        sum = __shfl_sync(mask,sum,(warp_id%4)*8);
        if(sum == count){
            if(lane_id==0){
                if(g_index<MAX_NE_V){
                    intersections[g_index] = candidate;
                }else{
                    helper_buffer[g_index-MAX_NE_V] = candidate;
                }
            }
            g_index++;
        }
    }
    (*intersection_len) = g_index;
    (*intersection_len) = __shfl_sync(mask,*intersection_len,(warp_id%4)*8);
}
__device__ unsigned int find_vertex_has_least_degree_virtual_warp(unsigned int *sigs,unsigned int *pre_copy,
                                                                  unsigned int *joins,unsigned int join_len,bool *joins_type,
                                                                  unsigned int *ds,unsigned int lane_id,unsigned int warp_id,
                                                                  unsigned int mask){
    unsigned int least = 0;
    unsigned int least_count = 1000000000;
    for(unsigned int i=lane_id;i<join_len;i+=8){
        bool type = joins_type[i];
        unsigned int v = pre_copy[joins[i]];
        unsigned int neighbors_count;
        if(type){
            neighbors_count = sigs[v*Signature_Properties+Out_degree_offset];
        }else{
            neighbors_count = sigs[v*Signature_Properties+In_degree_offset];
        }
        (*ds)+= neighbors_count;
        if(neighbors_count < least_count){
            least_count = neighbors_count;
            least = i;
        }
    }
    for(int i = 4;i >= 1;i /= 2) {
        unsigned int temp_least_count = __shfl_xor_sync(mask, least_count, i, 8);
        unsigned int temp_least = __shfl_xor_sync(mask, least, i, 8);
        if (temp_least_count < least_count) {
            least_count = temp_least_count;
            least = temp_least;
        }
    }
    for(int j=4;j>0;j=j/2){
        (*ds) += __shfl_down_sync(mask,(*ds),j);
    }
    (*ds) = __shfl_sync(mask,(*ds), (warp_id%4)*8);
    least = __shfl_sync(mask,least, (warp_id%4)*8);
    return least;
}
__device__ unsigned int compute_mask(unsigned int id){
    unsigned int mask;
    if(id < 8){
        mask = 0x000000FF;
    }else if(id>=8&&id<16){
        mask = 0x0000FF00;
    }else if(id>=16&&id<24){
        mask = 0x00FF0000;
    }else{
        mask = 0xFF000000;
    }
    return mask;
}
__global__ void search_kernel_virtual_warp(G_pointers q_p,G_pointers d_p,C_pointers c_p,S_pointers s_p,
                                           unsigned int U,unsigned int iter,unsigned int jobs_count,
                                           unsigned int jobs_offset,unsigned int *global_count){
    __shared__ unsigned int signature[Signature_Properties];
    __shared__ unsigned int joins[2*QUERY_NODES];
    __shared__ bool joins_type[2*QUERY_NODES];
    __shared__ unsigned int intersect1[WARPS_EACH_BLK*MAX_NE_V*4];
    __shared__ unsigned int intersect2[WARPS_EACH_BLK*MAX_NE_V*4];
    __shared__ unsigned int pre_copy[WARPS_EACH_BLK*QUERY_NODES*4];
    __shared__ unsigned int counter[WARPS_EACH_BLK*4];
    __shared__ unsigned int parents_count;
    __shared__ unsigned int join_len;
    unsigned int warp_id = threadIdx.x/8;
    unsigned int lane_id = threadIdx.x%8;
    unsigned int v = c_p.order_sqeuence[iter];
    unsigned int intersection_len = 0;
    unsigned int global_idx = (blockIdx.x)*WARPS_EACH_BLK*4+warp_id;
    unsigned int helperOffset = global_idx * HelperSize;
    unsigned int mask = compute_mask(threadIdx.x%32);
    unsigned int *intersection_result;
    unsigned int *intersection_helper_result;
    if (threadIdx.x/32 == 0) {
        initialize_joins(c_p.parents_offset,c_p.children_offset,c_p.parents,c_p.children,iter,joins,joins_type,
                         &join_len,&parents_count,threadIdx.x%32);
    }
    if (threadIdx.x/32 == 1) {
        memory_copy(&q_p.signatures[v*Signature_Properties],signature,threadIdx.x%32,Signature_Properties);
    }
    __syncthreads();
    while(true){
        unsigned int pre_idx;
        if(lane_id == 0){
            pre_idx = atomicAdd(&global_count[0],1);
        }
        pre_idx = __shfl_sync(mask,pre_idx,(warp_id%4)*8);
        if(pre_idx>=jobs_count){
            break ;
        }
        pre_idx += jobs_offset;
        intersection_len = 0;
        unsigned int least_index = 0;
        unsigned int ds = 0;
        construct_pre_path_virtual_warp(pre_copy,counter,warp_id,iter,s_p.lengths,s_p.results_table,s_p.indexes_table,
                                        pre_idx,lane_id,mask);
        least_index = find_vertex_has_least_degree_virtual_warp(d_p.signatures,&pre_copy[warp_id*QUERY_NODES],
                                                                joins,join_len,joins_type,&ds,lane_id,warp_id,mask);
        initialize_intersection_virtual_warp(&pre_copy[warp_id * QUERY_NODES],joins[least_index],joins_type[least_index],
                                             &intersect1[warp_id*MAX_NE_V],&s_p.helper_buffer1[helperOffset],
                                             d_p.neighbors_offset,d_p.r_neighbors_offset,d_p.neighbors,
                                             d_p.r_neighbors,signature,d_p.signatures,
                                             iter,counter,warp_id,lane_id,mask);
        intersection_len = counter[warp_id];
        intersection_len = __shfl_sync(mask,intersection_len,(warp_id%4)*8);
        if(intersection_len == 0){
            continue;
        }
        //for graph with low degree, p_intersect should be better
        if(parents_count>0){
            P_intersect_virtual_warp(&pre_copy[warp_id*QUERY_NODES],joins,lane_id,parents_count,&intersection_len,
                                     &intersect1[warp_id*MAX_NE_V],&s_p.helper_buffer1[helperOffset],
                                     d_p.r_neighbors,d_p.r_neighbors_offset,warp_id,mask);
        }
        intersection_len = __shfl_sync(mask,intersection_len,(warp_id%4)*8);
        if((join_len - parents_count)>0){
            P_intersect_virtual_warp(&pre_copy[warp_id*QUERY_NODES],&joins[parents_count],lane_id,
                                     join_len - parents_count,&intersection_len,&intersect1[warp_id*MAX_NE_V],
                                     &s_p.helper_buffer1[helperOffset],d_p.neighbors,d_p.neighbors_offset,warp_id,mask);
        }
        intersection_result = intersect1;
        intersection_helper_result = s_p.helper_buffer1;
        intersection_len = __shfl_sync(mask,intersection_len,(warp_id%4)*8);
        if(iter == q_p.V - 1){
            unsigned long long int write_pos;
            if(lane_id == 0){
                write_pos = atomicAdd(&s_p.write_pos[0],intersection_len+iter);
                unsigned long long int index_pos = atomicAdd(&s_p.indexes_pos[0],1);
                index_pos = 2*index_pos;
                s_p.final_results_row_ptrs[index_pos] = write_pos;
                s_p.final_results_row_ptrs[index_pos+1] = write_pos + intersection_len+iter;
            }
            if(lane_id == 1){
                atomicAdd(&s_p.final_count[0],intersection_len);
            }
            write_pos = __shfl_sync(mask,write_pos,(warp_id%4)*8);
            if(lane_id < 5){
                for(unsigned int i=lane_id;i<iter;i+=5){
                    s_p.final_results_table[write_pos+i] = pre_copy[warp_id*QUERY_NODES+i];
                }
            }else{
                unsigned int j = lane_id - 5;
                for(unsigned int i=j;i<intersection_len;i+=3){
                    unsigned int candidate;
                    if(i<MAX_NE){
                        candidate = intersection_result[warp_id*MAX_NE+i];
                    }else{
                        candidate = intersection_helper_result[helperOffset+i-MAX_NE];
                    }
                    s_p.final_results_table[write_pos+i+iter] = candidate;
                }
            }
        }else{
            unsigned int write_offset;
            for(unsigned int i=lane_id;i<intersection_len;i+=8){
                unsigned int candidate;
                if(i<MAX_NE_V){
                    candidate = intersection_result[warp_id*MAX_NE_V+i];
                }else{
                    candidate = intersection_helper_result[helperOffset+i-MAX_NE_V];
                }
                write_offset = atomicAdd(&s_p.lengths[iter+1],1);
                s_p.results_table[write_offset] = candidate;
                s_p.indexes_table[write_offset] = pre_idx + s_p.lengths[iter - 1];
            }
        }
    }
}
