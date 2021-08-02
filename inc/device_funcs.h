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

#ifndef CUTS_DEVICE_FUNCS_H
#define CUTS_DEVICE_FUNCS_H
#include "./common.h"
__device__ bool compare_signature_gpu(unsigned int *sig1, unsigned int *sig2);
__device__ bool basic_search(unsigned int node, unsigned int *buffer, unsigned int len);
__device__ void memory_copy(unsigned int *from,unsigned int *to,unsigned int lane_id,unsigned int copy_len);

__device__ void construct_pre_path(unsigned int *pre_copy,unsigned int *counter,unsigned int warp_id,unsigned int iter,
                                   unsigned int *lengths,unsigned int *result_table,unsigned int *index_table,
                                   unsigned int pre_idx,unsigned int lane_id);
__device__ void initialize_intersection(unsigned int *pre_copy,unsigned int first_index,bool join_type,
                                        unsigned int * intersection,unsigned int *helper_buffer,
                                        unsigned int *neighbors_offset,unsigned int *r_neighbors_offset,
                                        unsigned int *neighbors,unsigned int *r_neighbors,unsigned int *q_sig,
                                        unsigned int *d_sig,unsigned int iter,unsigned int *counter,
                                        unsigned int warp_id,unsigned int lane_id,unsigned int p_count,
                                        unsigned int c_count,unsigned int *p_words);


__device__ void initialize_joins(unsigned int *parents_offset,unsigned int *children_offset,
                                 unsigned int *parents,unsigned int * children,unsigned int v,
                                 unsigned int *joins,bool * joins_type,unsigned int *join_len,
                                 unsigned int *parents_count,unsigned int lane_id);

__device__ unsigned int C_intersect(unsigned int *joins,bool * joins_type,unsigned int join_len,
                                    unsigned int *pre_copy,unsigned int *neighbors,unsigned int *neighbors_offset,unsigned int *r_neighbors,
                                    unsigned int *r_neighbors_offset,unsigned int *intersect_buffer1,unsigned int *intersect_helper1,
                                    unsigned int *intersect_buffer2,unsigned int *intersect_helper2,unsigned int *counter,unsigned int lane_id,
                                    unsigned int least_index,unsigned int ini_inter_len);
__device__ void P_intersect(unsigned int *pre_copy,unsigned int *joins,unsigned int lane_id,
                            unsigned int count,unsigned int * intersection_len,unsigned int *intersections,
                            unsigned int *helper_buffer,unsigned int *neighbors,unsigned int *offset);
__device__ unsigned int find_vertex_has_least_degree(unsigned int *sigs,unsigned int *pre_copy,
                                                     unsigned int *joins,unsigned int join_len,bool *joins_type,unsigned int *ds,
                                                     unsigned int lane_id);

__global__ void initialize_searching(unsigned int *qSignatures,unsigned int *dSignatures,
                                     unsigned int *result_table,unsigned int *orderSequence,unsigned int U,
                                     unsigned int *lengths,int world_size, int rank);
__global__ void search_kernel(G_pointers q_p,G_pointers d_p,C_pointers c_p,S_pointers s_p,unsigned int U,unsigned int iter,
                              unsigned int jobs_count,unsigned int jobs_offset,unsigned int *global_count);
__device__ unsigned int compute_mask(unsigned int id);
__device__ void construct_pre_path_virtual_warp(unsigned int *pre_copy,unsigned int *counter,unsigned int warp_id,
                                                unsigned int iter,unsigned int *lengths,unsigned int *result_table,
                                                unsigned int *index_table,unsigned int pre_idx,unsigned int lane_id,
                                                unsigned int mask);
__device__ void initialize_intersection_virtual_warp(unsigned int *pre_copy,unsigned int first_index,bool join_type,
                                                     unsigned int *intersection,unsigned int *helper_buffer,
                                                     unsigned int *neighbors_offset,unsigned int *r_neighbors_offset,
                                                     unsigned int *neighbors,unsigned int *r_neighbors,unsigned int *q_sig,
                                                     unsigned int *d_sig,unsigned int iter,unsigned int *counter,
                                                     unsigned int warp_id,unsigned int lane_id,unsigned int mask);
__device__ unsigned int C_intersect_virtual_warp(unsigned int *joins,bool * joins_type,unsigned int join_len,
                                                 unsigned int *pre_copy,unsigned int *neighbors,unsigned int *neighbors_offset,
                                                 unsigned int *r_neighbors,unsigned int *r_neighbors_offset,
                                                 unsigned int *intersect_buffer1,unsigned int *intersect_helper1,
                                                 unsigned int *intersect_buffer2,unsigned int *intersect_helper2,
                                                 unsigned int *counter,unsigned int lane_id,
                                                 unsigned int least_index,unsigned int ini_inter_len,unsigned int warp_id,
                                                 unsigned int mask);
__device__ void P_intersect_virtual_warp(unsigned int *pre_copy,unsigned int *joins,unsigned int lane_id,
                                         unsigned int count,unsigned int * intersection_len,unsigned int *intersections,
                                         unsigned int *helper_buffer,unsigned int *neighbors,unsigned int *offset,
                                         unsigned int warp_id,unsigned int mask);
__device__ unsigned int find_vertex_has_least_degree_virtual_warp(unsigned int *sigs,unsigned int *pre_copy,
                                                                  unsigned int *joins,unsigned int join_len,bool *joins_type,
                                                                  unsigned int *ds,unsigned int lane_id,unsigned int warp_id,
                                                                  unsigned int mask);
__global__ void search_kernel_virtual_warp(G_pointers q_p,G_pointers d_p,C_pointers c_p,S_pointers s_p,
                                           unsigned int U,unsigned int iter,unsigned int jobs_count,
                                           unsigned int jobs_offset,unsigned int *global_count);
#endif //CUTS_DEVICE_FUNCS_H
