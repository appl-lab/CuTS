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
#ifndef CUTS_SCORE_H
#define CUTS_SCORE_H
#include "./common.h"
class Score{
public:
    unsigned int score1;
    unsigned int score2;
    unsigned int score3;
    Score(unsigned int a, unsigned int b, unsigned int c){
        score1 = a;
        score2 = b;
        score3 = c;
    }
};
bool compare_score(Score S1, Score S2);
unsigned int get_score1(unsigned int *selected_nodes,unsigned int orders_len,unsigned int *neighbors,
                        unsigned int *offset,unsigned int v);
unsigned int get_score2(unsigned int *neighbors,unsigned int *selected_vertexes,unsigned int ordered_len,
                        unsigned int *offset, unsigned int v);
unsigned int get_score3(const unsigned int *selected_vertexes,unsigned int sequence_len,
                        unsigned int *neighbors,unsigned int *offset, unsigned int v,unsigned int V);
#endif //CUTS_SCORE_H
