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
#include "../inc/common.h"

string base_name(string s) {
    char sep = '/';
    size_t i = s.rfind(sep, s.length());
    if (i != string::npos) {
        return(s.substr(i+1, s.length() - i));
    }
    return("");
}
bool binary_search(unsigned int v,const unsigned int *array, unsigned int start, unsigned int end){
    for(unsigned int i=start;i<end;++i){
        if(array[i]==v){
            return true;
        }
    }
    return false;
}
bool compare_signature(unsigned int *sig1, unsigned int *sig2){
    for(unsigned int i=0;i<Signature_Properties;++i){
        if(sig2[i]<sig1[i]){
            return false;
        }
    }
    return true;
}
void shuffle_array(unsigned int arr[], unsigned int n)
{
    srand (time(NULL));
    unsigned seed = rand() % 100 + 1;
    shuffle(arr, arr + n,default_random_engine(seed));
}
