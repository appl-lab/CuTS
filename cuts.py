#reproduce the result in our paper
import os,sys
import subprocess
import time
if __name__ == '__main__':
    data_graph_set = ['Enron.g','gowalla.g','wikiTalk.g','roadNetCa.g','roadNetPa.g','roadNetTx.g']
    query_graph_set = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]
    for i,d in enumerate(data_graph_set):
        data_graph = os.path.join('./data_set/data/ours_format',d)
        for j,q in enumerate(query_graph_set):
            if i <= 2 and j > 21:
                continue
            query_graph = os.path.join('./data_set/query/ours_format','{}.g'.format(q))
            try:
                if i <= 2:
                    subprocess.run(['./build/cuts',data_graph,query_graph,'4'])
                else:
                    subprocess.run(['./build/cuts',data_graph,query_graph])
            except:
                time.sleep(3)
                continue
