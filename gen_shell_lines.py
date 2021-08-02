ds = ['Enron.g','gowalla.g','roadNetCa.g','roadNetPa.g','roadNetTx.g','wikiTalk.g']
qs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]
#mpirun $prefix/gm_distributed/build/gm $prefix/gm_distributed/sm_dataset/data/ours_format/gowalla.g $prefix/gm_distributed/sm_dataset/query/ours_format/7nodes/5.g
for d in ds:
    data_graph = '/data_set/data/ours_format/{}'.format(d)
    for q in qs:
        query_graph = '/data_set/query/ours_format/{}.g'.format(q)
        print('mpirun $prefix/cuts/build/cuts_mpi $prefix/cuts{} $prefix/cuts{}'.format(data_graph,query_graph))
