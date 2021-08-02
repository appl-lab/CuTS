import codecs
import os
snap_graphs = ['email-Enron.txt','loc-gowalla_edges.txt','roadNet-CA.txt','roadNet-PA.txt','roadNet-TX.txt','wiki-Talk.txt']
cuts_graphs = ['Enron.g','gowalla.g','roadNetCa.g','roadNetPa.g','roadNetTx.g','wikiTalk.g']
for i,s in enumerate(snap_graphs):
    lines = []
    with codecs.open(s,'r','utf-8') as reader:
        temp_lines = reader.readlines()
        for line in temp_lines:
            if '#' in line:
                continue
            lines.append(line)
    max_node = 0
    for line in lines:
        parts = line.split('\t')
        s = int(parts[0])
        e = int(parts[1])
        t = max(s,e)
        if max_node < t:
            max_node = t
    cuts_file = './ours_format/{}'.format(cuts_graphs[i])
    with codecs.open(cuts_file,'w+','utf-8') as writter:
        writter.write('{}\n'.format(max_node+1))
        for line in lines:
            writter.write(line)
    print("finished achieving {} graph,{} graphs remaining".format(os.path.basename(cuts_file),5 - i))
for s in snap_graphs:
    os.remove(s)