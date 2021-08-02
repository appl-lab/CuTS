import codecs
import sys,os
def convert(data_file):
    write_lines = ['t # 0\n']
    reader = codecs.open(data_file,'r','utf-8')
    lines = reader.readlines()
    vertexes_nums = int(lines[0])
    write_lines.append('{} {} {} {}\n'.format(vertexes_nums,len(lines)-1,1,1))
    for i in range(0,vertexes_nums):
        line = '{} {} {}\n'.format('v',i,1)
        write_lines.append(line)
    for line in lines[1:]:
        s = int(line.split('\t')[0])
        e = int(line.split('\t')[1])
        line = 'e {} {} {}\n'.format(s,e,1)
        write_lines.append(line)
    write_lines.append('t # -1\n')
    gsi_write_file = os.path.join('/home/lizhi/research/cuts/sm_dataset/query/gsi_format/7nodes',os.path.basename(data_file))
    with codecs.open(gsi_write_file,'w+','utf-8') as writter:
        for line in write_lines:
            writter.write(line)
if __name__ == '__main__':
    query_dir = '/home/lizhi/research/cuts/sm_dataset/query/ours_format/7nodes'
    files = os.listdir(query_dir)
    for file in files:
        if '.g' in file:
           try:
              convert(os.path.join(query_dir,file))
           except:
              print(file)
