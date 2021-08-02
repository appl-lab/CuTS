#!/bin/sh
cd data_set/data
wget https://snap.stanford.edu/data/email-Enron.txt.gz
gunzip email-Enron.txt.gz
wget https://snap.stanford.edu/data/loc-gowalla_edges.txt.gz
gunzip loc-gowalla_edges.txt.gz
wget https://snap.stanford.edu/data/wiki-Talk.txt.gz
gunzip wiki-Talk.txt.gz
wget https://snap.stanford.edu/data/roadNet-CA.txt.gz
gunzip roadNet-CA.txt.gz
wget https://snap.stanford.edu/data/roadNet-PA.txt.gz
gunzip roadNet-PA.txt.gz
wget https://snap.stanford.edu/data/roadNet-TX.txt.gz
gunzip roadNet-TX.txt.gz
python cuts_graph_converter.py