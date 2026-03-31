import sys
import networkx as nx

def print_graph(G,filename):
    #open filename for writing
    with open(filename,'w') as f:
        f.write(str(len(G))+"\n")
        f.write(str(len(G.edges))+"\n")
        f.write("3\n")
        f.write("10\n")
        f.write("0.05\n")
        for n in G.nodes():
            f.write(str(n)+" "+str(G.nodes[n]['workload'])+" "+str(G.nodes[n]['demand'])+" "+str(G.nodes[n]['n_customers'])+"\n")
        for e in G.edges():
            f.write(str(e[0])+" "+str(e[1])+" "+str(G.edges[e]['distance'])+"\n")

if len(sys.argv)<3:
    print("Usage: python transform.py <graphfile> <outfile>")
else:
    #G = nx.read_graphml("TGraphInstances/planar500_G0.graphml")
    G = nx.read_graphml(sys.argv[1])
    print_graph(G,sys.argv[2])

