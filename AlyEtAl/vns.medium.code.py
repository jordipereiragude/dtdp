import sys
import time
import networkx as nx

def print_graph(G):
    print(len(G))
    print(len(G.edges))
    print(3)
    print(10)
    print(0.05)
    for n in G.nodes():
        print(n,G.nodes[n]['workload'], G.nodes[n]['demand'], G.nodes[n]['n_customers'])
    for e in G.edges():
        print(e[0],e[1],G.edges[e]['distance'])

def print_districts(G, districts, pos=None):

    totalWorkload = 0
    totalDemand = 0
    totalNCustomers = 0
    for k, nodes in districts.items():
        #print(f"District {k}: {nodes}",end=" ")
        longest_path = 0
        acc_workload = 0
        acc_demand = 0
        acc_n_customers = 0

        for i, source in enumerate(nodes):
            acc_workload += G.nodes[source].get("workload", 1)
            totalWorkload += G.nodes[source].get("workload", 1)
            acc_demand += G.nodes[source].get("demand", 1)
            totalDemand += G.nodes[source].get("demand", 1)
            acc_n_customers += G.nodes[source].get("n_customers", 1)
            totalNCustomers += G.nodes[source].get("n_customers", 1)
            # compute shortest distances from source to all other nodes
            lengths = nx.single_source_dijkstra_path_length(G, source, weight="distance")
            #lengths = nx.single_source_dijkstra_path_length(G, source)
            for target in nodes[i+1:]:
                if target in lengths:
                    #print(f"  Shortest path from {source} to {target}: length {lengths[target]}")
                    if lengths[target] > longest_path:
                        longest_path = lengths[target]
                else:
                    print(f"  No path from {source} to {target} within district {k}")
        print(f"Longest shortest path in district {k}: {longest_path}, wk: {acc_workload}, dem: {acc_demand}, n_cust: {acc_n_customers}")
    print(f"Total workload: {totalWorkload}, Total demand: {totalDemand}, Total n_customers: {totalNCustomers}")

start = time.time()

G = nx.read_graphml(sys.argv[1])

# """ resolucion
from DTDPAlgorithms import TerritoryDesignProblem, BVNS

tdp = TerritoryDesignProblem(
    graph_input=G,
    delta=0.05,          # balance tolerance between districts
    llambda=0.7,         # weight between dispersion vs. balance
    rcl_parameter=0.5,   # restricted candidate list threshold
    nr_districts=10
)

bvns = BVNS(tdp_instance=tdp, shaking_steps=10, fail_max=10, nrInitSolutions=100)
obj_hist, inf_hist, best_solution, timeline = bvns.performBVNS()
end = time.time()
print("Instance:",sys.argv[1],"Best objective:", obj_hist[-1], "Infeasibility:", inf_hist[-1],"Total time (s):", end - start)

