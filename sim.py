import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import networkx as nx

def generate_topology(degree):
    # Create topology without connections
    num_nodes = degree * 4
    topology = np.zeros((num_nodes, num_nodes))

    for n in range(num_nodes):
        # Add connections to adjacent nodes at same level
        if n%4 == 0:
            topology[n][n+1] = 1
            topology[n][n+3] = 1
        elif n%4 == 1 or n%4 == 2:
            topology[n][n+1] = 1
        
        # Add connection to router in outer ring
        if math.floor(n/4) < degree-1:
            topology[n][n+4] = 1

    return topology

def split_network(network):
    # Determine the border router and decide which area each router/node is in
    border_node = 0
    area1_nodes = [border_node]
    area2_nodes = [border_node]
    for n in range(1, len(network)):
        if n%4 == 0 or n%4 == 1:
            area1_nodes.append(n)
        else:
            area2_nodes.append(n)

    # Delete connections between areas
    degree = int(len(network)/4)
    tile = [[1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1]]
    area1_pos = np.tile(tile, (degree, degree))
    area1_pos[0] = [1] * len(area1_pos[0])
    network = np.multiply(network, area1_pos)

    # Return network with connections between areas deleted
    return network

def visualise_network(network, route_start=None, route_end=None):
    # Create a NetworkX graph from the network connections matrix
    nx_graph = nx.from_scipy_sparse_array(csr_matrix(network))

    # Determine the positions for each node
    pos_arr = []
    for n in range(0, len(network)):
        angle = 3*math.pi/4 - (math.pi/2)*n
        dist = math.floor(n/4) * 2 + 2
        pos_arr.append((dist * math.cos(angle), dist * math.sin(angle)))

    # Highlight the spanning tree
    cols = ["gray"] * len(nx_graph.edges)
    if route_start != None and route_end != None:
        dists, pred = dijkstra(network, indices=route_start, directed=False, return_predecessors=True)
        p = route_end
        while p != route_start:
            q = pred[p]
            try:
                cols[list(nx_graph.edges).index((p, q))] = "red"
            except ValueError:
                cols[list(nx_graph.edges).index((q, p))] = "red"
            p = q

    # Draw the network
    nx.draw(nx_graph, pos_arr, with_labels=True, node_color='lightblue', node_size=700, font_size=16, font_color='black', font_weight='bold', edge_color=cols)

    # Show the network using Matplotlib
    plt.show()

def get_route_distance(network, route_start, route_end):
    dists = dijkstra(network, indices=route_start, directed=False)
    return dists[route_end]
    
def get_avg_route_distance(network):
    dists = dijkstra(network, directed=False)
    return np.sum(dists) / np.size(dists)

def get_avg_hop_delay(network, num_areas=1, max_age_sec=1800, lsa_size_bits=1024*8*10, port_bandwidth_bps=1024**3, forward_delay_sec=0.0):
    N = len(network)    # Number of nodes
    n = num_areas
    m = N/n
    if num_areas == 1:
        num_lsa_per_link = (m*(n**2)*((m-1)**2)) / np.sum(network)
    else:
        num_lsa_per_link = ((m+n-1)*n*(m-1)**2 + 2*n*(n-1)**2) / np.sum(network)

    # Calculate the time spent queuing
    packet_bandwith = port_bandwidth_bps / lsa_size_bits
    packets_per_sec = num_lsa_per_link / max_age_sec
    queue_time = 1 / (packet_bandwith - packets_per_sec)
    if queue_time < 0:
        queue_time = math.inf

    return queue_time + forward_delay_sec


#
#  Start of main script
#


# degrees = [x for x in range(1, int(sys.argv[1]), 10)]
# times1 = []
# times2 = []
# for d in degrees:
#     t = generate_topology(d)
#     print(d, ": ", end="", flush=True)
#     avg_ping_time = get_avg_hop_delay(t, 1) * get_avg_route_distance(t) * 1000
#     times1.append(avg_ping_time)
#     print("Full ", end="", flush=True)

#     t = split_network(t)
#     avg_ping_time = get_avg_hop_delay(t, 2) * get_avg_route_distance(t) * 1000
#     times2.append(avg_ping_time)
#     print("Split", flush=True)


# fig, ax = plt.subplots()
# ax.get_yaxis().get_major_formatter().set_useOffset(False)
# ax.plot(degrees, times1, label="One OSPF Area")
# ax.plot(degrees, times2, label="Two OSPF Areas")
# ax.set_title("Estimated Latency Between Routers vs OSPF Network Size")
# ax.set_xlabel("Number of Routers in Network")
# ax.set_ylabel("Latency (ms)")
# plt.legend(loc="upper left")
# plt.show()



t = generate_topology(4)
visualise_network(t, 10, 9)
# print("Full network route distance:", get_route_distance(t, 10, 9))
# print("Full network avg route dist:", get_avg_route_distance(t))

new_net = split_network(t)
visualise_network(new_net, 10, 9)
# print("Split network route distance:", get_route_distance(new_net, 10, 9))
# print("Split network avg route dist:", get_avg_route_distance(new_net)) 
