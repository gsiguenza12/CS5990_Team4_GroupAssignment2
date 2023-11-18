from mpi4py import MPI
import networkx as nx
import pickle
import csv
import sys


def progress_bar(percent, complete=False):
    bar_length = 50
    block = int(round(bar_length * percent))
    progress = "=" * block + "-" * (bar_length - block)
    sys.stdout.write(f"\r[{progress}] {int(percent * 100)}%")

    if complete:
        sys.stdout.write("\n")

    sys.stdout.flush()


def distribute_nodes(nodes, rank, size):
    # Split the dictionary keys
    keys = list(nodes)
    total_nodes = len(nodes)
    keys_per_process = total_nodes // size
    remainder = total_nodes % size

    # Calculate the start and end indices for each process
    if rank < remainder:
        # Give one extra node to the first 'remainder' processes
        start = rank * (keys_per_process + 1)
        end = start + keys_per_process + 1
    else:
        # The rest of the processes get keys_per_process nodes
        start = remainder * (keys_per_process + 1) + (rank - remainder) * keys_per_process
        end = start + keys_per_process

    return keys[start:end]


def closeness_centrality(dist, graph):
    centrality = {}

    for node, d in dist.items():
        total_distance = sum(d.values())
        centrality[node] = (len(graph) - 1) / total_distance

    return centrality


def process_data(graph, nodes, rank):
    dist = {}
    centrality = {}
    count = 0
    output_file = 'concatenated_result.txt'
    with open(output_file, 'w') as file:
        for node in nodes:
            # Update the loading bar
            percent_complete = (count + 1) / len(nodes)
            progress_bar(percent_complete)

            dist[node], paths = nx.single_source_dijkstra(graph, node)
            centrality = closeness_centrality(dist, graph)
            file.write(f"Node {node}: Closeness Centrality = {centrality[node]:}\n")  # Write each node's centrality


    # Finish the loading bar
    progress_bar(1, complete=True)
    return centrality


def load_data(filename, print=False):
    with open(filename) as fin, open('fixed' + filename, 'w') as fout:
        for line in fin:
            fout.write(line.replace(' ', ','))
    fin.close()
    fout.close()
    filename = 'fixed' + filename

    with open(filename, 'r') as nodecsv:
        nodereader = csv.reader(nodecsv)
        nodes = [n for n in nodereader][1:]
    node_names = [n[0] for n in nodes]

    with open(filename, 'r') as edgecsv:
        edgereader = csv.reader(edgecsv)
        edges = [tuple(e) for e in edgereader][1:]

    return node_names, edges


def make_graph(node_names, edges):
    G = nx.Graph()
    G.add_nodes_from(node_names)
    G.add_edges_from(edges)
    return G


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # load data, make graph
        filename = 'facebook_combined.txt'
        
        # filename = 'facebook_combined_chunk.txt'
        node_names, edges = load_data(filename)
        graph = make_graph(node_names, edges)

        # Serialize the graph
        serialized_graph = pickle.dumps(graph)
    else:
        serialized_graph = None

    # Broadcasting the serialized graph
    serialized_graph = comm.bcast(serialized_graph, root=0)

    # Deserialize the graph on all processes
    graph = pickle.loads(serialized_graph)

    # Distribute nodes among processes
    assigned_nodes = distribute_nodes(graph.nodes(), rank, size)

    # Each process processes its assigned nodes
    partial_sum = process_data(graph, assigned_nodes, rank)

    # Gather the partial results from all processes
    all_res = comm.gather(partial_sum, root=0)

    if rank == 0:
        concatenated_result = {}
        for d in all_res:
            concatenated_result.update(d)
        # print("Concatenated Result:", concatenated_result)
        print('Run complete.')


if __name__ == "__main__":
    main()
