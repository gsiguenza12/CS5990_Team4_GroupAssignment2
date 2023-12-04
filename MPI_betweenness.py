import csv
import sys
import pickle
import networkx as nx
from mpi4py import MPI
import time
from datetime import datetime


def progress_bar(percent, complete=False):
    bar_length = 50
    block = int(round(bar_length * percent))
    progress = "=" * block + "-" * (bar_length - block)
    sys.stdout.write(f"\r[{progress}] {int(percent * 100)}%")

    if complete:
        sys.stdout.write("\n")

    sys.stdout.flush()


def main():
    # Create a graph object
    filename = 'facebook_combined.txt'
    G = nx.read_edgelist(filename, delimiter=" ")
    # G = nx.Graph()
    # data = {
    #     0: [(1, 1), (2, 4)],
    #     1: [(0, 1), (2, 2), (3, 5)],
    #     2: [(0, 4), (1, 2), (3, 1)],
    #     3: [(1, 5), (2, 1)]
    # }
    #
    # for node, edges in data.items():
    #     for edge in edges:
    #         G.add_edge(node, edge[0], weight=edge[1])

    start_time = time.time()  # Record the start time
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"Current time in 24-hour format: {current_time}")
    count = 0
    # Initialize MPI communicator, rank, and size
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Divide the graph nodes into chunks
    nodes = list(G.nodes())
    chunk_size = len(nodes) // size
    start = rank * chunk_size
    end = (rank + 1) * chunk_size if rank < size - 1 else len(nodes)
    chunk = nodes[start:end]

    # Compute the betweenness centrality for the chunk of nodes
    # bt = nx.betweenness_centrality_subset(G, chunk, nodes) # nx version for comparison
    # bt = nx.betweenness_centrality(G)

    bt = {}
    for node in chunk:
        # Update the loading bar
        percent_complete = (count + 1) / len(nodes)
        progress_bar(percent_complete)
        # Initialize the number of shortest paths and the dependency scores for each node
        sigma = {n: 0 for n in nodes}
        sigma[node] = 1
        delta = {n: 0 for n in nodes}
        # Use a queue to perform a breadth-first search from the node
        queue = [node]
        # Use a stack to store the nodes in the order of non-increasing distance from the node
        stack = []
        # Use a dictionary to store the predecessors of each node on the shortest paths from the node
        pred = {n: [] for n in nodes}
        # Use a dictionary to store the distance of each node from the node
        dist = {n: -1 for n in nodes}
        dist[node] = 0
        while queue:
            # Dequeue the first node in the queue
            v = queue.pop(0)
            # Push it to the stack
            stack.append(v)
            # For each neighbor of v
            for w in G.neighbors(v):
                # If w has not been visited, set its distance and enqueue it
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                # If w is at the same distance as v, it is a predecessor of v and increase its sigma value
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        # Pop the nodes from the stack and update their dependency scores
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != node:
                bt[w] = bt.get(w, 0) + delta[w]
        count += 1

    # Gather the partial results from each process
    bt_list = comm.gather(bt, root=0)

    # Combine the partial results into a final dictionary
    output_file = 'bt_notPickle.txt'
    with open(output_file, 'w') as file:
        if rank == 0:
            bt_final = {}
            for bt in bt_list:
                for node in bt:
                    bt_final[node] = bt[node]
                    file.write(f"Node {node}: Betweenness Centrality = {bt_final[node]}\n")
            # Finish the loading bar
            progress_bar(1, complete=True)

            # Print final elapsed time
            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            seconds = int(elapsed_time % 60)
            print(f"Elapsed time: {hours:02}:{minutes:02}:{seconds:02}")

            # print(bt_final)

            # Write the bt_final to a file named 'networkx_bt.txt' using pickle
            # with open('networkx_bt.txt', 'wb') as f:
            #     pickle.dump(bt_final, f)

            # Write the bt_final to a file named 'bt.txt' using pickle
            # with open('bt_notPickle.txt', 'wb') as f:
            #     f.write(f"Betweenness Centrality = {bt_final}\n")
# to view pickle file use this webapp: https://fire-6dcaa-273213.web.app/

if __name__ == "__main__":
    main()
