import networkx as nx

G = nx.DiGraph()
G.add_node("S")
G.add_node("T")

# 2 starved bases, needing 10 each
C = 10
G.add_edge("S", "V1", capacity=C)
G.add_edge("S", "V2", capacity=C)

# 1 demand node with capacity 5
G.add_edge("D1", "T", capacity=5)

G.add_edge("V1", "D1", capacity=float('inf'))
G.add_edge("V2", "D1", capacity=float('inf'))

flow, partition = nx.minimum_cut(G, "S", "T")
print("Flow:", flow)
starved = [n for n in partition[0] if n.startswith("V")]
print("Starved:", starved)
