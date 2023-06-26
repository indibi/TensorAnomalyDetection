import networkx as nx

def generate_connected_graph(size, type, seed=0, maxit=200, **kwargs):
    """Generate a random connected graph 

    Args:
        size (int): number of vertices
        type (int): Type of the random graph
        param (_type_): _description_
        seed (int): random generation 
        maxit (int, optional): _description_. Defaults to 200.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    it = 0
    while True:
        sd = seed+it
        if type == 'geometric':
            G = nx.random_geometric_graph(size, kwargs.get('radius',0.25), seed=sd)
        elif type == 'grid':
            if not isinstance(size,tuple):
                raise TypeError("Specified size must be a tuple for the grid option.")
            G = nx.grid_2d_graph(size[0],size[1])
        elif type == 'er':
            G = nx.erdos_renyi_graph(size, kwargs.get('p', 0.2), seed=sd)
        elif type == 'ba':
            G = nx.barabasi_albert_graph(size, kwargs.get('m', 1), seed=sd)
        if nx.is_connected(G):
            print("Graph is connected.")
            break
        it +=1
    if it == maxit:
        raise ValueError("Couldn't construct a connected graph")
    return G, sd

