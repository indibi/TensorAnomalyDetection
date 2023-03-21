# PyGraphLearning - a python package for graph signal processing based graph learning
# Copyright (C) 2021 Abdullah Karaaslanli <evdilak@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import networkx as nx

def topological_undirected(G, n_swap=1):
    """Perform edge swapping while preserving degree distribution. The method randomly select two 
    edges: (a, b, w_ab) and (c, d, w_cd) where w_ab and w_cd are edge weights. Then it removes these 
    edges and add (a, d, w_ab) and (c, b, w_cd). Edge swapping is done in place. There is a edge 
    swapping function in networkx, but it does not handle edge weights. 

    Parameters
    ----------
    G : networkx graph
        An undirected binary/weighted graph
    n_swap : int, optional
        Number of edge swap to perform, by default 1
    """

    n = G.number_of_nodes()
    m = G.number_of_edges()

    is_weighted = nx.is_weighted(G)

    # If the graph is fully connected, there is nothing to randomize
    if m == (n*(n-1)/2):
        return
    
    # Get the nodes of the graph as a list
    node_labels = [i for i in G.nodes]

    rng = np.random.default_rng()

    max_attempt = round(n*m/(n*(n-1)))
    
    for _ in range(n_swap):
        attempt = 0
        while attempt <= max_attempt:
            
            # Select edges to swap
            # First edge
            a = rng.choice(node_labels)
            if G.degree(a) == 0:
                continue
            b = rng.choice(list(G.neighbors(a)))

            # Second edge
            while True:
                c = rng.choice(node_labels)
                
                if G.degree(c) == 0:
                    continue
                
                d = rng.choice(list(G.neighbors(c)))

                if c != b and c != a and d!=a and d != b:
                    break
            
            # Rewire condition
            if (d not in G[a]) and (b not in G[c]):
                if is_weighted:
                    w_ab = G[a][b]['weight']
                    w_cd = G[c][d]['weight']
                    G.remove_edge(a, b)
                    G.remove_edge(c, d)
                    G.add_edge(a, d, weight=w_ab)
                    G.add_edge(c, b, weight=w_cd)
                else:
                    G.remove_edge(a, b)
                    G.remove_edge(c, d)
                    G.add_edge(a, d)
                    G.add_edge(c, b)
                break
            attempt += 1