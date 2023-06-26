import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl

# DOCSTRING MISSING
def draw_graph_signal(G, x, **kwargs):
    """DOCSTRING MISSING

    Args:
        G (nx.classes.graph.Graph): _description_
        x (np.ndarray): A single graph signal as a vector of length equal to number
        of graph vertices.
        pos (_type_, optional): _description_. Defaults to None.
    """
    figsize = kwargs.get('figsize', (10,10))
    pos = kwargs.get('pos', None)
    node_size = kwargs.get('node_size', 400)
    node_color = kwargs.get('node_color', '0.15')
    edge_width = kwargs.get('edge_width', 1.5)
    node_width = kwargs.get('node_width', 1)
    node_labels = kwargs.get('node_labels', None)
    anomaly_color = kwargs.get('anomaly_color', 'C3')
    colormap = kwargs.get('colormap', 'viridis')
    suptitle = kwargs.get('suptitle', 'Graph Signal Visualization')
    norm = kwargs.get('norm', cm.colors.Normalize(vmax=np.max(x), vmin=np.min(x)))
    anomaly_labels = kwargs.get('anomaly_labels', np.zeros(x.shape, dtype=bool))

    if isinstance(pos,type(None)):
        # if len(nx.get_node_attributes(G,'pos')) ==0:
        pos = nx.kamada_kawai_layout(G)
        # else:
            # pos = 

    if not isinstance(x, np.ndarray):
        raise TypeError('x provided is not a numpy.ndarray')
    if not isinstance(G, nx.classes.graph.Graph):
        raise TypeError('G provided is not an nx.classes.graph.Graph')
    if x.size!= len(G):
        raise ValueError('The dimensions of the signal x and the number of vertices'+
                         ' do not match.')

    cmap = mpl.colormaps[colormap]
    idxs = np.arange(x.size).reshape((x.shape[0],1))
    pos_array = np.zeros((len(G),2))
    for i in range(len(G)):
        pos_array[i,:] = pos[list(G)[i]]

    fig, axe = plt.subplots(figsize=figsize)
    # Draw the skeleton graph structure
    nx.draw_networkx_nodes(G, pos=pos, node_size=node_size,node_color='none',
                           edgecolors=node_color,  linewidths=node_width, ax=axe)
    nx.draw_networkx_edges(G, pos=pos, node_size=node_size,width=edge_width, ax=axe)
    nx.draw_networkx_labels(G, pos=pos, labels=node_labels, ax=axe)
    # Paint the signals and anomalies
    scat_signal = axe.scatter(pos_array[:,0],pos_array[:,1], s=node_size, c=x, cmap='viridis')
    scat_anomaly = axe.scatter(pos_array[idxs[anomaly_labels],0], pos_array[idxs[anomaly_labels],1],
                                s=node_size*0.7, facecolors='none', edgecolors=anomaly_color,
                                lw=node_width*2, label='Anomaly')
    # Set the colorbar and name the figure
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             ax=axe, orientation='horizontal',
             label='Signal strength', pad=0.01,fraction=0.05, aspect=80, location='bottom',
            extendrect=False, extend='both')
    axe.legend()
    fig.suptitle(suptitle, fontsize=figsize[0]*2)
    fig.tight_layout()
    plt.show()
    return fig, axe



class GraphSignalAnimation:
    def __init__(self, G, X, **kwargs):
        self.G = G
        self.X = X
        self.figsize = kwargs.get('figsize', (10,10))
        self.pos = kwargs.get('pos', None)
        self.node_size = kwargs.get('node_size', 400)
        self.node_color = kwargs.get('node_color', '0.15')
        self.edge_width = kwargs.get('edge_width', 1.5)
        self.node_width = kwargs.get('node_width', 1)
        self.node_labels = kwargs.get('node_labels', None)
        self.anomaly_color = kwargs.get('anomaly_color', 'C3')
        self.colormap = kwargs.get('colormap', 'viridis')
        self.suptitle = kwargs.get('suptitle', 'Sequential Graph Signal Visualization')
        self.norm = kwargs.get('norm', cm.colors.Normalize(vmax=np.max(X), vmin=np.min(X)))
        self.interval = kwargs.get('interval', 200)
        self.anomaly_labels = kwargs.get('anomaly_labels', np.zeros(X.shape, dtype=bool))
        
        if isinstance(self.pos,type(None)):
            self.pos = nx.kamada_kawai_layout(G)

        if not isinstance(X, np.ndarray):
            raise TypeError('x provided is not a numpy.ndarray')
        if not isinstance(G, nx.classes.graph.Graph):
            raise TypeError('G provided is not an nx.classes.graph.Graph')
        if X.shape[0]!= len(G):
            raise ValueError('The dimensions of the signal x and the number of vertices'+
                            ' do not match.')

        self.cmap = mpl.colormaps[self.colormap]
        self.idxs = np.arange(len(G)).reshape((len(G),1))
        self.pos_array = np.zeros((len(G),2))
        for i in range(len(G)):
            self.pos_array[i,:] = self.pos[list(G)[i]]
        # fig = plt.figure(figsize=figsize)
        self.fig, self.axe = plt.subplots(figsize=self.figsize)
        # Draw the skeleton graph structure
        nx.draw_networkx_nodes(self.G, pos=self.pos, node_size=self.node_size,node_color='none',
                        edgecolors=self.node_color,  linewidths=self.node_width, ax=self.axe)
        nx.draw_networkx_edges(self.G, pos=self.pos, node_size=self.node_size,width=self.edge_width, ax=self.axe)
        nx.draw_networkx_labels(self.G, pos=self.pos, labels=self.node_labels, ax=self.axe)
        self.fig.suptitle(self.suptitle, fontsize=self.figsize[0]*2)
        self.fig.tight_layout()

        x = X[:,0]
        anomaly_idxs = self.idxs[self.anomaly_labels[:,0].reshape((len(G),1))]
        self.scat_signal = self.axe.scatter(self.pos_array[:,0],self.pos_array[:,1], s=self.node_size, c=x,
                                cmap=self.colormap, norm=self.norm)
        self.scat_anomaly = self.axe.scatter(self.pos_array[anomaly_idxs,0], self.pos_array[anomaly_idxs,1],
                                    s=self.node_size*0.7, facecolors='none', edgecolors=self.anomaly_color,
                                    lw=self.node_width*2, label='Anomaly')

        self.fig.colorbar(mpl.cm.ScalarMappable(norm=self.norm, cmap=self.cmap),
                ax=self.axe, orientation='horizontal',
                label='Signal strength', pad=0.01,fraction=0.05, aspect=80, location='bottom',
                extendrect=False, extend='both')
        self.fig.suptitle(self.suptitle, fontsize=self.figsize[0]*2)
        self.fig.tight_layout()

    def __call__(self, i):
        x = self.X[:,i]
        anomaly_idxs = self.idxs[self.anomaly_labels[:,i].reshape((len(self.G),1))]
        self.scat_signal.remove()
        del self.scat_signal
        self.scat_anomaly.remove()
        del self.scat_anomaly
        self.scat_signal = self.axe.scatter(self.pos_array[:,0],self.pos_array[:,1], s=self.node_size, c=x,
                                cmap=self.colormap, norm=self.norm)
        self.scat_anomaly = self.axe.scatter(self.pos_array[anomaly_idxs,0], self.pos_array[anomaly_idxs,1],
                                   s=self.node_size*0.7, facecolors='none', edgecolors=self.anomaly_color,
                                   lw=self.node_width*2, label='Anomaly')
        self.axe.legend()
        return self.scat_signal, self.scat_anomaly
