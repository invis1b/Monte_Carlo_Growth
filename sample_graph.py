import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
from collections import Counter
def cartesian_from_4d(lattice4id):
    basispoints=np.array([[0,0,0],[2.26*np.sqrt(3)/2,-2.26/2,1.06]])
    LatticeVectors=np.array([[2.26*np.sqrt(3),0,0],[2.26*np.sqrt(3)/2,2.26*3/2,0],[0,0,2.12]])
    return lattice4id[0]*LatticeVectors[0]+lattice4id[1]*LatticeVectors[1]+lattice4id[2]*LatticeVectors[2]+basispoints[lattice4id[3]]
class sample_subgraph:
    def __init__(self):
        G=nx.Graph()
        for i in range(0,50):
            for j in range(0,50):
                for k in range(0,50):
                    for l in range(0,2):
                        G.add_node(50*50*2*i+50*2*j+k*2+l,lattice4id=(i,j,k,l))

        edges=[]
        neighborlist=[[0,0,0,1],[0,0,-1,1],[-1,0,0,1],[-1,0,-1,1],[-1,1,0,1],[-1,1,-1,1]]
        for i in range(0,50):
            for j in range(0,50):
                for k in range(0,50):
                    for l in range(0,6):
                        i_n=i+neighborlist[l][0]
                        j_n=j+neighborlist[l][1]
                        k_n=k+neighborlist[l][2]
                        if (i_n>=0 and i_n<50 and j_n>=0 and j_n<50 and k_n>=0 and k_n<50):
                            print(f"{i_n} {j_n} {k_n}")

                            particle1_index=50*50*2*i+50*2*j+k*2
                            particle2_index=50*50*2*i_n+50*2*j_n+k_n*2+1
                            edges.append((particle1_index,particle2_index))
        G.add_edges_from(edges)
        self.G=G
        subgraph_nodes=[50*50*2*25+50*2*25+25*2]
        subgraph=G.subgraph(subgraph_nodes)
        subgraph=nx.Graph(subgraph)
        self.subgraph=subgraph
        self.vertice=1
        self.edge=0
        self.step=0
        neighbors_count=[]
       
        for node in subgraph.nodes:
            neighbors=G.neighbors(node)
            neighbors_count.extend(neighbors)
        neighbors_count=[i for i in neighbors_count if i not in list(subgraph.nodes)]
        neighbor_multiplicity=Counter(neighbors_count)
        subgraph_neighbors=set(neighbors_count)
        print(neighbor_multiplicity) 
        self.neighbor_multiplicity=neighbor_multiplicity
        self.subgraph_neighbors=subgraph_neighbors
        self.nneighbors=len(self.subgraph_neighbors)
        self.padd=0.9
        with open("sample_graph.txt","w") as f:
            f.write(f"step particle_number bond_number graph\n")
    def write_subgraph(self):
        with open("sample_graph.txt","a") as f:
           
            f.write(f"{self.step} {self.vertice} {self.edge} [")
            lattice4id_dic=nx.get_node_attributes(self.subgraph,"lattice4id")
            for node,l in lattice4id_dic.items():
                f.write(f"[{l[0]},{l[1]},{l[2]},{l[3]}],")
            f.write(f"]\n")
    def add_node(self):
        multiplicity_list=[]
        neighbor_list=[]
        for neighbor,multiplicity in self.neighbor_multiplicity.items():
            neighbor_list.append(neighbor)
            multiplicity_list.append(multiplicity)
        multiplicity_array=np.array(multiplicity_list)
        neighbor_array=np.array(neighbor_list)
        max_multiplicity=np.max(multiplicity_array)
        max_mul_indices=np.where(multiplicity_array==max_multiplicity)[0]
        n_maxmul=max_mul_indices.shape[0]
        index=random.randint(0,n_maxmul-1)
        selected_node=set([neighbor_array[index]])
        origin_nodes=set(self.subgraph.nodes)
        origin_nodes.update(selected_node)
        subgraph=self.G.subgraph(origin_nodes)
        subgraph=nx.Graph(subgraph)
        self.subgraph=subgraph
        self.vertice=self.subgraph.number_of_nodes()
        self.edge=self.subgraph.number_of_edges()
        neighbors_count=[]
        for node in self.subgraph.nodes:
            neighbors=self.G.neighbors(node)
            neighbors_count.extend(neighbors)
        neighbors_count=[i for i in neighbors_count if i not in list(self.subgraph.nodes)]

        self.neighbor_multiplicity=Counter(neighbors_count)
        self.subgraph_neighbors=set(neighbors_count)
        self.nneighbors=len(self.subgraph_neighbors)
    def delete_node(self):
        if self.vertice>1:
            node_degrees=dict(self.subgraph.degree())
            min_degree=min(node_degrees.values())
            nodes_with_lowest_degree=[node for node, degree in node_degrees.items() if degree==min_degree]
            nlowd=len(nodes_with_lowest_degree)
            index=random.randint(0,nlowd-1)
            selected_node=nodes_with_lowest_degree[index]
            subgraph_copy=self.subgraph.copy()
            subgraph_copy.remove_node(selected_node)
            #test if the new subgraph is connected
            if nx.is_connected(subgraph_copy):
                self.subgraph.remove_node(selected_node)
                self.vertice=self.subgraph.number_of_nodes()
                self.edge=self.subgraph.number_of_edges()
                neighbors_count=[]
                for node in self.subgraph.nodes:
                    neighbors=self.G.neighbors(node)
                    neighbors_count.extend(neighbors)
                sneighbors_count=[i for i in neighbors_count if i not in list(self.subgraph.nodes)]

                self.neighbor_multiplicity=Counter(neighbors_count)
                self.subgraph_neighbors=set(neighbors_count)
                self.nneighbors=len(self.subgraph_neighbors)
    def sample(self):
        rand=np.random.random_sample()
        if (rand<self.padd):
            self.add_node()
        else:
            self.delete_node()
        self.write_subgraph()
        self.step+=1
sample=sample_subgraph()
for i in range(0,150):
    sample.sample()


        
        

