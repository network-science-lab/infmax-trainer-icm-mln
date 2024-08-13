# multi-node2vec

This is Python source code for the multi-node2vec algorithm. Multi-node2vec is a fast network embedding
method for multilayer networks that identifies a continuous and low-dimensional representation for the
unique nodes in the network. 

Details of the algorithm can be found in the paper: 
*Fast Embedding of Multilayer Networks: An Algorithm and Application to Group fMRI* by JD Wilson,
M Baybay, R Sankar, and P Stillman. 

**Preprint**: https://arxiv.org/pdf/1809.06437.pdf

__Contributors__:
- Melanie Baybay
University of San Francisco, Department of Computer Science
- Rishi Sankar
Henry M. Gunn High School
- James D. Wilson (maintainer)
University of San Francisco, Department of Mathematics and Statistics

**Questions or Bugs?** Contact James D. Wilson at jdwilson4@usfca.edu

## The Mathematical Objective

A multilayer network of length *m* is a collection of networks or graphs {G<sub>1</sub>, ..., G<sub>m</sub>},
where the graph G<sub>j</sub> models the relational structure of the *j*th layer of the network. Each
layer G<sub>j</sub> = (V<sub>j</sub>, W<sub>j</sub>) is described by the vertex set V<sub>j</sub>
that describes the units, or actors, of the layer, and the edge weights W<sub>j</sub> that describes
the strength of relationship between the nodes. Layers in the multilayer sequence may be heterogeneous
across vertices, edges, and size. Denote the set of unique nodes in {G<sub>1</sub>, ..., G<sub>m</sub>} 
by **N**, and let *N* = |**N**| denote the number of nodes in that set. 

The aim of the **multi-node2vec** is to learn an interpretable low-dimensional feature representation
of **N**. In particular, it seeks a *D*-dimensional representation

**F**: **N** --> R<sup>*D*</sup>, 

where *D* < < N. The function **F** can be viewed as an *N* x *D* matrix whose rows
{**f**<sub>v</sub>: v = 1, ..., N} represent the feature space of each node in **N**. 

## The Algorithm

The **multi-node2vec** algorithm estimates **F** through maximum likelihood estimation, and relies upon
two core steps

1) __NeighborhoodSearch__: a collection of vertex neighborhoods from the observed multilayer graph, also
    known as a *BagofNodes*, is identified. This is done through a 2nd order random walk on the multilayer
    network.

2) __Optimization__: Given a *BagofNodes*, **F** is then estimated through the maximization of the
    log-likelihood of **F** | **N**. This is done through the application of stochastic gradient
    descent on a two-layer Skip-gram neural network model.

## Running multi-node2vec

```bash
docker build -t multi-node2vec --platform=linux/amd64 .
docker run -itd -v .:/app --platform linux/amd64 --name multi-node2vec multi-node2vec 
```

### Example

This example runs **multi-node2vec** on a small test multilayer network with 2 layers and 264 nodes
in each layer. It takes about 2 minutes to run on a personal computer using 8 cores.

```bash
python multi_node2vec.py --dir data/test --output results/test --d 100 --window_size 10 --n_samples 1 --rvals 0.25 --pvals 1 --thresh 0.5 --qvals 0.5
```
