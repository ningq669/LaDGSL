# LaDGSL

LaDGSL is a novel SL relationship prediction model with three key modules: pairwise closed subgraph extraction, link-aware dual GNN attention, and feature fusion.  The pairwise closed subgraph extraction captures gene pairing interaction information.  The link-aware dual GNN attention differentiates edge importance using a relationship-weighted graph neural network and a link-aware graph attention layer.  The feature fusion module combines explicit omics features with latent ones for enhanced SL prediction.

# Requirements
  * Python 3.7 or higher
  * PyTorch 1.5.1 or higher
  * dgl 0.4.1 or higher 
  * GPU (default)

# Running  the Code
  * Execute ```python train.py``` to run the code.
  
# Note
```
   dgl has a strong dependency, so it is recommended to install a matching version.
```
