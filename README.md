
# LaDGSL

LaDGSL is a novel SL relationship prediction model with three key modules: pairwise closed subgraph extraction, link-aware dual GNN attention mechanism, and feature fusion.  The pairwise closed subgraph extraction captures gene pairing interaction information.  The link-aware dual GNN attention differentiates edge importance using a relationship-weighted graph neural network and a link-aware graph attention layer.  The feature fusion module combines explicit omics features with latent ones for enhanced SL prediction.

# Requirements
  * Python 3.7 or higher
  * PyTorch 1.5.1 or higher
  * dgl 0.4.1 or higher 
  * GPU (default)

# Data Description

### Synthetic Lethality (SL) Data
- **Location**: `./sl_use_data/`
- **Content**: 
  - Gene IDs and corresponding names
  - Known SL gene pairs
- **Format**: CSV/TXT files containing SL interactions and other information.

### Knowledge Graph (KG) Data
- **Location**: `./kg_raw_data/`
- **Content**:
  - Entity information: ID, type, name
  - Relationship information: ID, type, name  
  - Knowledge graph triples (head, relation, tail)
- **Format**: CSV/TXT files containing the above-mentioned content.

### Embeddings
- Pre-trained entity and relation embeddings generated using TransE model.
- Based on the knowledge graph triples.

# Running  the Code
1. **Create Virtual Environment**
   ```
   conda create -n LaDGSL python=3.7
   conda activate LaDGSL
   ```
2. **Install Dependencies**
   ```
   pip install -r requirements.txt
   ```
3. **Run Training**
   ```
   python train.py
   ```
  
# Note
```
   dgl has a strong dependency, so it is recommended to install a matching version.
```
