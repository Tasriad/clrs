# CLRS Algorithmic Reasoning Benchmark - Floyd-Warshall Algorithm Focus

## Project Overview

The CLRS Algorithmic Reasoning Benchmark is a machine learning framework for learning representations of classical algorithms. This explanation focuses specifically on the Floyd-Warshall algorithm implementation, which computes all-pairs shortest paths in a weighted graph.

## File Structure and Purpose (Floyd-Warshall Focus)

```
clrs/
├── clrs/                          # Main package directory
│   ├── __init__.py               # Package initialization
│   ├── _src/                     # Core source code
│   │   ├── __init__.py           # Source package init
│   │   ├── algorithms/           # Algorithm implementations
│   │   │   ├── __init__.py       # Algorithms package init
│   │   │   ├── graphs.py         # Graph algorithms (including Floyd-Warshall)
│   │   │   └── graphs_test.py    # Tests for graph algorithms
│   │   ├── clrs_text/            # CLRS text processing utilities
│   │   │   ├── __init__.py       # Text package init
│   │   │   ├── clrs_utils.py     # Utilities for CLRS text processing
│   │   │   ├── clrs_utils_test.py # Tests for CLRS utils
│   │   │   ├── generate_clrs_text.py # Text generation utilities
│   │   │   ├── huggingface_generators.py # HuggingFace integration
│   │   │   ├── huggingface_generators_test.py # Tests for HF generators
│   │   │   ├── colabs/           # Jupyter notebooks for analysis
│   │   │   │   ├── accuracy_data.csv # Accuracy data for analysis
│   │   │   │   └── accuracy_graphs.ipynb # Notebook for accuracy visualization
│   │   │   └── README.md         # Documentation for text processing
│   │   ├── model.py              # Core model interface and API
│   │   ├── nets.py               # Neural network architectures
│   │   ├── processors.py         # Graph neural network processors (GNN variants)
│   │   ├── processors_test.py    # Tests for processors
│   │   ├── encoders.py           # Input encoding mechanisms
│   │   ├── decoders.py           # Output decoding mechanisms
│   │   ├── decoders_test.py      # Tests for decoders
│   │   ├── samplers.py           # Data sampling and generation
│   │   ├── samplers_test.py      # Tests for samplers
│   │   ├── dataset.py            # Dataset creation and management
│   │   ├── dataset_test.py       # Tests for dataset
│   │   ├── evaluation.py         # Model evaluation metrics
│   │   ├── evaluation_test.py    # Tests for evaluation
│   │   ├── losses.py             # Loss functions for training
│   │   ├── losses_test.py        # Tests for losses
│   │   ├── probing.py            # Model probing and analysis
│   │   ├── probing_test.py       # Tests for probing
│   │   ├── baselines.py          # Baseline model implementations
│   │   ├── baselines_test.py     # Tests for baselines
│   │   └── specs.py              # Algorithm specifications and metadata
│   ├── examples/                 # Example usage and training scripts
│   │   └── run.py                # Main training example script
│   ├── models.py                 # High-level model definitions
│   ├── clrs_test.py              # Main package tests
│   └── py.typed                  # Type hints file
├── dataset/                      # Dataset storage directory
├── requirements/                 # Python dependencies
│   └── requirements.txt          # Package requirements
├── venv/                         # Virtual environment (if used)
├── setup.py                      # Package installation configuration
├── MANIFEST.in                   # Package manifest
├── LICENSE                       # License file
├── CONTRIBUTING.md               # Contribution guidelines
├── README.md                     # Main project documentation
├── changes.txt                   # Change log
└── Explanation.md                # This file
```

## Core Components Purpose (Floyd-Warshall Focus)

### Algorithm Implementation
- **graphs.py**: Contains the Floyd-Warshall algorithm implementation for computing all-pairs shortest paths in a weighted graph

### Neural Network Components
- **model.py**: Defines the core Model API interface that all models must implement
- **nets.py**: Contains neural network architectures and building blocks
- **processors.py**: Implements various Graph Neural Network processors:
  - Deep Sets
  - End-to-End Memory Networks
  - Graph Attention Networks (GAT)
  - Graph Attention Networks v2 (GATv2)
  - Message-Passing Neural Networks (MPNN)
  - Pointer Graph Networks
- **encoders.py**: Handles input encoding for graph data (adjacency matrices, edge weights)
- **decoders.py**: Handles output decoding for shortest path distance matrices

### Data Management
- **dataset.py**: Creates and manages training/evaluation/test datasets for Floyd-Warshall
- **samplers.py**: Handles data sampling and trajectory generation for graph problems
- **specs.py**: Defines Floyd-Warshall algorithm specifications, input/output formats, and metadata

### Training and Evaluation
- **losses.py**: Implements loss functions for training on shortest path predictions
- **evaluation.py**: Provides evaluation metrics for shortest path accuracy
- **baselines.py**: Contains baseline model implementations for comparison
- **probing.py**: Tools for analyzing and probing model behavior on Floyd-Warshall

### Utilities
- **clrs_text/**: Utilities for processing CLRS textbook content related to Floyd-Warshall
- **examples/run.py**: Main example script demonstrating how to train models on Floyd-Warshall

## Detailed File Analysis (Floyd-Warshall Context)

### Core Model Interface
**model.py** - Core model interface and API
- **Purpose**: Defines the abstract Model class that all neural network models must implement
- **Input**: `Feedback` objects containing graph features and expected outputs
- **Output**: Loss values and model predictions
- **Example**:
```python
class Model:
    def init(self, features, initial_seed):
        # Initialize model with graph features
        pass
    
    def feedback(self, rng_key, feedback):
        # Process feedback and return loss
        # For Floyd-Warshall: predicts distance matrix from weighted graph
        pass
```

### Neural Network Architectures
**nets.py** - Neural network architectures
- **Purpose**: Contains building blocks for constructing neural networks (MLPs, attention mechanisms, etc.)
- **Input**: Various tensor shapes depending on the network component
- **Output**: Processed tensors with learned representations
- **Example**:
```python
# MLP for processing node features
mlp = MLP([64, 32, 16])  # 3-layer MLP
node_features = mlp(input_features)  # Process node embeddings
```

### Graph Neural Network Processors
**processors.py** - Graph neural network processors (GNN variants)
- **Purpose**: Implements different GNN architectures for processing graph-structured data
- **Input**: Node features, edge features, adjacency matrix, hidden states
- **Output**: Updated node representations after message passing
- **Example**:
```python
# Graph Attention Network for Floyd-Warshall
gat = GraphAttentionNetwork(num_layers=3, num_heads=4)
# Input: weighted adjacency matrix, node features
# Output: updated node representations capturing shortest path information
updated_features = gat(node_fts, edge_fts, adj_mat, hidden)
```

### Input Encoding
**encoders.py** - Input encoding mechanisms
- **Purpose**: Converts raw graph data into neural network compatible representations
- **Input**: Raw adjacency matrices, edge weights, node features
- **Output**: Encoded features for nodes, edges, and global graph
- **Example**:
```python
# Encode weighted graph for Floyd-Warshall
node_enc = encode_nodes(adjacency_matrix)      # [batch, nodes, features]
edge_enc = encode_edges(edge_weights)          # [batch, nodes, nodes, features]
graph_enc = encode_graph(global_properties)    # [batch, features]
```

### Output Decoding
**decoders.py** - Output decoding mechanisms
- **Purpose**: Converts neural network outputs back to algorithm-specific formats
- **Input**: Neural network predictions (node/edge representations)
- **Output**: Algorithm outputs (distance matrices for Floyd-Warshall)
- **Example**:
```python
# Decode shortest path distance matrix
distance_decoder = DistanceMatrixDecoder()
predicted_distances = distance_decoder(node_predictions)  # [batch, nodes, nodes]
# Output: predicted shortest path distances between all node pairs
```

### Data Sampling and Generation
**samplers.py** - Data sampling and generation
- **Purpose**: Generates training data by executing Floyd-Warshall on random weighted graphs
- **Input**: Algorithm specifications, problem size, number of samples
- **Output**: Trajectories with input graphs, intermediate states (hints), and final outputs
- **Example**:
```python
# Generate Floyd-Warshall training data
sampler = FloydWarshallSampler(problem_size=16, num_samples=1000)
trajectories = sampler.sample()
# Output: List of (input_graph, hints, distance_matrix) tuples
```

### Dataset Management
**dataset.py** - Dataset creation and management
- **Purpose**: Creates TensorFlow datasets from generated trajectories
- **Input**: Trajectories from samplers, batch size, split information
- **Output**: TensorFlow dataset with batched training examples
- **Example**:
```python
# Create dataset for Floyd-Warshall
train_ds = create_dataset(
    trajectories=floyd_warshall_trajectories,
    batch_size=32,
    split='train'
)
# Output: TensorFlow dataset yielding Feedback objects
```

### Algorithm Specifications
**specs.py** - Algorithm specifications and metadata
- **Purpose**: Defines input/output formats, hints structure, and algorithm metadata
- **Input**: Algorithm name and configuration
- **Output**: Specification object with type information and constraints
- **Example**:
```python
# Get Floyd-Warshall specification
spec = get_algorithm_spec('floyd_warshall')
# Output: Spec with input types (adjacency_matrix), 
#         output types (distance_matrix), hint types (intermediate_distances)
```

### Loss Functions
**losses.py** - Loss functions for training
- **Purpose**: Computes loss between predicted and ground truth outputs
- **Input**: Predicted distance matrices, ground truth distance matrices
- **Output**: Loss value for backpropagation
- **Example**:
```python
# Compute loss for Floyd-Warshall predictions
loss = mean_squared_error(predicted_distances, true_distances)
# Or custom loss considering shortest path properties
loss = shortest_path_loss(predicted, true, mask)
```

### Model Evaluation
**evaluation.py** - Model evaluation metrics
- **Purpose**: Computes accuracy and other metrics for shortest path predictions
- **Input**: Predicted outputs, ground truth outputs
- **Output**: Evaluation metrics (accuracy, precision, recall, etc.)
- **Example**:
```python
# Evaluate Floyd-Warshall model
metrics = evaluate_shortest_paths(predicted_distances, true_distances)
# Output: {'accuracy': 0.95, 'mean_error': 0.02, 'exact_matches': 0.87}
```

### Model Probing
**probing.py** - Model probing and analysis
- **Purpose**: Analyzes what the model has learned about Floyd-Warshall internals
- **Input**: Trained model, test data
- **Output**: Analysis of model behavior and learned representations
- **Example**:
```python
# Probe model understanding of Floyd-Warshall
probe_results = probe_floyd_warshall_understanding(model, test_data)
# Output: Analysis of how well model captures intermediate distance updates
```

### Baseline Models
**baselines.py** - Baseline model implementations
- **Purpose**: Provides reference implementations of different model architectures
- **Input**: Model configuration and hyperparameters
- **Output**: Configured model ready for training
- **Example**:
```python
# Create baseline GNN model for Floyd-Warshall
baseline_model = BaselineModel(
    processor='gat',      # Graph Attention Network
    hidden_dim=64,
    num_layers=3
)
# Output: Ready-to-train model for shortest path prediction
```

## Full Flow When Running Example Code (Floyd-Warshall Focus)

### 1. Entry Point: `python3 -m clrs.examples.run`

The main entry point is `clrs/examples/run.py`, which demonstrates training a Graph Neural Network on the Floyd-Warshall algorithm.

### 2. Dataset Creation and Loading

```python
# Dataset is created specifically for Floyd-Warshall algorithm
train_ds, num_samples, spec = clrs.create_dataset(
    folder='/tmp/CLRS30', 
    algorithm='floyd_warshall',  # Floyd-Warshall algorithm
    split='train', 
    batch_size=32
)
```

**Flow:**
- **samplers.py**: Generates weighted graph inputs and Floyd-Warshall execution trajectories with hints
- **dataset.py**: Creates TensorFlow datasets from generated Floyd-Warshall trajectories
- **specs.py**: Provides Floyd-Warshall algorithm specifications and metadata
- Data is structured as `Feedback` named tuples with `features` (input graphs) and `outputs` (shortest path matrices)

### 3. Model Initialization

```python
# Model is initialized with first batch of graph features
model.init(feedback.features, initial_seed)
```

**Flow:**
- **model.py**: Defines the Model API interface
- **nets.py**: Constructs the neural network architecture
- **processors.py**: Sets up the GNN processor (e.g., Graph Attention Networks)
- **encoders.py**: Prepares input encoding for graph nodes, edges (weights), and global features

### 4. Training Loop

```python
for i, feedback in enumerate(train_ds.as_numpy_iterator()):
    if i == 0:
        model.init(feedback.features, initial_seed)
    loss = model.feedback(rng_key, feedback)
```

**Flow:**
- **processors.py**: GNN processes graph representations through multiple layers
- **decoders.py**: Generates predictions for shortest path distance matrices
- **losses.py**: Computes loss between predicted and ground truth shortest path distances
- **evaluation.py**: Tracks metrics during training

### 5. Floyd-Warshall Algorithm Execution Flow

1. **Input Generation**: `samplers.py` generates weighted graph inputs with specific properties
2. **Algorithm Execution**: `algorithms/graphs.py` runs Floyd-Warshall and captures intermediate states
3. **Hint Generation**: Intermediate algorithm states (distance matrices at each iteration) are converted to "hints"
4. **Graph Representation**: Input weighted graphs and hints are converted to graph format
5. **Neural Processing**: GNN processes the graph through multiple layers
6. **Output Prediction**: Model predicts final shortest path distance matrix
7. **Loss Computation**: Loss is computed against ground truth shortest path distances

### 6. Model Architecture Flow (Floyd-Warshall)

```
Weighted Graph → Encoders → GNN Processor → Decoders → Shortest Path Matrix
      ↓           ↓           ↓           ↓           ↓
  Adjacency    Node/Edge    Message     Distance    Loss
  Matrix &     Features    Passing     Matrix      Computation
  Weights
```

### 7. Evaluation and Testing

- **Train**: 1000 trajectories, problem size 16 (16-node graphs)
- **Eval**: 32 × multiplier trajectories, problem size 16  
- **Test**: 32 × multiplier trajectories, problem size 64 (64-node graphs)

The model is evaluated on out-of-distribution generalization by testing on larger graphs than training.

### 8. Floyd-Warshall Specific Features

- **Graph Representation**: Weighted graphs are represented with nodes, weighted edges, and global features
- **Hint Trajectories**: Intermediate distance matrices at each Floyd-Warshall iteration are exposed as hints
- **All-Pairs Shortest Paths**: The algorithm computes shortest paths between all pairs of nodes
- **Dynamic Programming Nature**: The algorithm's DP structure is captured in the hint trajectories
- **Out-of-Distribution Testing**: Models are tested on larger graphs than training to assess generalization

This framework enables neural networks to learn the Floyd-Warshall algorithm by observing its execution traces and learning to predict shortest path distance matrices from input weighted graphs, with the goal of generalizing to larger graphs and potentially discovering insights about shortest path computation.
