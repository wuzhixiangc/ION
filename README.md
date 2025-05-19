# ION (Integrated Object Network) Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Core Concepts](#core-concepts)
4. [Basic Usage](#basic-usage)
5. [Advanced Features](#advanced-features)
6. [Performance Optimization](#performance-optimization)
7. [API Reference](#api-reference)
8. [Usage Examples](#usage-examples)
9. [Best Practices](#best-practices)

## Introduction

ION (Integrated Object Network) is a powerful data structure designed for efficient in-memory graph-based data management. It combines features from both OND and ICombinedDataStructure to provide a versatile platform for complex data operations.

### Key Features

- **Fast retrieval**: Optimized hash-based key-value storage
- **Relationship management**: Create and manage complex relationships between data nodes
- **Metadata indexing**: Index and query nodes by metadata attributes
- **Transaction support**: ACID-compliant transaction processing
- **Advanced path finding**: Efficient algorithms for finding paths between nodes
- **Concurrency control**: Thread-safe operations with fine-grained locking
- **Large dataset support**: Partitioning strategies and memory optimization
- **Event system**: Callback mechanisms for various operations
- **Batch processing**: Efficient handling of bulk operations

## Installation

### Importing ION

```python
from ion import ION

# Create a basic ION instance
ion = ION()

# Create an ION instance with custom settings for large datasets
ion_large = ION(
    size=100000,
    max_workers=8,
    load_factor_threshold=0.6,
    large_dataset_mode=True,
    partition_config={"strategy": "field", "field": "category"}
)
```

## Core Concepts

### Nodes

Nodes are the basic data elements in ION. Each node has:
- A unique key
- A value
- Optional metadata (key-value pairs)
- Optional tags
- A weight value

Nodes can be connected to other nodes through relationships.

### Relationships

Relationships define connections between nodes. They can have:
- A relationship type
- A weight
- Optional metadata

Relationships are directional (from source to target) and can be used to model various types of connections.

### Indexes

ION maintains several types of indexes to speed up queries:
- **Key index**: Fast lookup by node key
- **Value index**: Lookup by node value
- **Metadata index**: Find nodes by metadata attributes
- **Tag index**: Find nodes by tags
- **Value type index**: Find nodes by value type
- **Relationship type index**: Find nodes with specific relationship types
- **Compound index**: Combine multiple criteria for complex queries

### Partitions

For large datasets, ION can distribute nodes across multiple partitions. Partitioning strategies include:
- **Hash partitioning**: Based on key hash
- **Field partitioning**: Based on a specific metadata field
- **Range partitioning**: Based on numeric ranges

## Basic Usage

### Creating Nodes

```python
# Create a simple node
node1 = ion.create_node(key="user:1001", val="John Doe")

# Create a node with metadata and tags
node2 = ion.create_node(
    key="user:1002",
    val="Jane Smith",
    metadata={"age": 28, "status": "active"},
    tags=["premium", "verified"]
)
```

### Retrieving Nodes

```python
# Get node by key
node = ion.get_node_by_key("user:1001")

# Get node by value
node = ion.get_node_by_value("John Doe")

# Find nodes by metadata
active_users = ion.find_by_metadata("status", "active")

# Find nodes by tag
premium_users = ion.find_by_tag("premium")
```

### Managing Relationships

```python
# Add a relationship
ion.add_relationship(
    source="user:1001",
    target="user:1002",
    rel_type="follows",
    rel_weight=1.0,
    metadata={"since": "2023-05-15"}
)

# Remove a relationship
ion.remove_relationship("user:1001", "user:1002", rel_type="follows")

# Find related nodes
followers = ion.find_related("user:1002", rel_type="follows")
```

### Path Finding

```python
# Find a path between two nodes
path = ion.search_path("user:1001", "user:1005")

# Advanced path finding with A* algorithm
path, cost = ion.astar_search(
    start="user:1001",
    goal="user:1005",
    heuristic_func=lambda n1, n2: 1.0 if "premium" in n1.tags else 2.0
)
```

## Advanced Features

### Transaction Management

```python
# Using a transaction with context manager
with ion.begin_transaction() as txn:
    node1 = ion.create_node_in_transaction(txn.id, "user:1003", "Alice Brown")
    node2 = ion.create_node_in_transaction(txn.id, "user:1004", "Bob Green")
    ion.add_relationship_in_transaction(txn.id, node1, node2, "friends")
    # Transaction automatically commits if no exceptions

# Manual transaction management
txn = ion.begin_transaction()
try:
    # Perform operations
    ion.create_node_in_transaction(txn.id, "user:1005", "Charlie White")
    ion.commit_transaction(txn.id)
except Exception as e:
    ion.abort_transaction(txn.id, reason=str(e))
```

### Concurrent Operations

```python
# Batch create nodes
node_data = [
    {"key": "product:1", "val": "Laptop", "metadata": {"price": 999.99}},
    {"key": "product:2", "val": "Smartphone", "metadata": {"price": 499.99}},
    {"key": "product:3", "val": "Tablet", "metadata": {"price": 349.99}}
]
nodes = ion.batch_create_nodes(node_data)

# Batch add relationships
relationships = [
    {"source": "user:1001", "target": "product:1", "type": "purchased"},
    {"source": "user:1002", "target": "product:2", "type": "viewed"}
]
ion.batch_add_relationships(relationships)
```

### Compound Indexing

```python
# Create a compound index for faster multi-criteria queries
ion.create_compound_index(('tag', 'metadata'))

# Query using compound index
results = ion.compound_search(tag="premium", metadata=("status", "active"))
```

### Event Callbacks

```python
# Register event callbacks
def on_node_added(node):
    print(f"Node added: {node}")

def on_relation_added(source, target, rel_type):
    print(f"Relation added: {source.key} --[{rel_type}]--> {target.key}")

ion.register_callback('node_added', on_node_added)
ion.register_callback('relation_added', on_relation_added)
```

## Performance Optimization

### Memory Management

```python
# Configure memory limits
ion.configure_cache(
    memory_limit_mb=1024,  # 1GB memory limit
    cache_strategy="lfu",  # Least Frequently Used cache strategy
    node_cache_size=5000,
    query_cache_size=1000
)

# Run garbage collection manually
ion._run_garbage_collection(force=True)
```

### Query Optimization

```python
# Optimize for path finding
ion.optimize_for_path_finding(max_cached_paths=2000)

# Use cached path finding
path, cost = ion.cached_astar_search(
    start="user:1001",
    goal="user:1005"
)

# Get path finding statistics
stats = ion.get_path_finding_stats()
print(f"Cache hit ratio: {stats['hit_ratio']:.2f}")
```

### Large Dataset Handling

```python
# Create partitions for large datasets
ion.create_partition("active_users")
ion.create_partition("inactive_users")

# Assign nodes to partitions
active_users = ion.find_by_metadata("status", "active")
for user in active_users:
    ion.assign_node_to_partition(user, "active_users")

# Query within a specific partition
results = ion.find_nodes_by_filter(
    lambda node: "premium" in node.tags,
    partitions=["active_users"]
)

# Optimize data distribution
ion.optimize_data_distribution()
```

## API Reference

### Constructors

```python
# Basic constructor
ION(
    start=None,              # Optional starting node
    size=1024,               # Initial bucket size
    max_workers=4,           # Number of worker threads
    load_factor_threshold=0.75,  # When to resize
    enable_memory_optimization=False,
    large_dataset_mode=False,    # Enable optimizations for large datasets
    partition_config=None,       # Partition strategy configuration
    memory_limit_mb=None         # Memory usage limit
)
```

### Core Methods

#### Node Operations

- `create_node(key, val=None, metadata=None, tags=None, weight=1.0)` - Create a node
- `get_node_by_key(key)` - Get node by key
- `get_node_by_value(val)` - Get node by value
- `remove_node_by_key(key)` - Delete node by key
- `update_node_metadata(node, new_metadata)` - Update node metadata
- `update_node_tag(node, operation='add', tags=None, clear_existing=False)` - Update node tags
- `update_node_weight(node, weight)` - Update node weight

#### Relationship Operations

- `add_relationship(source, target, rel_type=None, rel_weight=1.0, metadata=None)` - Add relationship
- `remove_relationship(source, target, rel_type=None)` - Remove relationship
- `find_related(node_input, rel_type=None, max_depth=2)` - Find related nodes
- `find_common_related(node1, node2, rel_type=None, max_depth=2)` - Find common related nodes

#### Query Operations

- `find_by_metadata(meta_key, meta_val)` - Find nodes by metadata
- `find_by_tag(tag)` - Find nodes by tag
- `find_by_value_type(type_name)` - Find nodes by value type
- `advanced_search(criteria=None, metadata_filters=None, tag_filters=None, value_type=None, rel_type=None, max_results=None)` - Advanced search
- `fuzzy_search(query, search_in=None, max_results=50, similarity_threshold=0.6, case_sensitive=False)` - Fuzzy search
- `compound_search(**conditions)` - Compound index search

#### Transaction Management

- `begin_transaction(isolation_level=None, timeout=None)` - Begin transaction
- `commit_transaction(transaction_id)` - Commit transaction
- `abort_transaction(transaction_id, reason=None)` - Abort transaction
- `get_transaction(transaction_id)` - Get transaction by ID
- `get_active_transactions()` - Get all active transactions

#### Path Finding

- `search_path(start, end, max_depth=10, rel_type=None)` - Find path between nodes
- `astar_search(start, goal, heuristic_func=None, weight_func=None, rel_type=None, max_iterations=10000)` - A* path finding
- `astar_search_advanced(start, goal, options=None)` - Advanced A* path finding

## Usage Examples

### Social Network Analysis

```python
# Create user nodes
alice = ion.create_node("user:1", "Alice", metadata={"age": 28})
bob = ion.create_node("user:2", "Bob", metadata={"age": 32})
charlie = ion.create_node("user:3", "Charlie", metadata={"age": 24})

# Add friendship relationships
ion.add_relationship(alice, bob, "friend")
ion.add_relationship(bob, charlie, "friend")

# Check if there's a path between Alice and Charlie
path = ion.search_path(alice, charlie)
if path:
    print(f"Path found: {' -> '.join([node.val for node in path])}")
    
# Find users who are friends of Bob
bobs_friends = ion.find_related(bob, rel_type="friend")
print(f"Bob's friends: {[friend.val for friend in bobs_friends]}")

# Find users above age 25
adults = ion.find_nodes_by_filter(
    lambda node: node.metadata.get("age", 0) > 25
)
print(f"Users above 25: {[user.val for user in adults]}")
```

### Product Recommendation System

```python
# Create product nodes
laptop = ion.create_node(
    "product:1", 
    "Laptop",
    metadata={"category": "electronics", "price": 999.99},
    tags=["electronics", "computer"]
)

smartphone = ion.create_node(
    "product:2", 
    "Smartphone",
    metadata={"category": "electronics", "price": 499.99},
    tags=["electronics", "mobile"]
)

headphones = ion.create_node(
    "product:3", 
    "Headphones",
    metadata={"category": "accessories", "price": 99.99},
    tags=["electronics", "audio"]
)

# Create customer nodes
customer1 = ion.create_node("customer:1", "John")
customer2 = ion.create_node("customer:2", "Sarah")

# Add purchase relationships
ion.add_relationship(customer1, laptop, "purchased", metadata={"date": "2023-01-15"})
ion.add_relationship(customer1, headphones, "purchased", metadata={"date": "2023-02-20"})
ion.add_relationship(customer2, smartphone, "purchased", metadata={"date": "2023-03-10"})

# Find products purchased by a customer
john_purchases = ion.find_related(customer1, rel_type="purchased")
print(f"John purchased: {[product.val for product in john_purchases]}")

# Find customers who purchased electronics
electronics = ion.find_by_tag("electronics")
customers = set()
for product in electronics:
    # Find customers with 'purchased' relationship to this product
    for bucket in ion.buckets:
        for node in bucket:
            for relation in node.r:
                if relation['node'] == product and relation['type'] == "purchased":
                    customers.add(node)
                    
print(f"Customers who purchased electronics: {[customer.val for customer in customers]}")

# Recommend products based on common purchases
def get_recommendations(customer, max_recommendations=3):
    # Get what the customer has already purchased
    purchased = ion.find_related(customer, rel_type="purchased")
    purchased_set = set(purchased)
    
    # Find similar customers (who purchased at least one same product)
    similar_customers = set()
    for product in purchased:
        # Find customers who purchased this product
        for bucket in ion.buckets:
            for node in bucket:
                if node != customer:  # Exclude the customer himself
                    for relation in node.r:
                        if relation['node'] == product and relation['type'] == "purchased":
                            similar_customers.add(node)
    
    # Find products purchased by similar customers but not by this customer
    recommendations = []
    for sim_customer in similar_customers:
        sim_purchases = ion.find_related(sim_customer, rel_type="purchased")
        for product in sim_purchases:
            if product not in purchased_set:
                recommendations.append(product)
    
    # Return top recommendations
    return recommendations[:max_recommendations]

# Get recommendations for John
recommendations = get_recommendations(customer1)
print(f"Recommended for John: {[product.val for product in recommendations]}")
```

## Best Practices

### Memory Optimization

1. **Use appropriate initial size**: Set the initial size based on expected data volume.
2. **Enable memory optimization**: For large datasets, use `enable_memory_optimization=True`.
3. **Configure cache limits**: Set appropriate cache sizes based on available memory.
4. **Use partitioning**: Distribute data across partitions for better memory management.
5. **Monitor memory usage**: Periodically check memory usage and adjust parameters if needed.

### Performance Tuning

1. **Create compound indexes**: For frequently combined query conditions.
2. **Use batch operations**: Prefer batch methods for bulk operations.
3. **Optimize path finding**: Use `optimize_for_path_finding()` for frequent path queries.
4. **Select appropriate concurrency mode**: Use 'optimistic' mode for read-heavy workloads and 'pessimistic' for write-heavy workloads.
5. **Configure worker threads**: Set `max_workers` based on available CPU cores and workload characteristics.

### Scaling Guidelines

1. **Enable large dataset mode**: Set `large_dataset_mode=True` for datasets with more than 100,000 nodes.
2. **Choose effective partitioning strategy**: Based on access patterns.
3. **Consider sharding**: For extremely large datasets, consider sharding across multiple ION instances.
4. **Implement periodic optimization**: Call `optimize()` or `optimize_data_distribution()` periodically.
5. **Monitor performance metrics**: Track operations that take too long and optimize accordingly. 
