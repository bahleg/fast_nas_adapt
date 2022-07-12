# Full DARTS
1. Run model_construction/search.py (check dataset used: fmnist2)
2. Obtain structure
3. Run model_construction/retrain.py

# DARTS 8 to 2
1. Run model_construction/search.py (check dataset used: fmnist3)
2. Obtain structure
3. Run "search_sigma.py" (important args: arc-checkpoint, pt, head, gs)
4. Obtain structure
5. Run model_construction/retrain.py

# Random search
1. Run model_construction/search.py
2. Obtain structure
3. Run "2_random_search" to obtain pruned structures
4. Run "retrain"

# Molchanov
1. Run model_construction/search.py
2. Obtain structure
3. Run "4_prepare_molchanov_pruning.ipynb"
4. Run "retrain"