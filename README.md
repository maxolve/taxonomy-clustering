# Taxonomy-Based Clustering for Archetype Identification

This project implements various clustering algorithms aimed at improving taxonomy-based archetype identification in Information Systems (IS) research. The goal is to address mismatches between the chosen clustering methods and the data types commonly used in taxonomy-building studies.

**Note:** This is a work in progress.

## Requirements

To run the project, you'll need the following Python packages:

- `pandas`
- `numpy`
- `scikit-learn`
- `sklearn-extra`
- `pyclustering`
- `kmodes` (https://github.com/nicodv/kmodes)
- `networkx`
- `stepmix` (https://github.com/Labo-Lacourse/stepmix)
- `LSHkCenters` (https://github.com/jgutierrezre/lshkcenters-1.0.3)
- `kPbC` (https://github.com/ClarkDinh/k-PbC)
- `fpmax` (for kPbC algorithm)
- `kSCC` (https://github.com/ClarkDinh/k-SCC)
- `SoftModes` (https://github.com/sharath-cgm/SoftModes)
- `cdclustering` (custom package)

Install the required packages using:

```bash
pip install pandas numpy scikit-learn sklearn-extra pyclustering kmodes networkx stepmix
```

Some custom packages used for specific algorithms (e.g., cdclustering) have to be included included in the local_packages directory.

## Usage

To run the clustering evaluation and analysis, execute the main script:

```bash
python main.py
```

This will load the datasets, preprocess the data, and evaluate multiple clustering algorithms across different metrics and cluster sizes. The results will be saved as CSV files in the output directory.

