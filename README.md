# Conserved-Blocks
A deep learning pipline for unsupervised identification of shared segments of DNA across a microbial population.

## FFT Implementation
Encodings are implemented in `fft_utils.py`. Dataset directory and output directory unfortunately must currently be manually set by editing the `.py` file they appear in. Other parameters are possible to set with command line arguments unless otherwise stated. 

### Clustering
Dataset and output directory required.

Using hardcoded parameters: `python fft_cluster_density.py`

Using command line: `python fft_cluster_density.py minPts epsilon method`

Valid methods: `ML-DSP` (Purine-Pyrimidine), `ML-DSP-INT`, `ML-DSP-REAL`, `MAFFT_COMPRESSED`

### Conserved Block Extraction
Dataset and output directory required.

Using hardcoded parameters: `python fft_block_density.py`

Using command line: `python fft_block_density.py labelfile`

`labelfile` will be automatically generated by `fft_cluster_density.py` and stored in your output directory. It will not be named `labels.pickle`, so it will be necessary to either update the hardcoded parameter or use the command line argument.

### BLAST Support
Dataset and output directory required. Additionally, you must run `blastn -outfmt 0 -subject /path/to/data.fasta -query /path/to/data.fasta > blast-out.txt` for your dataset then set the `blastfile` variable in `fft_cluster_eval.py` to the full file string for the output alignment data.

Using hardcoded parameters: `python fft_cluster_eval.py`

Using command line: `python fft_cluster_eval.py labelfile`

### Nucleotide Support Plots
Output directory required. Uses generated conserved block support data from most recent `fft_block_density.py` run, stored as `block_cluster_0.pickle`, `block_cluster_1.pickle`, etc. Delete data from previous runs; e.g., if one run generates 10 clusters, then the next generates 5, the last 5 sets of block support values in the output plot will be from the previous run unless they have been deleted.

`python plot.py`

## Pretrained Embedding Model
[dna2vec](https://arxiv.org/abs/1701.06279)

## Transformer
[Big Bird](https://proceedings.neurips.cc/paper/2020/hash/c8512d142a2d849725f31a9a7a361ab9-Abstract.html)
