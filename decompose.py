# Decompose compressed .fasta files into individual contigs

import bz2
import argparse
from tqdm import tqdm
import os
import glob

# Convert scaffolds with gaps labeled as 'N' to list of contigs
def scaffold_to_contigs(seq, min=100, max=4096):
    contigs = seq.split('N')
    # Break configs over max into sub-contigs, with redundancies
    in_range = []
    for contig in contigs:
        if len(contig) > max:
            in_range += [contig[i:i+max] for i in range(0, len(contig), max//2)]
        else:
            in_range.append(contig)
    # Return (sub-)contigs over min
    return [contig for contig in in_range if len(contig) > min]

# Decompose
def decompose(indir, outdir, min, max):
    files = sorted(glob.glob(os.path.join(indir, '*.bz2'))) # Fetch all files, e.g. with a glob.glob
    
    idx = 0
    maxlen = 0

    # TODO: check for already decomposed files

    for file in files:
        # Decompress bz2 file, extract lines
        with bz2.open(file, "rb") as f:
            fasta = f.readlines()
        
        # Extract sequences from lines, then lines from sequence
        name = ''
        seq = ''
        for file in files:
            with bz2.open(file, 'rt') as f: # read as text
                fasta = f.readlines()

            # Extract sequences from lines, then lines from sequence
            name = ''
            seq = ''
            for line in fasta:
                if line[0] == '>':
                    # If seq is not empty, we have found the end of a sequence; extract contigs and write to files
                    if seq != '':
                        contigs = scaffold_to_contigs(seq, min=min, max=max)

                        for contig in contigs:
                            # Generate filename, write as .txt, iterate idx
                            filename = str(idx) + '_' + os.path.basename(file) + '_' + name + '.txt'
                            with open(os.path.join(outdir, filename), 'w') as f:
                                f.write(contig)
                            
                            idx += 1

                    # Update name
                    name = line[1:-1]
                else:
                    seq += line.strip(' \n')
    
    print(idx, 'contig files created. Max length = ', maxlen)


# TODO: check for already existing files + idx so this can be ran multiple times

# Parse command line args
parser = argparse.ArgumentParser(description='Decompose dataset into contigs.')
parser.add_argument('--indir',      type=str,   default='./data-scratch/',  help='datset source directory')
parser.add_argument('--outdir',     type=str,   default='./data/',          help='directory to store output files')
parser.add_argument('--min',        type=int,   default=500,                help='minimum contig size')
parser.add_argument('--max',        type=int,   default=4096,               help='maximum contig size')

args = parser.parse_args()

# Run function
decompose(args.indir, args.outdir, args.min, args.max)