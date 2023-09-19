# generate synthetic dataset with conserved regions
# currently generates contiguous blocks with no variance in a fixed number of samples
import os
import random
import argparse

def gen_seq(length):
    return ''.join(random.choice('ACGT') for _ in range(length))

def gen_synth_dataset(args):
    # generate conserved region
    conserved = gen_seq(args.conserved_len)
    num_seqs_with_conserved = int(args.dataset_size * args.conserved_fraction)
    print("Conserved region:", conserved)

    with open(os.path.join(args.output_dir, args.output_file), 'w') as f:
        for i in range(args.dataset_size):
            if num_seqs_with_conserved > 0:
                # generate sequence with conserved block
                seq = gen_seq(args.seq_len - args.conserved_len)
                
                # pick random insertion point
                insert_point = random.randint(0, len(seq)-1)

                # generate final sequence and description
                seq = seq[:insert_point] + conserved + seq[insert_point:]
                desc = ">Includes conserved region\n"

                # decrement
                num_seqs_with_conserved -= 1
            else:
                # just generate a random sequence
                seq = gen_seq(args.seq_len)
                desc = ">No conserved region\n"

            f.write(desc)
            f.write(seq)
            f.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_size',       type=int,   default=1000, help='Number of sequences in the dataset')
    parser.add_argument('--seq_len',    type=int,   default=4096, help='Length of each sequence')

    parser.add_argument('--conserved_len',   type=int,   default=500, help='Length of conserved block')
    parser.add_argument('--conserved_fraction', type=float, default=0.8, help='Fraction of sequences with conserved block')

    parser.add_argument('--output_file',        type=str,   default='synthetic_dataset.fasta', help='Output file in .fasta format')
    parser.add_argument('--output_dir',         type=str,   default='./data/', help='Output directory')

    args = parser.parse_args()

    gen_synth_dataset(args)

    print(f"Synthetic DNA dataset has been generated and saved to '{os.path.join(args.output_dir, args.output_file)}'.")
