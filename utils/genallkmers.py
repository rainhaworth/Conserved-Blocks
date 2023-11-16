# generate a list of all kmers for some k
k = 4
outfile = str(k) + 'mers.txt'
kmers = []
num2nucleotide = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
with open(outfile, 'w') as f:
    for i in range(4 ** k):
        # compute kmer "digits" as base 4 number
        # then convert to character with dictionary
        kmer = ''
        for j in range(k):
            kmer += num2nucleotide[(i // (4 ** j)) % 4]
        f.write(kmer + '\n')