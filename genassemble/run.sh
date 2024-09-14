#!/bin/bash
#SBATCH --output=/fs/nexus-scratch/rhaworth/output/evosim.o
#SBATCH --qos=high
#SBATCH --mem=128gb

source ~/.bashrc; conda activate /fs/nexus-scratch/rhaworth/env/megahit

# clear all csvs in directory
rm *.csv

# set parameters
seqlens=("1000 2000 5000 10000")
psubs=("0.0 0.005 0.01 0.02 0.03")
eirs=("0.0005 0.001 0.005 0.01 0.05")
eiss=("1.0 5.0 10.0 25.0 100.0")

# print params to csv in order shown above
pfn="params.csv"
for seqlen in $seqlens
do
    echo -n "$seqlen," >> $pfn
done
echo >> $pfn
for psub in $psubs
do
    echo -n "$psub," >> $pfn
done
echo >> $pfn
for eir in $eirs
do
    echo -n "$eir," >> $pfn
done
echo >> $pfn
for eis in $eiss
do
    echo -n "$eis," >> $pfn
done

# run experiments

# length
for seqlen in $seqlens
do
    # substiution probability
    for psub in $psubs
    do
        mfn="mega-l${seqlen}-p${psub}.csv"
        nfn="nucm-l${seqlen}-p${psub}.csv"
        # expected indel rate
        for eir in $eirs
        do
            # expected indel size
            for eis in $eiss
            do
                python evosim.py -n 1 -l $seqlen -p $psub -r $eir -s $eis > test.fa
                ~/art_bin_MountRainier/art_illumina -ss HSXn -sam -i test.fa -p -l 150 -f 20 -m 200 -s 10 -o testreads > /dev/null
                # remove testmega if it exists; otherwise megahit won't run
                rm -rf testmega
                # run megahit, output average contig length to out.csv
                ~/MEGAHIT-1.2.9-Linux-x86_64-static/bin/megahit -1 testreads1.fq -2 testreads2.fq -o testmega 2>&1 | grep -o 'total [0-9]* bp' | awk -v sl="${seqlen}" '{printf $2 / sl}' >> $mfn
                echo -n "," >> $mfn

                # mummer
                # split test.fa into 2 separate fasta files, xaa and xab
                split -l 2 test.fa
                # align split files
                /fs/nexus-scratch/rhaworth/mummer-4.0.0/bin/nucmer xaa xab
                # compute score
                /fs/nexus-scratch/rhaworth/mummer-4.0.0/bin/show-coords out.delta | tail --lines=+6 | awk -v sl="${seqlen}" '{total += ($7 + $8) * $10 / 100} END {printf total / (sl * 2)}' >> $nfn
                echo -n "," >> $nfn
            done
            echo >> $mfn
            echo >> $nfn
        done
    done
done