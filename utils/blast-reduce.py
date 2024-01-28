import sys

# get filenames
#blastfile = '/fs/nexus-scratch/rhaworth/SRS042628-blast-out.txt'
blastfile = '/fs/nexus-scratch/rhaworth/output/prem-micro-blast-out.txt'
if len(sys.argv) > 1:
    blastfile = sys.argv[1]

with open(blastfile, 'r') as f:
    query = ''
    hits = []
    skipcount = 0
    for line in f:
        # search for queries
        if "Query=" in line:
            query = line[7:-1]
            skipcount = 5
            print(line[:-1])
        elif query != '':
            if skipcount != 0:
                skipcount -= 1
                continue
            elif line[0] == '>':
                query = ''
            elif query in line:
                continue
            else:
                print(line[:-1])
# old implementation
# open file, iterate over lines
"""
with open(blastfile, 'r') as f:
    query = ''
    gap = ''
    for line in f:
        # search for queries
        if "Query" in line:
            query = line
        elif query != '' and gap == '':
            gap = line
        elif query != '' and gap != '' and "Sbjct" in line:
            q = query.split()
            s = line.split()
            # skip self-alignments
            if q[1] == s[1]:
                query = ''
                gap = ''
                continue
            
            # print non-self-alignments
            print(query[:-1])
            print(gap[:-1])
            print(line)
            print()

            query = ''
            gap = ''
"""
