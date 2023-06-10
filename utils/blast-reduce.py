import sys

# get filenames
blastfile = '/fs/nexus-scratch/rhaworth/SRS042628-blast-out.txt'
if len(sys.argv) > 1:
    blastfile = sys.argv[1]

# open file, iterate over lines
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
