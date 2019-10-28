import sys
from os import walk, system

f = []
for (_, _, filenames) in walk(sys.argv[1]):
    for filename in filenames:
        system('sbatch ' + filename)
