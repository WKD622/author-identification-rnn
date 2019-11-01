#!/usr/bin/env bash
for f in ./scripts/*.sh; do  # or wget-*.sh instead of *.sh
  sbatch "$f" 
done
