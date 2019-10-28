#!/usr/bin/env bash
current_dir=.
source=plgjakubziarko@prometheus.cyfronet.pl:/net/people/plgjakubziarko/author-identification-rnn/src/results
mkdir -p results
scp -rp ${source} "${current_dir}/results"