#!/usr/bin/env bash
current_dir=.
source=plgjakubziarko@prometheus.cyfronet.pl:/net/people/plgjakubziarko/results
mkdir -p results
scp -rp ${source} "${current_dir}/results"