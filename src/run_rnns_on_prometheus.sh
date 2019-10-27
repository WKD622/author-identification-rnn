#!/usr/bin/env bash

results=results
to_run=(
    100, # hidden_size
    2, # num_layers
    5, # num_epochs
    40, # batch_size
    50, # timesteps
    0.04, # learning_rate
    100, # authors_size
    48, # vocab_size
    1, # save_path
    ../data/dutch/tensors/known/, # tensors_path
    DU #
)

for i in "${to_run[@]}" ; do
    IFS=",";
    set ${i};
    mkdir -p ${results}/$9
    echo $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11};
done