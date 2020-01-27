#!/usr/bin/env bash
# Collect training data (use -visible to visualize the data while collecting)
python env/block_env.py \
--length 20 \
--n_trials 50 \
--n_contexts 1
