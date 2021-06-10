#!/usr/bin/env bash
python3 dreamer_generate.py --logdir ./logdir/simulations/dmc_walker_stand/1 --task dmc_walker_stand --episodes 300
python3 dreamer_generate.py --logdir ./logdir/sim ulations/dmc_walker_run/1 --task dmc_walker_run --episodes 300
python3 dreamer_generate.py --logdir ./logdir/simulations/dmc_walker_walk/1 --task dmc_walker_walk --episodes 300
python3 dreamer_generate.py --logdir ./logdir/simulations/dmc_hopper_stand/2 --task dmc_hopper_stand --episodes 300
