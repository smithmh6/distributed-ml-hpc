#!/bin/bash

#SBATCH --job-name=MOE-Test
#SBATCH --output=/var/nfs/users/hsmith/test_moe/Uster_Plastic-ID_MOE_run6_%j.out
#SBATCH --error=/var/nfs/users/hsmith/test_moe/Uster_Plastic-ID_MOE_run6_%j.err
#SBATCH --export=ALL,LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=6

## activate python environment
source /home/$USER/env/bin/activate

mpiexec -n 6 python3 moepy/run.py -c /home/$USER/moepy/config/config.ini &
wait

# deactivate environment
deactivate
