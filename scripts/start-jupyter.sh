#!/bin/bash
#SBATCH --partition devel
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 8G
#SBATCH --time 6:00:00
#SBATCH --job-name jupyter-notebook
#SBATCH --output jupyter-notebook-%J.log

# copy from https://docs.ycrc.yale.edu/clusters-at-yale/guides/jupyter/

# get tunneling info
XDG_RUNTIME_DIR=""
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$(whoami)
cluster=$(hostname -f | awk -F"." '{print $(NF-2)"."$(NF-1)"."$NF}')

# print tunneling instructions jupyter-log
echo -e "MacOS or linux terminal command to create your ssh tunnel

ssh -N -L ${port}:${node}:${port} ${user}@${cluster}

Use a Browser on your local machine to go to: https://localhost:${port} 
"

# load modules or conda environments here
eval "$(conda shell.bash hook)"
conda activate nlp
jupyter-notebook --no-browser --port=${port} --ip=${node}