#!/bin/bash

echo "Note: conda version must be 4.6+ for this to work."
echo "Conda version: $(conda --version)"

eval "$(conda shell.bash hook)"
conda activate tf2

export PYTHONPATH=/home/noesyslab/remote_project_mirrors/KyrLinuxLaptop/RIS_simulation:$PYTHONPATH


for f in Power_*.json; do

      echo "Running experiment from parameters file $f ."
      echo "..."
      sleep 3;

      python ../train_UCB.py $f
      python ../train_DQN.py $f
      python ../train_Neural_Softmax.py $f

      sleep 3;

done;