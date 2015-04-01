#! /bin/bash

MODEL="3zpz.pdb"
MAP="2325.map"
RESOLUTION=8.9

powerfit $MODEL $MAP $RESOLUTION -a 4.71 -g -d results/lcc -c O
powerfit $MODEL $MAP $RESOLUTION -a 4.71 -g -d results/cw-lcc -cw -c O
powerfit $MODEL $MAP $RESOLUTION -a 4.71 -g -d results/l-lcc -l -c O
powerfit $MODEL $MAP $RESOLUTION -a 4.71 -g -d results/l-cw-lcc -l -cw -c O
