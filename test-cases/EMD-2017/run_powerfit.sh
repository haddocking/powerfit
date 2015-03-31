#! /bin/bash

powerfit 4adv_V.pdb 2017.map 13.5 -a 4.71 -g -d results/lcc
powerfit 4adv_V.pdb 2017.map 13.5 -a 4.71 -g -d results/cw-lcc -cw
powerfit 4adv_V.pdb 2017.map 13.5 -a 4.71 -g -d results/l-lcc -l
powerfit 4adv_V.pdb 2017.map 13.5 -a 4.71 -g -d results/l-cw-lcc -l -cw

#powerfit 4adv_V.pdb 2017.map 13.5 -a 5 -g -d results/no-trimming/l-lcc -l -nt
#powerfit 4adv_V.pdb 2017.map 13.5 -a 5 -g -d results/no-trimming/l-cw-lcc -l -cw -nt
