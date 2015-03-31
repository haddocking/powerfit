#! /bin/bash

powerfit 2ykr.pdb 1884.map 9.8 -a 4.71 -g -d results/lcc -c W
powerfit 2ykr.pdb 1884.map 9.8 -a 4.71 -g -d results/cw-lcc -cw -c W
powerfit 2ykr.pdb 1884.map 9.8 -a 4.71 -g -d results/l-lcc -l -c W
powerfit 2ykr.pdb 1884.map 9.8 -a 4.71 -g -d results/l-cw-lcc -l -cw -c W
