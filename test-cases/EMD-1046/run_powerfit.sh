#! /bin/bash

ROOT="results/cis"
powerfit 1gru.pdb 1046.map 23 -a 10 -n 7 -c A -d ${ROOT}/lcc
powerfit 1gru.pdb 1046.map 23 -a 10 -n 7 -c A -cw -d ${ROOT}/cw-lcc
powerfit 1gru.pdb 1046.map 23 -a 10 -n 7 -c A -l -d ${ROOT}/l-lcc
powerfit 1gru.pdb 1046.map 23 -a 10 -n 7 -c A -l -cw -d ${ROOT}/l-cw-lcc

ROOT="results/trans"
powerfit 1gru.pdb 1046.map 23 -a 10 -n 7 -c H -d ${ROOT}/lcc
powerfit 1gru.pdb 1046.map 23 -a 10 -n 7 -c H -cw -d ${ROOT}/cw-lcc
powerfit 1gru.pdb 1046.map 23 -a 10 -n 7 -c H -l -d ${ROOT}/l-lcc
powerfit 1gru.pdb 1046.map 23 -a 10 -n 7 -c H -l -cw -d ${ROOT}/l-cw-lcc

ROOT="results/groes"
powerfit GroES_1gru.pdb 1046.map 23 -a 10 -n 1 -d ${ROOT}/lcc
powerfit GroES_1gru.pdb 1046.map 23 -a 10 -n 1 -cw -d ${ROOT}/cw-lcc
powerfit GroES_1gru.pdb 1046.map 23 -a 10 -n 1 -l -d ${ROOT}/l-lcc
powerfit GroES_1gru.pdb 1046.map 23 -a 10 -n 1 -l -cw -d ${ROOT}/l-cw-lcc
