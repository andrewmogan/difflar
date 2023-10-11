#!/bin/bash 

DIRNAME="/pnfs/icarus/scratch/users/amogan//diffusion_sce_off_v09_56_00_01/July2023_v2_18000Evts/ntuple/3070430_12/"
FILETEXT="Supp*.root"
RUNTYPE="XX"
PLANENUM=2
PLOTNAME="plot.pdf"
FILELIST="diffusion_sce_off_list_Sep2023.txt"

make 
./WaveformStudy $DIRNAME $FILETEXT $RUNTYPE $PLANENUM $PLOTNAME $FILELIST
