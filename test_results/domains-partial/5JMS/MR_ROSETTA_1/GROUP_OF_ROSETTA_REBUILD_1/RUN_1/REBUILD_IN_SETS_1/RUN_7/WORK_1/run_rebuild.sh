#!/bin/sh  
pwd
export LD_LIBRARY_PATH=/home/wei/Software/phenix-1.19.2-4158/build/lib:/home/wei/Software/phenix-1.19.2-4158/build/../conda_base/lib
export PATH=/home/wei/Software/phenix-1.19.2-4158/build/bin:/home/wei/Software/phenix-1.19.2-4158/build/bin:/home/wei/Software/phenix-1.19.2-4158/build/bin:/usr/local/cuda-11.7/bin:/home/wei/.t_coffee/bin/linux:/home/wei/Software/phenix-1.19.2-4158/build/bin:/home/linuxbrew/.linuxbrew/bin:/home/linuxbrew/.linuxbrew/sbin:/home/wei/.local/bin:/home/wei/anaconda3/envs/colabfold152/bin:/home/wei/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/home/wei/Software/mmseqs2/mmseqs/bin
cd "/maintank/Wei/automation.colabfold/test45/batch_13/Q9BJF5/5JMS/MR_ROSETTA_1/GROUP_OF_ROSETTA_REBUILD_1/RUN_1/REBUILD_IN_SETS_1/RUN_7/WORK_1"
"/home/wei/Software/rosetta_src_2021.16.61629_bundle/main/source/bin/mr_protocols.linuxgccrelease" @"rebuild.flags" > "rebuild.log" 2>&1 
