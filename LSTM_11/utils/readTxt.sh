#!/usr/bin/env sh
#export LD_LIBRARY_PATH=/mnt/lustre/share/mvapich2-2.2rc1/lib

thread_num=16

for((i=0;i<${thread_num};i++));do 
{
  srun --partition=bj11part python readTxt.py ${i}
}&
done
wait
