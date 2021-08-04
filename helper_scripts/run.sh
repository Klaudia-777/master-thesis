#!/bin/bash -l
#SBATCH --partition=plgrid-gpu
#SBATCH --gres=gpu
#SBATCH --nodes=12
#SBATCH --time 03:00:00
#SBATCH --job-name counter
#SBATCH --output counter-log-%J.txt

#===============================================================================
# global config
#===============================================================================

cd ${SLURM_SUBMIT_DIR}

module add plgrid/tools/python

pip install norfair[video]
pip install yolov5

SUPERVISOR_HOSTNAME=`/bin/hostname`
HOSTSS=`scontrol show hostnames`
NHOSTS=`echo ${HOSTSS} | wc -w`
echo NHOSTS ${NHOSTS}
HOSTNAMES=`echo ${HOSTSS} | sed "s/\b$SUPERVISOR_HOSTNAME\b//g"`

#===============================================================================
# setup
#===============================================================================

echo `date +"%H:%M:%S"` starting

#===============================================================================
# exec
#===============================================================================


videos=($(ls videos/2021_03_30/*.mp4*))
counter=0
for WORKER_HOST in ${HOSTNAMES}; do
	counter2=0

		srun -w${WORKER_HOST} -N1 -n1 \
				python3 yolo_counter.py \
				--img_size=1920 \
				--device=cuda \
				--files ${PLG_USER_STORAGE}/${videos[${counter}]} \
				--detector_path yolov5l.pt \
				--track_points=centroid \
				--classes 0 1 2 80&
	echo `date +"%H:%M:%S"` path  ${PLG_USER_STORAGE}/${videos[${counter}]}

	((counter=${counter}+1))

	
done

 python3 yolo_counter.py --img_size=1920 --files ${PLG_USER_STORAGE}/${videos[${counter}]} --detector_path yolov5l.pt --track_points=centroid --classes 0 1 2 80
 echo `date +"%H:%M:%S"` plik ${videos[${counter}]}

sleep 20

echo `date +"%H:%M:%S"` done

while true; do
	sleep 2
done