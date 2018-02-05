!/bin/sh
echo 'submitting job..'
### General options

### –- specify queue --
##BSUB -q gputitanxpascal
##BSUB -q gpuk80
#BSUB -q gpuv100

### -- set the job Name --
#BSUB -J train_restnet_localization
##BSUB -J train_restnet_classes_extra_conv
##BSUB -J train_restnet_classes_flatten
##BSUB -J train_restnet_localization_no_activation

### -- ask for number of cores (default: 1) --
# BSUB -n 4

### -- reserve GPUs exclusively
# BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --
#BSUB -W 24:00

# request 5GB of memory
#BSUB -R "rusage[mem=8GB]"

### -- set the email address --
#BSUB -u jakub.czerny@tum.de

### -- send notification at start --
#BSUB -B

### -- send notification at completion --
#BSUB -N

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o Output_%J.out
#BSUB -e Error_%J.err

module load python/2.7.12_ucs4
module load cuda/8
module load cudnn/v6.0-prod

source ~/envs/dl4cv/bin/activate

nvidia-smi
python

echo 'starting main.py..'
time python localization/resnet50.py > job_output_localization.txt
#time python localization/resnet50.py > job_output_localization_extra_conv.txt
#time python localization/resnet50.py > job_output_localization_flatten.txt
#time python localization/resnet50.py > job_output_localization_no_activation.txt

echo 'job finished'
