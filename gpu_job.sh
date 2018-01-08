#!/bin/sh
echo 'submitting job..'
### General options

### –- specify queue --
#BSUB -q gputitanxpascal

### -- set the job Name --
#BSUB -J train_VGG16

### -- ask for number of cores (default: 1) --
# BSUB -n 2

### -- reserve GPUs exclusively
# BSUB -R "rusage[ngpus_excl_p=1]"

### -- set walltime limit: hh:mm --
#BSUB -W 24:00

### -- set the email address --
#BSUB -u kuba.czerny@poczta.fm

### -- send notification at start --
#BSUB -B

### -- send notification at completion --
#BSUB -N

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o Output_%J.out
#BSUB -e Error_%J.err

module load cuda/8.0
module load cudnn

# here follow the commands you want to execute
echo 'starting main.py..'
##time python main.py > job_output.txt
time python models/vgg16.py > job_output.txt
echo 'job finished'
