!/bin/sh
echo 'submitting job..'
### General options

### –- specify queue --
#BSUB -q gputitanxpascal

### -- set the job Name --
#BSUB -J train_VGG16

### -- ask for number of cores (default: 1) --
# BSUB -n 2

### -- reserve GPUs exclusively
# BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --
#BSUB -W 24:00

# request 5GB of memory
#BSUB -R "rusage[mem=5GB]"

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

module load python/2.7.12_ucs4
module load cuda/8.0
module load cudnn/v6.0-prod

source ~/envs/dl4cv/bin/activate

nvidia-smi
python

# here follow the commands you want to execute
echo 'starting main.py..'
time python models/vgg16.py > job_output.txt
echo 'job finished'
