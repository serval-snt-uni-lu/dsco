#! /bin/bash

if [ -f  /etc/profile ]; then
    .  /etc/profile
fi


module load lang/Python/2.7.6-goolf-1.4.10
source /home/users/dli/pyenv/bin/activate

cd /work/users/dli/dsco

python segmentation.py $@
