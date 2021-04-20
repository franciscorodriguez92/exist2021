#!/bin/bash

#chmod 777 ./experiments.sh
#nohup ./experiments.sh &> ./experiments-logs/experiments.log &

source ../../env_exist2021/bin/activate
exp1=bert_task1_concat_metwo
echo "${exp1}"
python3 train.py --sample --batch_size 15 --model_path ../models/$exp1.pt > ./experiments-logs/experiment_$exp1.log 2>&1
python3 generate_submissions.py --sample --batch_size 15 --model_path ../models/$exp1.pt --output_path ../submissions/submission_$exp1.tsv > ./experiments-logs/submission_$exp1.log 2>&1

gsutil cp ./experiments-logs/experiment_$exp1.log gs://exist2021/experiments-logs
gsutil cp ./experiments-logs/submission_$exp1.log gs://exist2021/experiments-logs
gsutil cp ../models/$exp1.pt gs://exist2021/models
gsutil cp ../submissions/submission_$exp1.tsv gs://exist2021/submissions

#remove .logs and models
rm ./experiments-logs/experiment_$exp1.log
rm ./experiments-logs/submission_$exp1.log
rm ../models/$exp1.pt
echo "Finished"
# shutdown -h now