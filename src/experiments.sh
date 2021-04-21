#!/bin/bash

#chmod 777 ./experiments.sh
#nohup ./experiments.sh &> ./experiments-logs/experiments.log &

source ../../env_exist2021/bin/activate
exp1=roberta_multitask
echo "${exp1}"
python3 train.py --basenet roberta --task multitask --model_path ../models/$exp1.pt > ./experiments-logs/experiment_$exp1.log 2>&1
python3 generate_submissions.py --basenet roberta --task multitask --model_path ../models/$exp1.pt --output_path ../submissions/submission_$exp1.tsv > ./experiments-logs/submission_$exp1.log 2>&1

gsutil cp ./experiments-logs/experiment_$exp1.log gs://exist2021/experiments-logs
gsutil cp ./experiments-logs/submission_$exp1.log gs://exist2021/experiments-logs
gsutil cp ../models/$exp1.pt gs://exist2021/models
gsutil cp ../submissions/submission_$exp1.tsv gs://exist2021/submissions

#remove .logs and models
rm ./experiments-logs/experiment_$exp1.log
rm ./experiments-logs/submission_$exp1.log
rm ../models/$exp1.pt

exp2=roberta_task1_concat_metwo
echo "${exp2}"
python3 train.py --basenet roberta --concat_metwo_train --model_path ../models/$exp2.pt > ./experiments-logs/experiment_$exp2.log 2>&1
python3 generate_submissions.py --basenet roberta --model_path ../models/$exp2.pt --output_path ../submissions/submission_$exp2.tsv > ./experiments-logs/submission_$exp2.log 2>&1

gsutil cp ./experiments-logs/experiment_$exp2.log gs://exist2021/experiments-logs
gsutil cp ./experiments-logs/submission_$exp2.log gs://exist2021/experiments-logs
gsutil cp ../models/$exp2.pt gs://exist2021/models
gsutil cp ../submissions/submission_$exp2.tsv gs://exist2021/submissions

#remove .logs and models
rm ./experiments-logs/experiment_$exp2.log
rm ./experiments-logs/submission_$exp2.log
rm ../models/$exp2.pt

exp3=roberta_task1
echo "${exp3}"
python3 train.py --basenet roberta --model_path ../models/$exp3.pt > ./experiments-logs/experiment_$exp3.log 2>&1
python3 generate_submissions.py --basenet roberta --model_path ../models/$exp3.pt --output_path ../submissions/submission_$exp3.tsv > ./experiments-logs/submission_$exp3.log 2>&1

gsutil cp ./experiments-logs/experiment_$exp3.log gs://exist2021/experiments-logs
gsutil cp ./experiments-logs/submission_$exp3.log gs://exist2021/experiments-logs
gsutil cp ../models/$exp3.pt gs://exist2021/models
gsutil cp ../submissions/submission_$exp3.tsv gs://exist2021/submissions

#remove .logs and models
rm ./experiments-logs/experiment_$exp3.log
rm ./experiments-logs/submission_$exp3.log
rm ../models/$exp3.pt

exp4=roberta_task2
echo "${exp4}"
python3 train.py --basenet roberta --task 2 --model_path ../models/$exp4.pt > ./experiments-logs/experiment_$exp4.log 2>&1
python3 generate_submissions.py --basenet roberta --task 2 --model_path ../models/$exp4.pt --output_path ../submissions/submission_$exp4.tsv > ./experiments-logs/submission_$exp4.log 2>&1

gsutil cp ./experiments-logs/experiment_$exp4.log gs://exist2021/experiments-logs
gsutil cp ./experiments-logs/submission_$exp4.log gs://exist2021/experiments-logs
gsutil cp ../models/$exp4.pt gs://exist2021/models
gsutil cp ../submissions/submission_$exp4.tsv gs://exist2021/submissions

#remove .logs and models
rm ./experiments-logs/experiment_$exp4.log
rm ./experiments-logs/submission_$exp4.log
rm ../models/$exp4.pt

echo "Finished"

sudo shutdown -h now