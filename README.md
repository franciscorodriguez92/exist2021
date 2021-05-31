# Participation nlp_uned_team in EXIST 2021 

[Link to competition website](http://nlp.uned.es/exist2021/)

Link to paper coming soon!

## Instructions

`python >= 3.8` needed

0. Install requirements

```
pip install -r requirements.txt
```

1. Get data

You can contact me at frodriguez.sanchez@invi.uned.es to get data (script get_data.py).

Now, in `data/` you will have the datasets needed.

**NOTE**: Yo need to check that validation split is available at data/input/EXIST2021_dataset-test/EXIST2021_dataset/validation/EXIST2021_validation_split.tsv


2. Train Transformers


To train a bert model for task 1 with default parameters, just run:

```
cd /src
python train.py 
```

For instance, to train a roberta model using a multitask approach, just run:

```
cd /src
python train.py --basenet roberta --task multitask --model_path ../models/roberta_multitask.pt
```


### Generate submissions

Run this command to generate a .tsv file with submissions (a _task2.tsv file is generated when generating submissions for task 2 or when multitasking):
```
cd /src
python generate_submissions.py
```

For instance (it will generate a roberta_multitask_task2.tsv file too):
```
cd /src
python generate_submissions.py --basenet roberta --task multitask --model_path ../models/roberta_multitask.pt --output_path ../submissions/roberta_multitask.tsv
```

## Contact information

Please feel free to contact me at frodriguez.sanchez@invi.uned.es if you have any questions about our participation or the competition.
