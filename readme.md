# Political Stance Detection

## Training and Evaluation

### Datasets

Download JSON dataset (ask me) and run `python src/json_transform.py` to preprocess the data.

This will create `data/processed/train.json`, `data/processed/test1.json`, `data/processed/test2.json`, `data/processed/test3.json`.

### Initial Training

```bash
python src/main.py --mode train --data_path data/processed/train.pkl --model_name roberta-base --batch_size 8 --epochs 1 --learning_rate 2e-5
```

## Experimental Results

### Sequential Fine-tuning Experiment
```bash
python src/sequential_eval.py --initial_model outputs/YOUR_MODEL_PATH/best_model.pt
```

We conducted a sequential fine-tuning experiment to test the model's adaptability to new data:

```
+------------------+---------+---------+---------+
| Model            | Test1   | Test2   |   Test3 |
+==================+=========+=========+=========+
| Initial          | 0.9244  | 0.9244  |  0.9173 |
+------------------+---------+---------+---------+
| After test1.json | -       | 0.9537  |  0.9513 |
+------------------+---------+---------+---------+
| After test2.json | -       | -       |  0.9659 |
+------------------+---------+---------+---------+
```

We can see that the model's performance improves after each dataset is added, indicating that the model is able to adapt to new data.


### Finetuning

```bash
python src/finetune.py --base_model outputs/YOUR_MODEL_PATH/best_model.pt --data_path data/processed/new_data.json --epochs 3 --batch_size 8 --learning_rate 2e-5
```

This will create a new model in `outputs/finetuned_<timestamp>/best_model.pt`.


### Inference

```bash
python src/main.py --mode eval --model_path outputs/finetuned_<timestamp>/best_model.pt --data_path data/processed/single_datapoint.json
```

This will output the model's prediction for the single datapoint.
