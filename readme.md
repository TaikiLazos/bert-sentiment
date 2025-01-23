# Commands info

```
# To train
python src/main.py --mode train --data_path data/small_data.csv --batch_size 10 --learning_rate 2e-5 --epochs 10

# For inference
python src/main.py --mode eval --model_path .\outputs\overfitting_07750\best_model.pt

```