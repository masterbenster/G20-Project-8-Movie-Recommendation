# G20-Project-8-Movie-Recommendation
This is the codebase that we will be using to create our movie recommendation system based off this kaggle datasets:
https://grouplens.org/datasets/movielens/10m/,
https://grouplens.org/datasets/movielens/1m/

to run use commands as such:
python -m src.cli --user 1 --model als --topn 10
python -m src.cli --user 1 --model knn --topn 5
python -m src.cli --user 1 --model neumf --topn 10