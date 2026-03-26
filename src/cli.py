import argparse, pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=int)
    parser.add_argument("--model", default="als")
    parser.add_argument("--topn", type=int, default=10)
    args = parser.parse_args()

    model = pickle.load(open(f"models/{args.model}.pkl","rb"))
    recs  = model.recommend(args.user, N=args.topn)
    for movie_id, score in recs:
        print(f"{movie_id}: {score:.3f}")

if __name__ == "__main__":
    main()