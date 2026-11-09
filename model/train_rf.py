"""Random forest per task on the clue table, GroupKFold by child. Owner: Daryn (Nov 16-19).

Saves models/task{n}_rf.joblib as {"model": fitted classifier, "features": CLUE_NAMES[n] without
duration_s} (the format model/predict.py loads). Gate on Nov 19: macro-F1 and kappa above the
baseline. For task 2 also try a per-second forest (one-second windows labelled with the score of
their try, probabilities averaged over the try) and keep whichever scores higher.
"""


def main():
    raise NotImplementedError("Daryn, Nov 16: model/train_rf.py")


if __name__ == "__main__":
    main()
