"""Random forest per task on the clue table, GroupKFold by child. Owner: Marty (Nov 16-19).

Saves models/task{n}_rf.joblib as {"model": fitted classifier, "features": CLUE_NAMES[n]}
(the format model/predict.py loads). Gate on Nov 19: macro-F1 and kappa above the baseline.
"""


def main():
    raise NotImplementedError("Marty, Nov 16: model/train_rf.py")


if __name__ == "__main__":
    main()
