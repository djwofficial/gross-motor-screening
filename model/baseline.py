"""Most-common-score baseline per task, GroupKFold by child. Owner: Marty (Nov 13).

Expected (Job Card 1, our earlier check with scikit-learn 1.9.1): macro-F1 0.19 / 0.20 / 0.26
for tasks 1 / 2 / 3, quadratic weighted kappa 0.00, and the child-leak check prints PASS.
Tools: DummyClassifier(strategy="most_frequent"), GroupKFold(5), f1_score(average="macro"),
cohen_kappa_score(weights="quadratic").
"""


def main():
    raise NotImplementedError("Marty, Nov 13: model/baseline.py")


if __name__ == "__main__":
    main()
