#!/usr/bin/python
# Author: Hao WANG
###############################################
# An implementation of paired bootstrap resampling for testing the statistical
# significance of the difference between two systems from (Koehn 2004 @ EMNLP)
# Specified for Categorical F1 Scores.
# Usage: ./bootstrap-resampling.py hypothesis_1 hypothesis_2 reference_1 [ reference_2 ... ]
###############################################


from sklearn.metrics import matthews_corrcoef, f1_score
import sys
import numpy as np
from tqdm import tqdm
# constants
TIMES_TO_REPEAT_SUBSAMPLING = 1000
SUBSAMPLE_SIZE = 0
# if 0 then subsample size is equal to the whole set


def read_order(line):
    """Read one example from the order file."""
    if not line:
        return None
    line = line[:-1]
    order = line.split()
    order = [int(item) for item in order]
    return order


def getAcc(refs, hypos, indices=None):
    return _CalculateAccScores(refs, hypos, indices)


def _CalculateAccScores(refs, hypos, indices=None):
    num = 0
    skipped = 0
    sum_ = 0
    if indices is None:
        candidates = list(range(len(refs)))
    else:
        candidates = indices
    candidates = [
        idx for idx in candidates if not refs[idx].startswith("Other")]

    sample_refs = [refs[idx] for idx in candidates]
    sample_hypos = [hypos[idx] for idx in candidates]

    return f1_score(sample_refs, sample_hypos, average='macro')


def main(argv):
    # checking cmdline argument consistency
    if len(argv) != 4:
        print("Usage: ./bootstrap-hypothesis-difference-significance.py hypothesis_1 hypothesis_2 reference\n", file=sys.stderr)
        sys.exit(1)
    print("reading data", file=sys.stderr)
    # read all data
    data = readAllData(argv)
    # #calculate each sentence's contribution to BP and ngram precision
    # print("rperforming preliminary calculations (hypothesis 1); ", file=sys.stderr)
    # preEvalHypo(data, "hyp1")

    # print("rperforming preliminary calculations (hypothesis 2); ", file=sys.stderr)
    # preEvalHypo(data, "hyp2")

    # start comparing
    print("comparing hypotheses -- this may take some time; ", file=sys.stderr)

    # bootstrap_report(data, "Fuzzy Reordering Scores",  getFRS)
    # bootstrap_report(data, "Normalized Kendall's Tau", getNKT)
    bootstrap_report(data, "ACC", getAcc)

#####


def bootstrap_report(data, title, func):
    subSampleIndices = np.random.choice(
        data["size"], SUBSAMPLE_SIZE if SUBSAMPLE_SIZE > 0 else data["size"], replace=True)
    print(f1_score(data["refs"], data["hyp1"], average='macro'))
    print(f1_score(data["refs"], data["hyp2"], average='macro'))

    realScore1 = func(data["refs"], data["hyp1"], subSampleIndices)
    realScore2 = func(data["refs"], data["hyp2"], subSampleIndices)
    subSampleScoreDiffArr, subSampleScore1Arr, subSampleScore2Arr = bootstrap_pass(
        data, func)

    scorePValue = bootstrap_pvalue(
        subSampleScoreDiffArr, realScore1, realScore2)

    (scoreAvg1, scoreVar1) = bootstrap_interval(subSampleScore1Arr)
    (scoreAvg2, scoreVar2) = bootstrap_interval(subSampleScore2Arr)

    print("\n---=== %s score ===---\n" % title)
    print("actual score of hypothesis 1: %f" % realScore1)
    print("95/100 confidence interval for hypothesis 1 score: %f +- %f" %
          (scoreAvg1, scoreVar1) + "\n-----\n")
    print("actual score of hypothesis 1: %f" % realScore2)
    print("95/100 confidence interval for hypothesis 2 score:  %f +- %f" %
          (scoreAvg2, scoreVar2) + "\n-----\n")
    print("Assuming that essentially the same system generated the two hypothesis translations (null-hypothesis),\n")
    print("the probability of actually getting them (p-value) is: %f\n" %
          scorePValue)


#####
def bootstrap_pass(data, scoreFunc):
    subSampleDiffArr = []
    subSample1Arr = []
    subSample2Arr = []

    # applying sampling
    for idx in tqdm(range(TIMES_TO_REPEAT_SUBSAMPLING), ncols=80, postfix="Subsampling"):
        subSampleIndices = np.random.choice(
            data["size"], SUBSAMPLE_SIZE if SUBSAMPLE_SIZE > 0 else data["size"], replace=True)
        score1 = scoreFunc(data["refs"], data["hyp1"], subSampleIndices)
        score2 = scoreFunc(data["refs"], data["hyp2"], subSampleIndices)
        subSampleDiffArr.append(abs(score2 - score1))
        subSample1Arr.append(score1)
        subSample2Arr.append(score2)
    return np.array(subSampleDiffArr), np.array(subSample1Arr), np.array(subSample2Arr)
#####
#
#####


def bootstrap_pvalue(subSampleDiffArr, realScore1, realScore2):
    realDiff = abs(realScore2 - realScore1)

    # get subsample difference mean
    averageSubSampleDiff = subSampleDiffArr.mean()

    # calculating p-value
    count = 0.0

    for subSampleDiff in subSampleDiffArr:
        if subSampleDiff - averageSubSampleDiff >= realDiff:
            count += 1
    return count / TIMES_TO_REPEAT_SUBSAMPLING

#####
#
#####


def bootstrap_interval(subSampleArr):
    sortedArr = sorted(subSampleArr, reverse=False)
    lowerIdx = int(TIMES_TO_REPEAT_SUBSAMPLING / 40)
    higherIdx = TIMES_TO_REPEAT_SUBSAMPLING - lowerIdx - 1

    lower = sortedArr[lowerIdx]
    higher = sortedArr[higherIdx]
    diff = higher - lower
    return (lower + 0.5 * diff, 0.5 * diff)

#####
# read 2 hyp and 1 to \infty ref data files
#####


def readAllData(argv):
    print(argv)
    assert len(argv[1:]) == 3
    hypFile1, hypFile2 = argv[1:3]
    refFile = argv[3]
    result = {}
    # reading hypotheses and checking for matching sizes
    result["hyp1"] = [line for line in open(hypFile1)]
    result["size"] = len(result["hyp1"])

    result["hyp2"] = [line for line in open(hypFile2)]
    assert len(result["hyp2"]) == len(result["hyp1"])

    refDataX = [line for line in open(refFile)]
    # updateCounts($result{ngramCounts}, $refDataX);
    result["refs"] = refDataX
    # print(result)
    return result


if __name__ == '__main__':
    main(sys.argv)
