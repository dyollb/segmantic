from enum import Enum


class EnsembleCombination(str, Enum):
    mean = "mean"
    vote = "vote"
    select_best = "select_best"
