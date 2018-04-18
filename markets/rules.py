from mlxtend.frequent_patterns import apriori, association_rules
from itertools import groupby
import pickle
from collections import namedtuple

MINIMUM_OCCURRENCES = 2

RulesSet = namedtuple("RulesSet", "words_set support confidence rules")


def _get_words_set(row):
    row["words_set"] = set(row["antecedants"])
    row["words_set"].update(row["consequents"])
    return row


def extract_rules_to_file(df, output_filename):
    number_of_rows = len(df.index)
    min_support = round(MINIMUM_OCCURRENCES / (number_of_rows), 5)

    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence")
    rules.drop(columns=["leverage", "conviction"], inplace=True)

    cols_with_strings = ["antecedants", "consequents"]
    cols_with_numbers = [c for c in list(rules) if c not in cols_with_strings]


    rules = rules.apply(lambda row: _get_words_set(row), axis=1)


    rules["antecedants"] = rules["antecedants"].apply(", ".join)
    rules["consequents"] = rules["consequents"].apply(", ".join)
    rules[cols_with_numbers] = rules[cols_with_numbers].round(8)

    rules_as_dict = rules.to_dict('records')
    rules_sets = group_into_sets(rules_as_dict)

    with open(output_filename, "wb") as f:
        pickle.dump(rules_sets, f)

    return rules_sets


def read_rules_sets(filename):
    with open(filename, "rb") as f:
        rules_sets = pickle.load(f)
        return rules_sets


def group_into_sets(rules):
    rules_sets = []
    for words_set, group in groupby(rules, lambda x: x["words_set"]):
        rules = [r for r in group]
        r = RulesSet(words_set, rules[0]["support"], rules[0]["confidence"], rules)
        rules_sets.append(r)
    return rules_sets
