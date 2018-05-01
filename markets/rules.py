"""
Contains functions used to do association rules learning.
"""
import pickle
from itertools import groupby
from collections import namedtuple
from mlxtend.frequent_patterns import apriori, association_rules

MINIMUM_OCCURRENCES = 2

RulesSet = namedtuple("RulesSet", "words_set support confidence rules")
Rule = namedtuple("Rule", "antecedants antecedent_support consequents consequent_support lift")


def extract_rules_to_file(df, output_filename):
    number_of_rows = len(df.index)
    min_support = round(MINIMUM_OCCURRENCES / number_of_rows, 5)

    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence")
    rules = process_df(rules)
    rules_as_dict = rules.to_dict('records')
    rules_sets = group_into_sets(rules_as_dict)

    with open(output_filename, "wb") as f:
        pickle.dump(rules_sets, f)

    return rules_sets


def process_df(rules_df):
    rules_df.drop(columns=["leverage", "conviction"], inplace=True)
    rules_df = rules_df.apply(lambda row: _append_words_set(row), axis=1)
    rules_df = _fix_lists_and_numbers(rules_df)
    return rules_df


def _fix_lists_and_numbers(rules_df):
    cols_with_strings = ["antecedants", "consequents"]
    cols_with_numbers = [c for c in list(rules_df) if c not in cols_with_strings]

    rules_df["antecedants"] = rules_df["antecedants"].apply(", ".join)
    rules_df["consequents"] = rules_df["consequents"].apply(", ".join)
    rules_df[cols_with_numbers] = rules_df[cols_with_numbers].round(8)
    return rules_df


def _append_words_set(row):
    row["words_set"] = set(row["antecedants"])
    row["words_set"].update(row["consequents"])
    return row


def read_rules_sets(filename):
    with open(filename, "rb") as f:
        rules_sets = pickle.load(f)
        return rules_sets


def group_into_sets(rules):
    def _construct_rule(r):
        lift = r["lift"]
        antecedants = r["antecedants"]
        consequents = r["consequents"]
        antecedent_support = r["antecedent support"]
        consequent_support = r["consequent support"]
        return Rule(antecedants, antecedent_support, consequents, consequent_support, lift)

    rules_sets = []
    for words_set, group in groupby(rules, lambda x: x["words_set"]):
        group = list(group)
        rules = [_construct_rule(r) for r in group]
        r = RulesSet(words_set, group[0]["support"], group[0]["confidence"], rules)
        rules_sets.append(r)
    return rules_sets
