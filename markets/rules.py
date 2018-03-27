from mlxtend.frequent_patterns import apriori, association_rules
from markets.association import FEATURES_WITH_EFFECT_FILE
import pandas as pd


MINIMUM_OCCURENCIES = 2


def get_rules(df):
    # df = pd.read_csv(FEATURES_WITH_EFFECT_FILE)
    df = df.drop(columns=["Market_change", "Tweet_sentiment"])

    number_of_rows = len(df.index)
    min_support = round(MINIMUM_OCCURENCIES/number_of_rows, 5)

    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence")
    rules.drop(columns=["leverage", "conviction"])
    return rules

