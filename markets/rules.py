import os
import pandas as pd
from markets.helpers import DATA_PATH, get_features_df_filename
from mlxtend.frequent_patterns import apriori, association_rules

MINIMUM_OCCURRENCES = 2


def get_rules_filename(currency):  # todo test
    return os.path.join(DATA_PATH, currency + "_rules.csv")


def extract_rules_to_file(df, currency):
    number_of_rows = len(df.index)
    min_support = round(MINIMUM_OCCURRENCES / (number_of_rows), 5)

    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence")
    rules.drop(columns=["leverage", "conviction"], inplace=True)

    cols_with_strings = ["antecedants", "consequents"]
    cols_with_numbers = [c for c in list(rules) if c not in cols_with_strings]

    rules["antecedants"] = rules["antecedants"].apply(", ".join)
    rules["consequents"] = rules["consequents"].apply(", ".join)
    rules[cols_with_numbers] = rules[cols_with_numbers].round(8)

    rules.to_csv(get_rules_filename(currency), index=False)
    return rules


def get_rules_data(currency):
    rules_data = pd.read_csv(get_rules_filename(currency))
    return rules_data.to_dict("split")


def build_all_rules():
    for currency in ["USD", "MEX", "EUR"]:
        filename = get_features_df_filename(currency)  # todo test co jak nie ma?
        features_df = pd.read_csv(filename)
        extract_rules_to_file(features_df, currency)


if __name__ == '__main__':
    build_all_rules()
