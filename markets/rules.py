from mlxtend.frequent_patterns import apriori, association_rules

MINIMUM_OCCURRENCES = 2


def extract_rules_to_file(df, output_filename):
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

    rules.to_csv(output_filename, index=False)
    return rules

