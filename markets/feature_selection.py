import os
import pandas as pd
import subprocess
import tempfile
from markets.dataset import TweetsDataSet
from markets.tweets_features_extraction import remark_features

ASSOCIATION_MODEL_FILE = os.path.join(os.path.dirname(__file__), "assoc_model.pickle")
pd.set_option('display.width', 1500)
WEKA_JAR_FILE = os.path.join(os.path.dirname(__file__), "weka-3-8-2", "weka.jar")


def read_features_from_file(filename):
    return [line.strip() for line in open(filename, 'r')]


def select_features(df, filename):
    if os.path.isfile(filename):
        features = read_features_from_file(filename)
    else:
        features = get_features_from_weka(df)
        save_selected_features(features, filename)

    main_df = filter_features(df, features)
    return main_df


def save_selected_features(list_of_features, filename):
    with open(filename, "w") as f:
        f.write("\n".join(list_of_features))


def filter_features(dataset, features_to_leave, with_dropping=True):
    sifted_dataset = TweetsDataSet(dataset.get_no_features_df())
    remark_features(sifted_dataset, features_to_leave, with_dropping)
    return sifted_dataset


def save_features_with_target_to_file(df, filename):
    df = df.drop(columns=["Text", "Tweet_sentiment"])
    df.to_csv(filename, index=False)


def run_weka_with_file(temp_filename):
    command = ['java', '-classpath', WEKA_JAR_FILE,
               'weka.attributeSelection.WrapperSubsetEval',
               '-T', '0.5',
               '-B', 'weka.classifiers.bayes.NaiveBayesMultinomial',
               '-s', 'weka.attributeSelection.BestFirst',
               '-i', temp_filename]
    stdoutdata = subprocess.getoutput(command)
    found_features = False
    features = []
    for l in stdoutdata.split("\n"):
        if found_features:
            features.append(l.strip())
        if l.startswith("Selected attributes:"):
            found_features = True
    features = [f for f in features if f]  # make sure no empty line added
    if not features:
        print(stdoutdata)
        raise Exception("Problem while doing feature selection with Weka library.")

    return features


def get_features_from_weka(df):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as fp:
        save_features_with_target_to_file(df, fp.name)
        features = run_weka_with_file(fp.name)
    return features
