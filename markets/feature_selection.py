import os
import pandas as pd
import subprocess
import tempfile
import copy
from markets.phrases_extractor import PhrasesExtractor

ASSOCIATION_MODEL_FILE = os.path.join(os.path.dirname(__file__), "assoc_model.pickle")
pd.set_option('display.width', 1500)
WEKA_JAR_FILE = os.path.join(os.path.dirname(__file__), "weka-3-8-2", "weka.jar")


def get_features_from_file(filename):  # todo przenisc wyzej
    return [line.strip() for line in open(filename, 'r')]


def select_features(df, filename=None):
    has_got_faeture_already_selected = filename and os.path.isfile(filename)  # todo czy są tam featerki
    if has_got_faeture_already_selected:
        features = get_features_from_file(filename)
    else:
        features = get_features_from_weka(df)
        save_selected_features(features, filename)

    main_df = filter_features(df, features)
    main_df.drop_instances_without_features()
    print(main_df.df["Market_change"].size)
    return main_df


def save_selected_features(list_of_features, filename):
    with open(filename, "w") as f:
        f.write("\n".join(list_of_features))


def filter_features(dataset, features_to_leave): # returns copy
    sifted_dataset = copy.deepcopy(dataset)
    sifted_dataset.filter_features(features_to_leave) # TODO uzyc feature extractora
    extr = PhrasesExtractor() # tu jak sie nie zamarkuje od nowa to lepsze accuracy ale mnniej tweetow
    extr.set_features(features_to_leave)
    sifted_dataset.set_phrase_features(extr.extract_features)
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
    features = [f for f in features if f] # make sure no empty line added
    if not features:
        print(stdoutdata)
        raise Exception("Problem while doing feature selection with Weka library.")

    return features


def get_features_from_weka(df):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as fp:
        print(fp.name)
        save_features_with_target_to_file(df, fp.name)
        features = run_weka_with_file(fp.name)
    return features
