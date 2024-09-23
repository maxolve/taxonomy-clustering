import pandas as pd

def load_dataset(dataset_name):
    if dataset_name == "zoo":
        try:
            return pd.read_csv('datasets/zoo.csv', sep=',')
        except UnicodeDecodeError:
            return pd.read_csv('datasets/zoo.csv', encoding='ISO-8859-1', sep=',')
    elif dataset_name == "soybean-small":
        try:
            return pd.read_csv('path_to-soybean_small.csv', sep=',')
        except UnicodeDecodeError:
            return pd.read_csv('path_to-soybean_small.csv', encoding='ISO-8859-1', sep=',')
    elif dataset_name == "torno":
        try:
            return pd.read_csv('path_to-torno.csv', sep='\t')
        except UnicodeDecodeError:
            return pd.read_csv('path_to-torno.csv', encoding='ISO-8859-1', sep='\t')
    elif dataset_name == "gerlach":
        try:
            return pd.read_csv('path_to-gerlach.csv', sep=',')
        except UnicodeDecodeError:
            return pd.read_csv('path_to-gerlach.csv', encoding='ISO-8859-1', sep=',')
    elif dataset_name == "fischer":
        try:
            return pd.read_csv('path_to-fischer.csv', sep=',')
        except UnicodeDecodeError:
            return pd.read_csv('path_to-fischer.csv', encoding='ISO-8859-1', sep=',')
    elif dataset_name == "thiebes":
        try:
            return pd.read_csv('path_to-thiebes.csv', sep=',')
        except UnicodeDecodeError:
            return pd.read_csv('path_to-thiebes.csv', encoding='ISO-8859-1', sep=',')
    elif dataset_name == "schmidt-kraeplin":
        try:
            return pd.read_csv('path_to-schmidt-kraeplin.csv', sep=',')
        except UnicodeDecodeError:
            return pd.read_csv('path_to-schmidt-kraeplin.csv', encoding='ISO-8859-1', sep=',')
    elif dataset_name == "maas":
        try:
            return pd.read_csv('path_to-maas.csv', sep=',')
        except UnicodeDecodeError:
            return pd.read_csv('path_to-maas.csv', encoding='ISO-8859-1', sep=',')
    elif dataset_name == "muller":
        try:
            return pd.read_csv('path_to-muller.csv', sep=',')
        except UnicodeDecodeError:
            return pd.read_csv('path_to-muller.csv', encoding='ISO-8859-1', sep=',')    

# add more datasets as needed