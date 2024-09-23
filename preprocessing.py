import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_data(dataset_name, df, return_type='categorical_pandas_df'):
    """
    Preprocess the data based on the dataset name and return the specified format.
    
    Parameters:
        dataset_name (str): The name of the dataset.
        df (pandas.DataFrame): The raw dataset.
        return_type (str): The format to return, options are 'categorical_pandas_df', 
                           'categorical_numpy', 'onehot_encoded_pandas_df', 'onehot_encoded_numpy'.
    
    Returns:
        Preprocessed data in the requested format.
    """
    if dataset_name == "zoo":
        return preprocess_zoo_data(df, return_type)
    elif dataset_name == "soybean-small":
        return preprocess_soybean_data(df, return_type)
    elif dataset_name == "torno":
        return preprocess_torno_data(df, return_type)
    elif dataset_name == "gerlach":
        return preprocess_gerlach_data(df, return_type)
    elif dataset_name == "fischer":
        return preprocess_fischer_data(df, return_type)
    elif dataset_name == "thiebes":
        return preprocess_thiebes_data(df, return_type)
    elif dataset_name == "schmidt-kraeplin":
        return preprocess_schmidt_kraeplin_data(df, return_type)
    elif dataset_name == "maas":
        return preprocess_maas_data(df, return_type)
    elif dataset_name == "muller":
        return preprocess_muller_data(df, return_type)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

def preprocess_zoo_data(df, return_type='categorical_pandas_df'):
    """
    Preprocess zoo dataset: one-hot encode specific columns, drop unnecessary columns,
    and return in the requested format.
    """
    columns_to_encode = ['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 
                         'predator', 'toothed', 'backbone', 'breathes', 'venomous', 
                         'fins', 'legs', 'tail', 'domestic', 'catsize']
    
    if return_type == 'categorical_pandas_df':
        return df.drop(['animal_name', 'class_type'], axis=1)

    if return_type == 'categorical_numpy':
        df_cat = df.drop(['animal_name', 'class_type'], axis=1)
        le = LabelEncoder()
        for col in df_cat.select_dtypes(include=['object']).columns:
            df_cat[col] = le.fit_transform(df_cat[col])
        return df_cat.to_numpy()

    # One-hot encode for 'onehot_encoded_pandas_df' and 'onehot_encoded_numpy'
    df_encoded = pd.get_dummies(df, columns=columns_to_encode, dtype=int)
    df_encoded = df_encoded.drop(['animal_name', 'class_type'], axis=1)

    if return_type == 'onehot_encoded_pandas_df':
        return df_encoded
    elif return_type == 'onehot_encoded_numpy':
        return df_encoded.to_numpy()

    raise ValueError(f"Unknown return_type {return_type}")

def preprocess_soybean_data(df, return_type='categorical_pandas_df'):
    """
    Preprocess soybean-small dataset: one-hot encode specific columns, drop unnecessary columns,
    and return in the requested format.
    """
    columns_to_encode = ['date', 'plant.stand', 'precip', 'temp', 'hail', 'crop.hist',
                         'area.dam', 'sever', 'seed.tmt', 'germ', 'plant.growth', 
                         'leaves', 'leaf.halo', 'leaf.marg', 'leaf.size', 'leaf.shread', 
                         'leaf.malf', 'leaf.mild', 'stem', 'lodging', 'stem.cankers', 
                         'canker.lesion', 'fruiting.bodies', 'ext.decay', 'mycelium', 
                         'int.discolor', 'sclerotia', 'fruit.pods', 'fruit.spots', 
                         'seed', 'mold.growth', 'seed.discolor', 'seed.size', 'shriveling', 
                         'roots']

    if return_type == 'categorical_pandas_df':
        df.drop(columns=[''], errors='ignore')
        return df.drop(['Class'], axis=1)

    if return_type == 'categorical_numpy':
        df_cat = df.drop(['Class'], axis=1)
        le = LabelEncoder()
        for col in df_cat.select_dtypes(include=['object']).columns:
            df_cat[col] = le.fit_transform(df_cat[col])
        return df_cat.to_numpy()

    df_encoded = pd.get_dummies(df, columns=columns_to_encode, dtype=int)
    df_encoded = df_encoded.drop(['Class'], axis=1)

    if return_type == 'onehot_encoded_pandas_df':
        return df_encoded
    elif return_type == 'onehot_encoded_numpy':
        return df_encoded.to_numpy()

    raise ValueError(f"Unknown return_type {return_type}")

def decode_one_hot(df, prefix):
    """
    Helper function to decode one-hot encoded columns into categorical values.
    """
    relevant_columns = [col for col in df.columns if col.startswith(prefix)]
    return df[relevant_columns].idxmax(axis=1).str.replace(prefix + '.', '')


def preprocess_torno_data(df, return_type='onehot_encoded_pandas_df'):
    """
    Preprocess the torno dataset, supporting one-hot encoding and decoding back to categorical data.
    """
    selected_cols = ['Cost.Free', 'Cost.In-app', 'Cost.Sub', 'Cost.Invest', 'Cost.Premium', 'Cost.Hybrid',
                     'Transaction.None', 'Transaction.Manual', 'Transaction.Assist', 'Internet.Mandatory',
                     'Internet.Periodically', 'Internet.Offline', 'Flows.To-User', 'Flows.From-User', 'Flows.Uni',
                     'DataTransfer.App', 'DataTransfer.bank', 'DataTransfer.Other', 'DataTransfer.bank+Other', 
                     'Account.Yes', 'Account.No', 'Advice.Hybrid', 'Advice.Auto', 'Advice.Human', 'Advice.None', 
                     'Credit.Offer', 'Credit.Monitor', 'Credit.None', 'Budget.Manual', 'Budget.Auto', 'Budget.Man+Auto', 
                     'Budget.None', 'Money.Bank', 'Money.Retail', 'Money.Hybrid', 'Money.Legit', 'Money.None', 
                     'Invest.Trad', 'Invest.Other', 'Invest.Trad+Other', 'Invest.Monitor', 'Invest.None', 
                     'Inform.Not-Individual', 'Inform.Individual', 'Inform.Hybrid+Edu', 'Inform.None']

    if return_type == 'onehot_encoded_pandas_df':
        df.drop(columns=[''], errors='ignore')
        return df[selected_cols]
    
    if return_type == 'onehot_encoded_numpy':
        return df[selected_cols].to_numpy()

    if return_type == 'categorical_pandas_df':
        column_groups = ['Cost', 'Transaction', 'Internet', 'Flows', 'DataTransfer', 'Account', 'Advice', 
                         'Credit', 'Budget', 'Money', 'Invest', 'Inform']

        df_cat = pd.DataFrame()
        for group in column_groups:
            df_cat[group] = decode_one_hot(df, group)
        return df_cat

    if return_type == 'categorical_numpy':
        column_groups = ['Cost', 'Transaction', 'Internet', 'Flows', 'DataTransfer', 'Account', 'Advice', 
                         'Credit', 'Budget', 'Money', 'Invest', 'Inform']

        df_cat = pd.DataFrame()
        for group in column_groups:
            df_cat[group] = decode_one_hot(df, group)

        le = LabelEncoder()
        for col in df_cat.select_dtypes(include=['object']).columns:
            df_cat[col] = le.fit_transform(df_cat[col])

        return df_cat.to_numpy()

    raise ValueError(f"Unknown return_type {return_type}")

def preprocess_maas_data(df, return_type='onehot_encoded_pandas_df'):
    selected_cols = ['Process.with-Micro-Hub', 'Process.without-Micro-Hub', 'Transshipment.no-transshipment', 
                     'Transshipment.vehicle-to-hub-transshipment', 'Transshipment.vehicle-to-human-transshipment', 
                     'Transshipment.vehicle-to-vehicle-transshipment', 'Num_vehicles.1', 'Num_vehicles.2', 
                     'Use_case.Groceries', 'Use_case.Mail-', 'Use_case.Medical-Products-&-Pharmacy', 'Use_case.Packages', 
                     'Use_case.ready-to-eat-meals', 'Range.1', 'Range.2', 'Range.3', 'Vehicle.Areal-drone', 
                     'Vehicle.Autonomous-delivery-van', 'Vehicle.Autonomous-truck', 'Vehicle.Delivery-robot', 
                     'Pyload.1', 'Pyload.2', 'Pyload.3', 'Intelligence_level.1', 'Intelligence_level.2', 
                     'Delivery_option.Predefined-delivery-Windows', 'Delivery_option.on-Demand', 'Price_sensitivity.1', 
                     'Price_sensitivity.2', 'Price_sensitivity.3', 'Effort-expectancy-.1', 'Effort-expectancy-.2', 
                     'Effort-expectancy-.3', 'Hedonic_motivation.1', 'Hedonic_motivation.2', 'Hedonic_motivation.3', 
                     'Hedonic_motivation.4', 'Facilitating-Conditions.1', 'Facilitating-Conditions.2', 
                     'Facilitating-Conditions.3']

    if return_type == 'onehot_encoded_pandas_df':
        return df[selected_cols]
    
    if return_type == 'onehot_encoded_numpy':
        return df[selected_cols].to_numpy()

    if return_type == 'categorical_pandas_df':
        column_groups = ['Process','Transshipment','Num_vehicles','Use_case',
                        'Range','Vehicle','Pyload','Intelligence_level',
                        'Delivery_option','Price_sensitivity','Effort-expectancy-',
                        'Hedonic_motivation','Facilitating-Conditions']

        df_cat = pd.DataFrame()
        for group in column_groups:
            df_cat[group] = decode_one_hot(df, group)
        return df_cat

    if return_type == 'categorical_numpy':
        column_groups = ['Process','Transshipment','Num_vehicles','Use_case',
                        'Range','Vehicle','Pyload','Intelligence_level',
                        'Delivery_option','Price_sensitivity','Effort-expectancy-',
                        'Hedonic_motivation','Facilitating-Conditions']

        df_cat = pd.DataFrame()
        for group in column_groups:
            df_cat[group] = decode_one_hot(df, group)

        le = LabelEncoder()
        for col in df_cat.select_dtypes(include=['object']).columns:
            df_cat[col] = le.fit_transform(df_cat[col])

        return df_cat.to_numpy()

    raise ValueError(f"Unknown return_type {return_type}")




def preprocess_schmidt_kraeplin_data(df, return_type='onehot_encoded_pandas_df'):
    
    selected_cols = ['gamification.direct', 'gamification.mediated', 'user_identity.virtual_character', 
                     'user_identity.self-selected_identity', 'rewards.internal', 'rewards.internal_e1ternal', 
                     'rewards.no', 'competition.direct', 'competition.indirect', 'competition.no', 
                     'target_group.patients', 'target_group.healthy_individuals', 'target_group.health_professionals', 
                     'collaboration.cooperative', 'collaboration.supportive_only', 'collaboration.no', 
                     'goal_setting.self-set', 'goal_setting.e1ternally_set', 'narrative.continuous', 'narrative.episodical', 
                     'reinforcement.positive', 'reinforcement.positive-negative', 'persuasive_intent.compliance_change', 
                     'persuasive_intent.behavior_change', 'persuasive_intent.attitude_change', 
                     'level_of_integration.independent', 'level_of_integration.inherent', 
                     'user_advancement.presentation_only', 'user_advancement.progressive', 'user_advancement.no']

    if return_type == 'onehot_encoded_pandas_df':
        return df[selected_cols]
    
    if return_type == 'onehot_encoded_numpy':
        return df[selected_cols].to_numpy()

    if return_type == 'categorical_pandas_df':
        column_groups = ['gamification', 'user_identity', 'rewards', 'competition', 'target_group', 'collaboration',
                         'goal_setting', 'narrative', 'reinforcement', 'persuasive_intent', 
                         'level_of_integration', 'user_advancement']

        df_cat = pd.DataFrame()
        for group in column_groups:
            df_cat[group] = decode_one_hot(df, group)
        return df_cat

    if return_type == 'categorical_numpy':
        column_groups = ['gamification', 'user_identity', 'rewards', 'competition', 'target_group', 'collaboration',
                         'goal_setting', 'narrative', 'reinforcement', 'persuasive_intent', 
                         'level_of_integration', 'user_advancement']

        df_cat = pd.DataFrame()
        for group in column_groups:
            df_cat[group] = decode_one_hot(df, group)

        le = LabelEncoder()
        for col in df_cat.select_dtypes(include=['object']).columns:
            df_cat[col] = le.fit_transform(df_cat[col])

        return df_cat.to_numpy()

    raise ValueError(f"Unknown return_type {return_type}")

def preprocess_thiebes_data(df, return_type='onehot_encoded_pandas_df'):
  
    selected_cols = ['Business_purpose.For_profit', 'Business_purpose.Non-profit',
                     'Region_of_operation.Local', 'Region_of_operation.Worldwide',
                     'Consumer_target_group.Enthusiasts',
                     'Consumer_target_group.Specific_information_seekers',
                     'Consumer_target_group.Enthusiasts_a_specific_information_seekers',
                     'Consumer_target_group.Chronic_health_issue_risk_group',
                     'Consumer_research_consent.Mandatory',
                     'Consumer_research_consent.Optional',
                     'Consumer_research_consent.Data_not_used',
                     'Distribution_channel.Internet_only',
                     'Distribution_channel.Health_care_professionals_only',
                     'Distribution_channel.Multi-contact_service',
                     'Sample_site.Home_Collection_Kit', 'Sample_site.Lab_Collection',
                     'Sample_site.Home_Collection_Kit_lab_Collection',
                     'Sampling_kit_provider.Service_provider',
                     'Sampling_kit_provider.Third_party',
                     'Sampling_kit_provider.Service_provider_&_Third_party',
                     'Sample_storage.Never', 'Sample_storage.Mandatory',
                     'Sample_storage.Consumer_decision', 'Genome_test_type.Genotyping',
                     'Genome_test_type.Sequencing',
                     'Genome_test_type.Genotyping_&_sequencing',
                     'Data_storage.No_storage', 'Data_storage.Isolated_storage',
                     'Data_storage.Database_for_service_provider',
                     'Data_ownership.Consumer', 'Data_ownership.Service_provider',
                     'Data_processing.No_interpretation',
                     'Data_processing.Basic_interpretation',
                     'Data_processing.Value_added_interpretation',
                     'Fee_type.Pay-per-use', 'Fee_type.Pay-per-use_&_subscription',
                     'Fee_type.No_fee', 'Fee_payer.Consumer_only',
                     'Fee_payer.Consumer_&_health_insurance',
                     'Reselling_of_genome_data.Yes', 'Reselling_of_genome_data.No']




    if return_type == 'onehot_encoded_pandas_df':
        return df[selected_cols]
    
    if return_type == 'onehot_encoded_numpy':
        return df[selected_cols].to_numpy()

    if return_type == 'categorical_pandas_df':
        column_groups = ['Business_purpose', 'Region_of_operation', 'Consumer_target_group', 'Consumer_research_consent',
                         'Distribution_channel', 'Sample_site', 'Sampling_kit_provider', 'Sample_storage', 'Genome_test_type',
                         'Data_storage', 'Data_ownership', 'Data_processing', 'Fee_type', 'Fee_payer', 'Reselling_of_genome_data']

        df_cat = pd.DataFrame()
        for group in column_groups:
            df_cat[group] = decode_one_hot(df, group)
        return df_cat

    if return_type == 'categorical_numpy':
        column_groups = ['Business_purpose', 'Region_of_operation', 'Consumer_target_group', 'Consumer_research_consent',
                         'Distribution_channel', 'Sample_site', 'Sampling_kit_provider', 'Sample_storage', 'Genome_test_type',
                         'Data_storage', 'Data_ownership', 'Data_processing', 'Fee_type', 'Fee_payer', 'Reselling_of_genome_data']

        df_cat = pd.DataFrame()
        for group in column_groups:
            df_cat[group] = decode_one_hot(df, group)

        le = LabelEncoder()
        for col in df_cat.select_dtypes(include=['object']).columns:
            df_cat[col] = le.fit_transform(df_cat[col])

        return df_cat.to_numpy()

    raise ValueError(f"Unknown return_type {return_type}")




def preprocess_gerlach_data(df, return_type='onehot_encoded_pandas_df'):

    selected_cols = ['c1.2', 'c1.3', 'c1.4', 'c1.5', 'c2.2', 'c2.3', 'c2.4', 'c2.5', 'c3.1', 'c3.2', 'c3.3', 'c3.4',
                     'c4.1', 'c4.2', 'c4.3', 'c4.4', 'c5.1', 'c5.2', 'c6.1', 'c6.2', 'c7.1', 'c7.2', 'c7.3', 'c8.1', 
                     'c8.2', 'c8.3', 'c8.5', 'c9.1', 'c9.2', 'c9.3', 'c9.4', 'c9.5', 'c10.1', 'c10.2', 'c10.3', 'c10.4', 
                     'c10.5', 'c11.1', 'c11.2', 'c11.3', 'c11.4', 'c12.1', 'c12.2', 'c12.3', 'c12.4', 'c13.1', 'c13.2']


    if return_type == 'onehot_encoded_pandas_df':
        return df[selected_cols]
    
    if return_type == 'onehot_encoded_numpy':
        return df[selected_cols].to_numpy()

    if return_type == 'categorical_pandas_df':
        column_groups = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13']



        df_cat = pd.DataFrame()
        for group in column_groups:
            df_cat[group] = decode_one_hot(df, group)
        return df_cat

    if return_type == 'categorical_numpy':
        column_groups = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13']



        df_cat = pd.DataFrame()
        for group in column_groups:
            df_cat[group] = decode_one_hot(df, group)

        le = LabelEncoder()
        for col in df_cat.select_dtypes(include=['object']).columns:
            df_cat[col] = le.fit_transform(df_cat[col])

        return df_cat.to_numpy()

    raise ValueError(f"Unknown return_type {return_type}")



def preprocess_fischer_data(df, return_type='onehot_encoded_pandas_df'):

    selected_cols = ['Configuration.Configurable', 'Configuration.Not_Configurable',
                     'Service_Object.Consumer-centric', 'Service_Object.Peripheral',
                     'Benefit.During_Service_Delivery', 'Benefit.After_Service_Delivery',
                     'Duration.On_Demand', 'Duration.Continuous',
                     'Data_Analytics.Descriptive_Analytics',
                     'Data_Analytics.Predictive_Analytics',
                     'Data_Analytics.Exploratory_Analytics',
                     'Data_Analytics.Prescriptive_Analytics', 'Capability_Level.Networked',
                     'Capability_Level.Enhanced', 'Capability_Level.Aware',
                     'Communication.Consumer_to_Service',
                     'Communication.Service_to_Consumer', 'Communication.Bidirectional',
                     'Data_Source.Consumer', 'Data_Source.Environment']


    if return_type == 'onehot_encoded_pandas_df':
        return df[selected_cols]
    
    if return_type == 'onehot_encoded_numpy':
        return df[selected_cols].to_numpy()

    if return_type == 'categorical_pandas_df':
        column_groups = ['Configuration', 'Service_Object', 'Benefit', 'Duration', 'Data_Analytics',
                         'Capability_Level', 'Communication', 'Data_Source']

        df_cat = pd.DataFrame()
        for group in column_groups:
            df_cat[group] = decode_one_hot(df, group)
        return df_cat

    if return_type == 'categorical_numpy':
        column_groups = ['Configuration', 'Service_Object', 'Benefit', 'Duration', 'Data_Analytics',
                         'Capability_Level', 'Communication', 'Data_Source']

        df_cat = pd.DataFrame()
        for group in column_groups:
            df_cat[group] = decode_one_hot(df, group)

        le = LabelEncoder()
        for col in df_cat.select_dtypes(include=['object']).columns:
            df_cat[col] = le.fit_transform(df_cat[col])

        return df_cat.to_numpy()

    raise ValueError(f"Unknown return_type {return_type}")




def preprocess_muller_data(df, return_type='onehot_encoded_pandas_df'):

    selected_cols = ['1.Psychoeducation', '1.MedicalAssessment', '1.SymptomManagement', '1.SupportiveResources',
                     '1.TherapeuticTreatment', '1.MultiplePurposes', '1.Audioonly', '1.Te1tonly', '1.Visual', 
                     '1.Multimedia', '2.Informational', '2.Reporting', '2.Interactive', 
                     '3.Informationone1ternalsupp.systems&hotlines', '3.personalsupport_emergencycall',
                     '3.integrated"panicmanagement"_safetyplan', '3.None', 'TailoringFeatures.None', 
                     'TailoringFeatures.interfacecostumization', 'TailoringFeatures.treatment-orientedcostumization', 
                     'TailoringFeatures.app-driventailoring', 'TailoringFeatures.mood-drivenpersonalization', 
                     'TailoringFeatures.customizationofnotifications', 'TailoringFeatures.multiple', 
                     'Pricemodel.Free', 'Pricemodel.FreemiumSubscription', 'Pricemodel.FrremiumOnetimepayment', 
                     'Pricemodel.FrremiumHybrid', 'Pricemodel.PremiumOnetimepayment', 
                     'Pricemodel.PremiumSubscriptionerstattungsf�hig', 'Pricemodel.AufRezept', 
                     'Pricemodel.onetimepaymenterstattungsf�hig', 'Typeofapp-usage.independent', 
                     'Typeofapp-usage.therapistcanbecontacted', 'Typeofapp-usage.partoftherapy_accompaniedbytherapist', 
                     'Certified.Yes', 'Certified.No', 'Betriebssystem.Mobile', 'Betriebssystem.Web', 
                     'Betriebssystem.both', 'DataCollection.Yes_onlytechnischeDaten_Zugriffsdaten', 
                     'DataCollection.Yes_Nutzungs-undpersonen-bezogeneDaten', 'DataCollection.No', 
                     'Sharingofinformation.Nosharing_appprovideronly_gesetzlichvorgeschrieben', 
                     'Sharingofinformation.appproviderplusserviceprovider', 
                     'Sharingofinformation.appproviderplustserviceproviderplusotherthirdparties']


    if return_type == 'onehot_encoded_pandas_df':
        return df[selected_cols]
    
    if return_type == 'onehot_encoded_numpy':
        return df[selected_cols].to_numpy()

    if return_type == 'categorical_pandas_df':
        column_groups = ['1', '2', '3', 'TailoringFeatures', 'Pricemodel', 'Typeofapp-usage', 'Certified', 
                         'Betriebssystem', 'DataCollection', 'Sharingofinformation']

        df_cat = pd.DataFrame()
        for group in column_groups:
            df_cat[group] = decode_one_hot(df, group)
        return df_cat

    if return_type == 'categorical_numpy':
        column_groups = ['1', '2', '3', 'TailoringFeatures', 'Pricemodel', 'Typeofapp-usage', 'Certified', 
                         'Betriebssystem', 'DataCollection', 'Sharingofinformation']

        df_cat = pd.DataFrame()
        for group in column_groups:
            df_cat[group] = decode_one_hot(df, group)

        le = LabelEncoder()
        for col in df_cat.select_dtypes(include=['object']).columns:
            df_cat[col] = le.fit_transform(df_cat[col])

        return df_cat.to_numpy()

    raise ValueError(f"Unknown return_type {return_type}")


