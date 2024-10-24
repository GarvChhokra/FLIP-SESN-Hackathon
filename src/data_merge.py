import pandas as pd
import os

# Set the base path where the datasets are located
base_path = "/Users/user/Desktop/SESN HACKATHON/raw-data"

# Helper function to load data with filtering and dynamic paths
def load_data(filename, column_names, sep=r'$', id_column=None, filter_ids=None):
    file_path = os.path.join(base_path, filename)  # Construct full file path
    try:
        df = pd.read_csv(file_path, sep=sep, header=None, names=column_names)
        if id_column and filter_ids is not None:
            df = df[df[id_column].isin(filter_ids)]  # Filter by matching IDs
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        df = pd.DataFrame(columns=column_names)  # Empty DataFrame on error
    return df

# Load the first 1000 rows from 'reports.txt'
df_report = pd.read_csv(
    os.path.join(base_path, 'reports.txt'), 
    sep=r'$', 
    header=None, 
    names=['REPORT_ID', 'REPORT_NO', 'VERSION_NO', 'DATRECEIVED', 'DATINTRECEIVED', 'MAH_NO', 
           'REPORT_TYPE_CODE', 'REPORT_TYPE_ENG', 'REPORT_TYPE_FR', 'GENDER_CODE', 
           'GENDER_ENG', 'GENDER_FR', 'AGE', 'AGE_Y', 'AGE_UNIT_ENG', 'AGE_UNIT_FR', 
           'OUTCOME_CODE', 'OUTCOME_ENG', 'OUTCOME_FR', 'WEIGHT', 'WEIGHT_UNIT_ENG', 
           'WEIGHT_UNIT_FR', 'HEIGHT', 'HEIGHT_UNIT_ENG', 'HEIGHT_UNIT_FR', 
           'SERIOUSNESS_CODE', 'SERIOUSNESS_ENG', 'SERIOUSNESS_FR', 'DEATH', 
           'DISABILITY', 'CONGENITAL_ANOMALY', 'LIFE_THREATENING', 'HOSP_REQUIRED', 
           'OTHER_MEDICALLY_IMP_COND', 'REPORTER_TYPE_ENG', 'REPORTER_TYPE_FR', 
           'SOURCE_CODE', 'SOURCE_ENG', 'SOURCE_FR', 'E2B_IMP_SAFETYREPORT_ID', 
           'AUTHORITY_NUMB', 'COMPANY_NUMB'],
    nrows=1000
)

# Extract relevant IDs for filtering
report_ids = df_report['REPORT_ID'].unique()

# Load related datasets with filtering
df_report_drug = load_data('report_drug.txt', 
                           ['REPORT_DRUG_ID', 'REPORT_ID', 'DRUG_PRODUCT_ID', 'DRUGNAME', 
                            'DRUGINVOLV_ENG', 'DRUGINVOLV_FR', 'ROUTEADMIN_ENG', 'ROUTEADMIN_FR', 
                            'UNIT_DOSE_QTY', 'DOSE_UNIT_ENG', 'DOSE_UNIT_FR', 'FREQUENCY', 
                            'FREQ_TIME', 'FREQUENCY_TIME_ENG', 'FREQUENCY_TIME_FR', 
                            'FREQ_TIME_UNIT_ENG', 'FREQ_TIME_UNIT_FR', 'THERAPY_DURATION', 
                            'THERAPY_DURATION_UNIT_ENG', 'THERAPY_DURATION_UNIT_FR', 
                            'DOSAGEFORM_ENG', 'DOSAGEFORM_FR'],
                           id_column='REPORT_ID', filter_ids=report_ids)

# Extract relevant DRUG_PRODUCT_IDs for further filtering
drug_product_ids = df_report_drug['DRUG_PRODUCT_ID'].unique()

df_drug = load_data('drug_products.txt', 
                    ['DRUG_PRODUCT_ID', 'DRUGNAME'], 
                    id_column='DRUG_PRODUCT_ID', filter_ids=drug_product_ids)

df_ingredients = load_data('drug_product_ingredients.txt', 
                           ['DRUG_PRODUCT_INGREDIENT_ID', 'DRUG_PRODUCT_ID', 'DRUGNAME', 
                            'ACTIVE_INGREDIENT_ID', 'ACTIVE_INGREDIENT_NAME'], 
                           id_column='DRUG_PRODUCT_ID', filter_ids=drug_product_ids)

df_reaction = load_data('reactions.txt', 
                        ['REACTION_ID', 'REPORT_ID', 'DURATION', 'DURATION_UNIT_ENG', 
                         'DURATION_UNIT_FR', 'PT_NAME_ENG', 'PT_NAME_FR', 'SOC_NAME_ENG', 
                         'SOC_NAME_FR', 'MEDDRA_VERSION'], 
                        id_column='REPORT_ID', filter_ids=report_ids)

outcome_codes = df_report['OUTCOME_CODE'].unique()
df_outcome = load_data('outcome_lx.txt', 
                       ['OUTCOME_LX_ID', 'OUTCOME_CODE', 'OUTCOME_EN', 'OUTCOME_FR'], 
                       id_column='OUTCOME_CODE', filter_ids=outcome_codes)

df_report_drug_indication = load_data('report_drug_indication.txt', 
                                      ['REPORT_DRUG_ID', 'REPORT_ID', 'DRUG_PRODUCT_ID', 'DRUGNAME', 
                                       'INDICATION_NAME_ENG', 'INDICATION_NAME_FR'], 
                                      id_column='REPORT_ID', filter_ids=report_ids)

# Merge DataFrames based on relationships
df_combined = pd.merge(df_report, df_report_drug, on='REPORT_ID', how='left')
df_combined = pd.merge(df_combined, df_drug, on='DRUG_PRODUCT_ID', how='left')
df_combined = pd.merge(df_combined, df_ingredients, on='DRUG_PRODUCT_ID', how='left')
df_combined = pd.merge(df_combined, df_reaction, on='REPORT_ID', how='left')
df_combined = pd.merge(df_combined, df_outcome, on='OUTCOME_CODE', how='left')
df_combined = pd.merge(df_combined, df_report_drug_indication, on=['REPORT_ID', 'DRUG_PRODUCT_ID'], how='left')

# Display the first few rows of the combined DataFrame
print(df_combined.head())
