import pandas as pd
import os
from datetime import datetime

# Set the base path where the datasets are located
base_path = "/Users/user/Desktop/SESN HACKATHON/raw-data"

# Custom date parser function
def custom_date_parser(date_series):
    dates = []
    for date_str in date_series:
        try:
            # Parse the date string
            dt = datetime.strptime(date_str.strip('"'), '%d-%b-%y')
            # Adjust years greater than the current year to be in the previous century
            if dt.year > datetime.now().year:
                dt = dt.replace(year=dt.year - 100)
            dates.append(dt)
        except Exception:
            dates.append(pd.NaT)  # Handle parsing errors by setting as NaT
    return dates

# Helper function to load datasets
def load_data(filename, column_names, sep='$', id_column=None, filter_ids=None):
    file_path = os.path.join(base_path, filename)  # Construct full path
    try:
        df = pd.read_csv(
            file_path, 
            sep=sep, 
            header=None, 
            names=column_names, 
            dtype=str,  # Read all data as strings
        )
        if id_column and filter_ids is not None:
            df = df[df[id_column].isin(filter_ids)]  # Filter based on IDs
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        df = pd.DataFrame(columns=column_names)  # Empty DataFrame on error
    return df

# Modified function to load and filter reports
def load_filtered_reports(file_path, date_column, start_date='2022-06-28', sep='$', chunksize=500, max_records=None):
    """
    Load 'reports.txt' and filter rows by date during loading, limiting to a maximum number of records if specified.
    """
    filtered_data = []
    start_date = pd.to_datetime(start_date)  # Convert start_date to datetime for comparison
    total_records = 0  # Keep track of total records loaded

    for chunk in pd.read_csv(
        file_path, 
        sep=sep, 
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
        parse_dates=[date_column],  # Parse the DATRECEIVED column
        date_parser=custom_date_parser,
        chunksize=chunksize,  # Load in chunks
        dtype=str,
        quotechar='"',
        engine='python'
    ):
        # Filter the chunk by the given start date
        chunk[date_column] = pd.to_datetime(chunk[date_column], errors='coerce')
        chunk_filtered = chunk[chunk[date_column] > start_date]
        filtered_data.append(chunk_filtered)
        total_records += len(chunk_filtered)
        if max_records and total_records >= max_records:
            break  # Exit the loop if we've reached the maximum number of records

    # Concatenate all filtered chunks into a single DataFrame
    result = pd.concat(filtered_data, ignore_index=True)
    if max_records:
        result = result.iloc[:max_records]  # Take only the first max_records rows
    return result

# Load 'reports.txt' with filtering during the loading process, limiting to 1000 records
df_report = load_filtered_reports(
    os.path.join(base_path, 'reports.txt'),
    date_column='DATRECEIVED',
    start_date='01-JAN-20',
    
)

# Ensure REPORT_ID and REPORT_NO are treated as strings
df_report['REPORT_ID'] = df_report['REPORT_ID'].astype(str)
df_report['REPORT_NO'] = df_report['REPORT_NO'].astype(str)

# Extract REPORT_IDs and REPORT_NOs for filtering related datasets
report_ids_6_digit = df_report['REPORT_ID'].unique()
report_ids_9_digit = df_report['REPORT_NO'].unique()

# Load and filter related datasets

# Report_Drug.txt uses REPORT_ID (6 digits)
df_report_drug = load_data('report_drug.txt', 
                           ['REPORT_DRUG_ID', 'REPORT_ID', 'DRUG_PRODUCT_ID', 'DRUGNAME', 
                            'DRUGINVOLV_ENG', 'DRUGINVOLV_FR', 'ROUTEADMIN_ENG', 'ROUTEADMIN_FR', 
                            'UNIT_DOSE_QTY', 'DOSE_UNIT_ENG', 'DOSE_UNIT_FR', 'FREQUENCY', 
                            'FREQ_TIME', 'FREQUENCY_TIME_ENG', 'FREQUENCY_TIME_FR', 
                            'FREQ_TIME_UNIT_ENG', 'FREQ_TIME_UNIT_FR', 'THERAPY_DURATION', 
                            'THERAPY_DURATION_UNIT_ENG', 'THERAPY_DURATION_UNIT_FR', 
                            'DOSAGEFORM_ENG', 'DOSAGEFORM_FR'],
                           id_column='REPORT_ID', filter_ids=report_ids_6_digit)

df_report_drug['REPORT_ID'] = df_report_drug['REPORT_ID'].astype(str)
df_report_drug['DRUG_PRODUCT_ID'] = df_report_drug['DRUG_PRODUCT_ID'].astype(str)

# Extract DRUG_PRODUCT_IDs for further filtering
drug_product_ids = df_report_drug['DRUG_PRODUCT_ID'].unique()

# Reactions.txt uses REPORT_ID (9 digits)
df_reaction = load_data('reactions.txt', 
                        ['REACTION_ID', 'REPORT_ID', 'DURATION', 'DURATION_UNIT_ENG', 
                         'DURATION_UNIT_FR', 'PT_NAME_ENG', 'PT_NAME_FR', 'SOC_NAME_ENG', 
                         'SOC_NAME_FR', 'MEDDRA_VERSION'], 
                        id_column='REPORT_ID', filter_ids=report_ids_6_digit)

df_reaction['REPORT_ID'] = df_reaction['REPORT_ID'].astype(str)

# Load ingredients data
df_ingredients = load_data('drug_product_ingredients.txt', 
                           ['DRUG_PRODUCT_INGREDIENT_ID', 'DRUG_PRODUCT_ID', 'DRUGNAME', 
                            'ACTIVE_INGREDIENT_ID', 'ACTIVE_INGREDIENT_NAME'], 
                           id_column='DRUG_PRODUCT_ID', filter_ids=drug_product_ids)

df_ingredients['DRUG_PRODUCT_ID'] = df_ingredients['DRUG_PRODUCT_ID'].astype(str)

# Load outcome data
df_outcome = load_data('outcome_lx.txt', 
                       ['OUTCOME_LX_ID', 'OUTCOME_CODE', 'OUTCOME_EN', 'OUTCOME_FR'], 
                       id_column='OUTCOME_CODE', filter_ids=df_report['OUTCOME_CODE'].unique())

# Merge datasets
# Merge df_report and df_report_drug on 'REPORT_ID' (6 digits)
df_combined = pd.merge(df_report, df_report_drug, on='REPORT_ID', how='inner')

# Merge df_combined and df_reaction on:
df_combined = pd.merge(df_combined, df_reaction, on='REPORT_ID', how='inner', suffixes=('', '_REACTION'))

# Merge df_combined and df_ingredients on 'DRUG_PRODUCT_ID'
df_combined = pd.merge(df_combined, df_ingredients, on='DRUG_PRODUCT_ID', how='inner', suffixes=('', '_INGREDIENT'))

# Merge df_combined and df_outcome on 'OUTCOME_CODE'
df_combined = pd.merge(df_combined, df_outcome, on='OUTCOME_CODE', how='left', suffixes=('', '_OUTCOME'))

# Display merged data
print(df_combined.head())
print(f"Total merged records: {df_combined.shape[0]}")

# Example Analysis: Medications and Their Side Effects
medication_effects = df_combined[['DRUGNAME', 'ACTIVE_INGREDIENT_NAME', 'PT_NAME_ENG', 'DURATION', 'WEIGHT', 'HEIGHT']]
print(medication_effects.head())

# Save the merged data for further analysis
df_combined.to_csv('merged_data.csv', index=False)