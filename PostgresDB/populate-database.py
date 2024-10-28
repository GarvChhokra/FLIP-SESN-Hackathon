import psycopg2
import csv
from io import StringIO
from datetime import datetime
from tqdm import tqdm

def retry_batch_insert(conn, cursor, table_name, columns, records):
    successful_records = []
    for record in records:
        try:
            buffer = StringIO()
            buffer.write(record + '\n')
            buffer.seek(0)
            cursor.copy_from(buffer, table_name, null='\\N', columns=columns)
            conn.commit()
            successful_records.append(record)
        except Exception as e:
            conn.rollback()
            # Record failed; continue to next
            # Optionally, print or log the error
            print(f"Failed to insert record: {record}. Error: {e}")
    return successful_records

def convert_date(date_str):
    # Same as defined earlier
    if not date_str or date_str == '':
        return '\\N'  # Represent NULL in COPY
    try:
        # Attempt to parse date in multiple formats if needed
        for fmt in ('%d-%b-%y', '%Y-%m-%d'):
            try:
                date_obj = datetime.strptime(date_str.strip(), fmt)
                return date_obj.strftime('%Y-%m-%d')
            except ValueError:
                continue
        return '\\N'  # Return NULL if no format matches
    except Exception:
        return '\\N'

def process_and_insert_data(conn, file_path, table_name, columns, data_types):
    cursor = conn.cursor()
    batch_size = 100000  # Adjust based on memory capacity
    buffer = StringIO()
    count = 0
    total_inserted = 0
    failed_records = []

    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='$', quotechar='"')
        for row_num, row in enumerate(reader, start=1):
            # Skip empty lines
            if not row:
                continue
            report_no_substitution = ""
            e2b_imp_safetyreport_id_substitution = ""
            # Process each field
            processed_row = []
            for idx, value in enumerate(row):
                # Handle missing columns
                if idx >= len(data_types):
                    continue

                data_type = data_types[idx]

                if value == '':
                    processed_value = '\\N'  # Represent NULL
                else:
                    value = value.strip()
                    try:
                        if data_type == 'int':
                            processed_value = int(value)
                        elif data_type == 'float':
                            processed_value = float(value)
                        elif data_type == 'date':
                            processed_value = convert_date(value)
                        elif data_type == 'int-to-varchar':
                            processed_value = str(int(value)).zfill(9)
                        elif data_type == 'report_no':
                            if report_no_substitution != "" and value.startswith('E2B'):
                                processed_value = report_no_substitution
                                e2b_imp_safetyreport_id_substitution = value
                            else:
                                processed_value = value
                        elif data_type == 'e2b':
                            if e2b_imp_safetyreport_id_substitution != "":
                                processed_value = e2b_imp_safetyreport_id_substitution
                            else:
                                processed_value = value
                        elif data_type == 'report_id':
                            if len(value) > 6:
                                report_no_substitution = value
                            processed_value = int(value)
                        else:  # Assume string
                            processed_value = value.replace('\t', ' ').replace('\n', ' ')
                    except ValueError:
                        processed_value = '\\N'  # Handle conversion errors as NULL
                # Ensure strings do not contain tabs or newlines
                if isinstance(processed_value, str) and processed_value != '\\N':
                    processed_value = processed_value.replace('\t', ' ').replace('\n', ' ')

                processed_row.append(str(processed_value))

            # Handle missing columns in the row
            while len(processed_row) < len(columns):
                processed_row.append('\\N')

            # Create the data line for COPY
            data_line = '\t'.join(processed_row) + '\n'
            buffer.write(data_line)
            count += 1

            # Bulk insert when batch size is reached
            if count % batch_size == 0:
                buffer.seek(0)
                try:
                    cursor.copy_from(buffer, table_name, null='\\N', columns=columns)
                    conn.commit()
                    total_inserted += count
                except Exception as e:
                    conn.rollback()
                    print(f"Error during bulk insert into {table_name} at batch ending with record {row_num}: {e}")
                    # Handle batch failure
                    # Re-process the batch to identify and exclude problematic records
                    failed_records_batch = buffer.getvalue().split('\n')[:-1]  # Exclude the last empty split
                    successful_records = retry_batch_insert(conn, cursor, table_name, columns, failed_records_batch)
                    total_inserted += len(successful_records)
                    failed_records.extend([record for record in failed_records_batch if record not in successful_records])
                buffer.close()
                buffer = StringIO()
                count = 0  # Reset count after batch processing

        # Insert any remaining data
        if buffer.tell() > 0:
            buffer.seek(0)
            try:
                cursor.copy_from(buffer, table_name, null='\\N', columns=columns)
                conn.commit()
                total_inserted += count
            except Exception as e:
                conn.rollback()
                print(f"Error during final bulk insert into {table_name}: {e}")
                # Handle batch failure
                failed_records_batch = buffer.getvalue().split('\n')[:-1]
                successful_records = retry_batch_insert(conn, cursor, table_name, columns, failed_records_batch)
                total_inserted += len(successful_records)
                failed_records.extend([record for record in failed_records_batch if record not in successful_records])
            buffer.close()

    cursor.close()

    # Report the results
    print(f"Total records inserted into {table_name}: {total_inserted}")
    if failed_records:
        print(f"Total failed records in {table_name}: {len(failed_records)}")
        # Optionally, write failed records to a file for further analysis
        with open(f'failed_records_{table_name}.txt', 'w', encoding='utf-8') as failed_file:
            for record in failed_records:
                failed_file.write(record + '\n')

if __name__ == '__main__':
    # Database connection parameters
    conn = psycopg2.connect(dbname="postgres", user="postgres", password="postgres", host="localhost")

    # Process each table
    reports_columns = [
    'report_id',
    'report_no',
    'version_no',
    'datreceived',
    'datintreceived',
    'mah_no',
    'report_type_code',
    'report_type_eng',
    'report_type_fr',
    'gender_code',
    'gender_eng',
    'gender_fr',
    'age',
    'age_y',
    'age_unit_eng',
    'age_unit_fr',
    'outcome_code',
    'outcome_eng',
    'outcome_fr',
    'weight',
    'weight_unit_eng',
    'weight_unit_fr',
    'height',
    'height_unit_eng',
    'height_unit_fr',
    'seriousness_code',
    'seriousness_eng',
    'seriousness_fr',
    'death',
    'disability',
    'congenital_anomaly',
    'life_threatening',
    'hosp_required',
    'other_medically_imp_cond',
    'reporter_type_eng',
    'reporter_type_fr',
    'source_code',
    'source_eng',
    'source_fr',
    'e2b_imp_safetyreport_id',
    'authority_numb',
    'company_numb'
]
    reports_types = [
    'report_id',      # REPORT_ID
    'report_no',   # REPORT_NO
    'int',      # VERSION_NO
    'date',     # DATRECEIVED
    'date',     # DATINTRECEIVED
    'string',   # MAH_NO
    'string',   # REPORT_TYPE_CODE
    'string',   # REPORT_TYPE_ENG
    'string',   # REPORT_TYPE_FR
    'string',   # GENDER_CODE
    'string',   # GENDER_ENG
    'string',   # GENDER_FR
    'float',    # AGE
    'float',    # AGE_Y
    'string',   # AGE_UNIT_ENG
    'string',   # AGE_UNIT_FR
    'string',   # OUTCOME_CODE
    'string',   # OUTCOME_ENG
    'string',   # OUTCOME_FR
    'float',    # WEIGHT
    'string',   # WEIGHT_UNIT_ENG
    'string',   # WEIGHT_UNIT_FR
    'float',    # HEIGHT
    'string',   # HEIGHT_UNIT_ENG
    'string',   # HEIGHT_UNIT_FR
    'string',   # SERIOUSNESS_CODE
    'string',   # SERIOUSNESS_ENG
    'string',   # SERIOUSNESS_FR
    'string',   # DEATH
    'string',   # DISABILITY
    'string',   # CONGENITAL_ANOMALY
    'string',   # LIFE_THREATENING
    'string',   # HOSP_REQUIRED
    'string',   # OTHER_MEDICALLY_IMP_COND
    'string',   # REPORTER_TYPE_ENG
    'string',   # REPORTER_TYPE_FR
    'string',   # SOURCE_CODE
    'string',   # SOURCE_ENG
    'string',   # SOURCE_FR
    'e2b',   # E2B_IMP_SAFETYREPORT_ID
    'string',   # AUTHORITY_NUMB
    'string'    # COMPANY_NUMB
]
    
    process_and_insert_data(
        conn=conn,
        file_path='C:/cvponline_extract_20240630/reports.txt',
        table_name='reports',
        columns=reports_columns,
        data_types=reports_types
    )

    # A. Drug_Product
    drug_product_columns = ['drug_product_id', 'drugname']
    drug_product_types = ['int', 'string']
    process_and_insert_data(
        conn=conn,
        file_path='C:/cvponline_extract_20240630/drug_products.txt',
        table_name='drug_product',
        columns=drug_product_columns,
        data_types=drug_product_types
    )

    # B. Drug_Product_Ingredients
    drug_product_ingredients_columns = [
        'drug_product_ingredient_id',
        'drug_product_id',
        'drugname',
        'active_ingredient_id',
        'active_ingredient_name'
    ]
    drug_product_ingredients_types = ['int', 'int', 'string', 'int', 'string']
    process_and_insert_data(
        conn=conn,
        file_path='C:/cvponline_extract_20240630/drug_product_ingredients.txt',
        table_name='drug_product_ingredients',
        columns=drug_product_ingredients_columns,
        data_types=drug_product_ingredients_types
    )

    # C. Reactions
    reactions_columns = [
        'reaction_id',
        'report_id',
        'duration',
        'duration_unit_eng',
        'duration_unit_fr',
        'pt_name_eng',
        'pt_name_fr',
        'soc_name_eng',
        'soc_name_fr',
        'meddra_version'
    ]
    reactions_types = ['int', 'int-to-varchar', 'float', 'string', 'string', 'string', 'string', 'string', 'string', 'string']
    process_and_insert_data(
        conn=conn,
        file_path='C:/cvponline_extract_20240630/reactions.txt',
        table_name='reactions',
        columns=reactions_columns,
        data_types=reactions_types
    )

    # D. Outcome_LX
    outcome_lx_columns = ['outcome_lx_id', 'outcome_code', 'outcome_en', 'outcome_fr']
    outcome_lx_types = ['int', 'string', 'string', 'string']
    process_and_insert_data(
        conn=conn,
        file_path='C:/cvponline_extract_20240630/outcome_lx.txt',
        table_name='outcome_lx',
        columns=outcome_lx_columns,
        data_types=outcome_lx_types
    )

    # E. Gender_LX
    gender_lx_columns = ['gender_lx_id', 'gender_code', 'gender_en', 'gender_fr']
    gender_lx_types = ['int', 'string', 'string', 'string']
    process_and_insert_data(
        conn=conn,
        file_path='C:/cvponline_extract_20240630/gender_lx.txt',
        table_name='gender_lx',
        columns=gender_lx_columns,
        data_types=gender_lx_types
    )

    # F. Report_Type_LX
    report_type_lx_columns = ['report_type_lx_id', 'report_type_code', 'report_type_en', 'report_type_fr']
    report_type_lx_types = ['int', 'string', 'string', 'string']
    process_and_insert_data(
        conn=conn,
        file_path='C:/cvponline_extract_20240630/report_type_lx.txt',
        table_name='report_type_lx',
        columns=report_type_lx_columns,
        data_types=report_type_lx_types
    )

    # G. Seriousness_LX
    seriousness_lx_columns = ['seriousness_lx_id', 'seriousness_code', 'seriousness_en', 'seriousness_fr']
    seriousness_lx_types = ['int', 'string', 'string', 'string']
    process_and_insert_data(
        conn=conn,
        file_path='C:/cvponline_extract_20240630/seriousness_lx.txt',
        table_name='seriousness_lx',
        columns=seriousness_lx_columns,
        data_types=seriousness_lx_types
    )

    # H. Source_LX
    source_lx_columns = ['source_lx_id', 'source_code', 'source_en', 'source_fr']
    source_lx_types = ['int', 'string', 'string', 'string']
    process_and_insert_data(
        conn=conn,
        file_path='C:/cvponline_extract_20240630/source_lx.txt',
        table_name='source_lx',
        columns=source_lx_columns,
        data_types=source_lx_types
    )

    # I. Report_Links_LX
    # DOESN"T EXIST IN THE DATA
    # report_links_lx_columns = ['report_link_id', 'report_id', 'record_type_eng', 'record_type_fr', 'report_link_no']
    # report_links_lx_types = ['int', 'int', 'string', 'string', 'string']
    # process_and_insert_data(
    #     conn=conn,
    #     file_path='C:/cvponline_extract_20240630/report_links_lx.txt',
    #     table_name='report_links_lx',
    #     columns=report_links_lx_columns,
    #     data_types=report_links_lx_types
    # )

    # J. Report_Drug
    report_drug_columns = [
        'report_drug_id',
        'report_id',
        'drug_product_id',
        'drugname',
        'druginvolv_eng',
        'druginvolv_fr',
        'routeadmin_eng',
        'routeadmin_fr',
        'unit_dose_qty',
        'dose_unit_eng',
        'dose_unit_fr',
        'frequency',
        'freq_time',
        'frequency_time_eng',
        'frequency_time_fr',
        'freq_time_unit_eng',
        'freq_time_unit_fr',
        'therapy_duration',
        'therapy_duration_unit_eng',
        'therapy_duration_unit_fr',
        'dosageform_eng',
        'dosageform_fr'
    ]
    report_drug_types = [
        'int', 'int', 'int', 'string', 'string', 'string', 'string', 'string',
        'float', 'string', 'string', 'int', 'float', 'string', 'string', 'string', 'string',
        'float', 'string', 'string', 'string', 'string'
    ]
    process_and_insert_data(
        conn=conn,
        file_path='C:/cvponline_extract_20240630/report_drug.txt',
        table_name='report_drug',
        columns=report_drug_columns,
        data_types=report_drug_types
    )

    # K. Report_Drug_Indication
    report_drug_indication_columns = [
        'report_drug_id',
        'report_id',
        'drug_product_id',
        'drugname',
        'indication_name_eng',
        'indication_name_fr'
    ]
    report_drug_indication_types = ['int', 'int', 'int', 'string', 'string', 'string']
    process_and_insert_data(
        conn=conn,
        file_path='C:/cvponline_extract_20240630/report_drug_indication.txt',
        table_name='report_drug_indication',
        columns=report_drug_indication_columns,
        data_types=report_drug_indication_types
    )

    # L. Literature_Reference
    literature_reference_columns = [
        'report_id',
        'report_no',
        'version_no',
        'seq_literature',
        'literature_reference'
    ]
    literature_reference_types = ['int', 'string', 'int', 'int', 'string']
    process_and_insert_data(
        conn=conn,
        file_path='C:/cvponline_extract_20240630/literature_reference.txt',
        table_name='literature_reference',
        columns=literature_reference_columns,
        data_types=literature_reference_types
    )

    # Close the connection
    conn.close()

    print("Data import for all tables completed successfully.")
