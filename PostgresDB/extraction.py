import csv
import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm

tqdm.pandas()

# Database connection parameters
db_params = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': '5432'  # Default port for PostgreSQL
}

# Create the connection string
connection_string = f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"

# Create the SQLAlchemy engine for PostgreSQL
pg_engine = create_engine(connection_string)

# **Step 1: Retrieve all unique active ingredient IDs**
ingredients_query = """
SELECT DISTINCT active_ingredient_id
FROM drug_product_ingredients
"""

unique_ingredients_df = pd.read_sql_query(ingredients_query, pg_engine)
unique_ingredients = unique_ingredients_df['active_ingredient_id'].tolist()
# Prefix the ingredient IDs to create column names
ingredient_columns = [f"ingredient_{col}" for col in unique_ingredients]

target_reaction = 'Fall'

# SQL query
query = f"""
SELECT reports.report_id, drug_product_ingredients.active_ingredient_id
FROM reports 
INNER JOIN report_drug ON reports.report_id = report_drug.report_id
INNER JOIN drug_product ON report_drug.drug_product_id = drug_product.drug_product_id
INNER JOIN drug_product_ingredients ON drug_product.drug_product_id = drug_product_ingredients.drug_product_id
WHERE NOT EXISTS ( SELECT 1 FROM reactions AS r WHERE r.report_id = reports.report_no AND r.pt_name_eng = '{target_reaction}' )
ORDER BY reports.report_id
"""

query_count = f"""
SELECT COUNT(*)
FROM reports 
INNER JOIN report_drug ON reports.report_id = report_drug.report_id
INNER JOIN drug_product ON report_drug.drug_product_id = drug_product.drug_product_id
INNER JOIN drug_product_ingredients ON drug_product.drug_product_id = drug_product_ingredients.drug_product_id
WHERE NOT EXISTS ( SELECT 1 FROM reactions AS r WHERE r.report_id = reports.report_no AND r.pt_name_eng = '{target_reaction}' )
"""

print(query_count)

count = pd.read_sql_query(query_count, pg_engine)
print(count)


query_ingredients = f"""
SELECT DISTINCT drug_product_ingredients.active_ingredient_id
FROM reports 
INNER JOIN report_drug ON reports.report_id = report_drug.report_id
INNER JOIN drug_product ON report_drug.drug_product_id = drug_product.drug_product_id
INNER JOIN drug_product_ingredients ON drug_product.drug_product_id = drug_product_ingredients.drug_product_id
WHERE EXISTS ( SELECT 1 FROM reactions AS r WHERE r.report_id = reports.report_no AND r.pt_name_eng = '{target_reaction}' )
"""

unique_ingredients_df = pd.read_sql_query(query_ingredients, pg_engine)

# Function to write accumulated rows in batch to CSV

ingredient_columns = set(ingredient_columns)
chunksize = 100000
ingredients = {}
target_value = 0
count_reports = 0
done = False

with open('output-nofall.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['report_id'] + list(ingredient_columns) + ['target'])

    for chunk in tqdm(pd.read_sql_query(query, pg_engine, chunksize=chunksize), total=count['count'][0] // chunksize + 1):
        grouped = chunk.groupby('report_id')['active_ingredient_id'].apply(set)
        for report_id, active_ingredient_ids in grouped.items():
            ingredients[report_id] = active_ingredient_ids

            # Write the data when more than 2 reports are collected
            if len(ingredients) > 1:
                report_to_write = next(iter(ingredients.keys()))
                ingredient_ids = ingredients.pop(report_to_write)
                row = [report_to_write] + ['1' if float(f"{ing.removeprefix('ingredient_')}") in ingredient_ids else '0' for ing in ingredient_columns] + [target_value]
                writer.writerow(row)
                count_reports += 1
                if count_reports > 20000:
                    done = True
                    break
        if done:
            break
    
    if not done:
        for report_id, ingredient_ids in ingredients.items():
            row = [report_id] + ['1' if float(f"{ing.removeprefix('ingredient_')}") in ingredient_ids else '0' for ing in ingredient_columns] + [target_value]
            writer.writerow(row)
            count_reports += 1
            if count_reports > 20000:
                break

    