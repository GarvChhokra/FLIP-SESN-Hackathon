# SESN Hackathon - Fall Prevention via Predictive ML Models
This project contains two main parts - fall prediction using the Canada Vigilance Adverse Reaction dataset and gait analysis.

## 1. Canada Vigilance Adverse Reaction Data Extraction, Database Population, and Model Training

## Overview

This project provides Python scripts for extracting and processing data from the [Canada Vigilance Adverse Reaction Database](https://www.canada.ca/en/health-canada/services/drugs-health-products/medeffect-canada/adverse-reaction-database/canada-vigilance-online-database-data-extract.html). 

### Project Contents:
1. **extraction.py**: Extracts fall and not-fall data into CSV files for model training.
2. **populate-database.py**: Populates a PostgreSQL database with the Canada Vigilance data.
3. **init_db.sql**: SQL script to set up the PostgreSQL schema for the database.
4. **model.py**: A machine learning model training script that uses the extracted data.
5. **Dockerfile**: A Dockerfile to set up a PostgreSQL container and initialize the database schema.

---

## Workflow

### 1. Set Up PostgreSQL with Docker

The easiest way to set up PostgreSQL is using Docker. The `Dockerfile` provided in this project helps you create a PostgreSQL database and automatically initializes it with the required schema.

#### Step 1: Build the Docker Image

In the project directory where the `Dockerfile` is located, run the following command to build the Docker image:

```bash
docker build -t sesn_hack_pg_db .
```

#### Step 2: Run the Docker Container

Once the image is built, run the container to start the PostgreSQL database:

```bash
docker run -d --name sesn_hack_pg_container -p 5432:5432 sesn_hack_pg_db
```

This starts the PostgreSQL container and exposes it on port 5432. The container is now running the database with the schema created from the `init_db.sql` file.

#### Step 3: Connect to the PostgreSQL Database

Once the container is running, connect to the PostgreSQL instance using the `psql` client:

```bash
psql -h localhost -U postgres -d CanadaVigilanceAdverseReaction
```

You will be prompted for the password, which is `postgres` by default (as set in the `Dockerfile`).

### 2. Populate the Database

Once the PostgreSQL database is set up and running, the next step is to populate it with data from the Canada Vigilance database.

#### Step 1: Download and Unzip the Data

Download the data files from [Canada Vigilance Online Database](https://www.canada.ca/en/health-canada/services/drugs-health-products/medeffect-canada/adverse-reaction-database/canada-vigilance-online-database-data-extract.html). After downloading, unzip the data files and ensure they are in the same directory as the `populate-database.py` script.

#### Step 2: Run `populate-database.py`

Run the `populate-database.py` script to populate the PostgreSQL database with the unzipped data files:

```bash
python populate-database.py
```

This will read the data files and insert them into the database.

### 3. Extract Training Data

Once the database is populated, you can extract the training data for the machine learning model.

#### Step 1: Run `extraction.py`

The `extraction.py` script extracts two sets of data: one for fall-related events and one for non-fall-related events.

- **Not-Fall Data Extraction**: This is configured by default. Run the script as-is to extract data where no fall occurred:
  ```bash
  python extraction.py
  ```

- **Fall Data Extraction**: To extract fall-related data, modify the first two queries in `extraction.py` to use `EXISTS` instead of `NOT EXISTS` and set `target = 1`. After that, run the script:
  ```bash
  python extraction.py
  ```

The script will output two CSV files:
- **output-fall.csv**: Contains data for fall-related incidents.
- **output-nofall.csv**: Contains data for non-fall-related incidents.

### 4. Train the Machine Learning Model

Once the training data is extracted, you can use the `model.py` script to train a machine learning model on the data.

#### Step 1: Run `model.py`

Ensure that the `output-fall.csv` and `output-nofall.csv` files are available in the project directory. Then, run the `model.py` script to train a machine learning model:

```bash
python model.py
```

This script serves as an example of how to load the extracted data and train a model on it.

---

## File Descriptions

- **extraction.py**: A script to extract data from the PostgreSQL database and output it into CSV files.
- **populate-database.py**: A script to populate the PostgreSQL database with the provided Canada Vigilance data files.
- **init_db.sql**: A SQL file that defines the schema for the PostgreSQL database.
- **model.py**: An example machine learning script that trains a model on the extracted data.
- **Dockerfile**: A Dockerfile that sets up a PostgreSQL container and initializes the schema.

---

## Docker Usage Summary

1. **Build Docker Image**: `docker build -t sesn_hack_pg_db .`
2. **Run Docker Container**: `docker run -d --name sesn_hack_pg_container -p 5432:5432 sesn_hack_pg_db`
3. **Connect to Database**: `psql -h localhost -U postgres -d CanadaVigilanceAdverseReaction`
4. **Populate Database**: `python populate-database.py`
5. **Extract Data**: `python extraction.py`
6. **Train Model**: `python model.py`



### Notes:

- **Docker Setup**: Ensure Docker Desktop is running on your machine before building and running the container.
- **PostgreSQL Password**: The default password for PostgreSQL is set to `postgres` in the `Dockerfile`. You can modify it as needed.

---
