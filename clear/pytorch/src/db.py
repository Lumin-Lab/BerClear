from sqlalchemy import create_engine
import psycopg2
import pandas as pd
import os
from .utils import convert_column_names

def fetch_user_ber_record(dbname, user, password, host, port, schema, berID, mapping):
    
    # Connect to the PostgreSQL database
    conn = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port
    )

    # Create a cursor object
    cur = conn.cursor()

    # Execute a SQL query to fetch the row
    cur.execute(f"SELECT * FROM {schema}.ber OFFSET {berID}-1 LIMIT 1;")

    # Fetch the result
    row = cur.fetchone()

    # Close the cursor and connection
    cur.close()
    conn.close()

    # If a row was fetched
    if row:
        # Get column names from the database table
        column_names = [desc[0] for desc in cur.description]
        
        # Create a DataFrame
        df = pd.DataFrame([row], columns=column_names)

        df = convert_column_names(mapping, df)

    else:
        raise FileNotFoundError("No row fetched.")

    return df


def load_df_from_db():
    POSTGRES_USER = os.environ.get('POSTGRES_USER')
    POSTGRES_PASSWORD = os.environ.get('POSTGRES_PASSWORD')
    POSTGRES_DB = os.environ.get('POSTGRES_DB')
    POSTGRES_PORT = os.environ.get('POSTGRES_PORT')
    POSTGRES_HOST = os.environ.get('POSTGRES_HOST')
    POSTGRES_SCHEMA = os.environ.get('POSTGRES_SCHEMA')
    DATABASE_URL = f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}'

    print("Database_url is ", DATABASE_URL)

    # Create an engine
    engine = create_engine(DATABASE_URL)


    # Connect to the database
    connection = engine.connect()

    data = pd.concat(pd.read_sql(f"SELECT * FROM {POSTGRES_SCHEMA}.ber", connection, chunksize=50000))
    
    return data

