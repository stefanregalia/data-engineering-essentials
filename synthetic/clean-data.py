import os
import duckdb

input_file = "https://s3.amazonaws.com/uvasds-systems/data/synthdata.parquet"

def clean_parquet():
    con = None
    try:
        # Connect to local DuckDB
        con = duckdb.connect(database='synthdata.duckdb', read_only=False)

        # Clear and import
        con.execute(f"""
            -- SQL goes here
            DROP TABLE IF EXISTS synthdata;
            CREATE TABLE synthdata
                AS
            SELECT * FROM read_parquet('{input_file}');
        """)

        """ More cleaning steps
        1. Add age column and populate it (see transform/README.md)
        2. Delete rows with NULL for the 'score' column
        3. Deduplicate the data set (see transform/CLEANING.md)
        4. Max age? Min age? How many over 100?
        5. How many records are left?
        """


        con.execute(f"""
        ALTER TABLE synthdata 
        ADD COLUMN age INTEGER;
        UPDATE synthdata 
        SET age = date_diff('year', birth_date, CURRENT_DATE);
        """)

        con.execute

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    clean_parquet()

