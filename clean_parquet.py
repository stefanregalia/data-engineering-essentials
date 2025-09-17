import duckdb

input_file = "https://s3.amazonaws.com/uvasds-systems/data/synthdata.parquet"

def clean_parquet():
    con = None
    try:
        # Connect to local DuckDB
        con = duckdb.connect(database='synthdata.duckdb', read_only=False)

        # Drop/recreate the table
        con.execute(f"""
            DROP TABLE IF EXISTS synthdata;
            CREATE TABLE synthdata AS
            SELECT * FROM read_parquet('{input_file}');
        """)

        # Add and populate age column
        con.execute("""
            ALTER TABLE synthdata 
            ADD COLUMN age INTEGER;
        """)
        con.execute("""
            UPDATE synthdata 
            SET age = datediff('year', CAST(birth_date AS DATE), current_date);
        """)

        # Drop duplicates
        con.execute("""
            CREATE OR REPLACE TABLE synthdata AS
            SELECT DISTINCT * FROM synthdata;
        """)

        # Remove null scores
        con.execute("""
            DELETE FROM synthdata
            WHERE score IS NULL;
        """)

        # Get age stats
        stats = con.execute("""
            SELECT 
                MIN(age) AS min_age,
                MAX(age) AS max_age,
                SUM(CASE WHEN age > 100 THEN 1 ELSE 0 END) AS over_100
            FROM synthdata;
        """).fetchdf()
        print("Age stats:\n", stats)

        # Get final record count
        final_count = con.execute("SELECT COUNT(*) FROM synthdata").fetchone()[0]
        print(f"Final number of records: {final_count}")

        # Show a preview
        preview = con.execute("SELECT birth_date, age FROM synthdata LIMIT 5").fetchdf()
        print("Preview:\n", preview)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if con:
            con.close()

if __name__ == "__main__":
    clean_parquet()
