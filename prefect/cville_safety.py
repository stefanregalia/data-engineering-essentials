import duckdb
from prefect import flow, task

"""
Charlottesville Crime and Arrests Data - updated nightly
https://opendata.charlottesville.org/search?tags=Public%2520Safety

Within a Prefect flow:
1. From ARRESTS data retrieve counts of arrests by SEX for 2025.
2. From CRIME data retrieve TOP FIVE Offenses for the entire record.
"""

arrests_url = "https://s3.amazonaws.com/uvasds-systems/data/cville/Arrests.csv"
crime_url = "https://s3.amazonaws.com/uvasds-systems/data/cville/Crime_data.csv"

@task
def task_one():
  pass

@flow
def run_task():
  task_one()
  
if __name__ == "__main__":
    run_task()
