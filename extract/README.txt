# Extract Data

- `curl`
- `wget`
- `requests`
- `boto3`

Try it out: 

- https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet
- https://s3.amazonaws.com/uvasds-systems/data/SAU-GLOBAL-1-v48-0.csv
- s3://uvasds-systems/data/SAU-GLOBAL-1-v48-0.csv

## View Schema

You can use a tool like DuckDB to view the schema of a data file (CSV, Parquet, Avro, etc.) remotely.

Install the DuckDB command-line tool:

```
curl https://install.duckdb.org | sh
```

Then, use the `DESCRIBE` SQL command within a DuckDB prompt:

```
duckdb
```

```
DESCRIBE SELECT * FROM read_csv('local.csv');
DESCRIBE SELECT * FROM read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet');
```

Which will return a schema:

```
┌───────────────────────┬─────────────┬─────────┬─────────┬─────────┬─────────┐
│      column_name      │ column_type │  null   │   key   │ default │  extra  │
│        varchar        │   varchar   │ varchar │ varchar │ varchar │ varchar │
├───────────────────────┼─────────────┼─────────┼─────────┼─────────┼─────────┤
│ VendorID              │ INTEGER     │ YES     │ NULL    │ NULL    │ NULL    │
│ tpep_pickup_datetime  │ TIMESTAMP   │ YES     │ NULL    │ NULL    │ NULL    │
│ tpep_dropoff_datetime │ TIMESTAMP   │ YES     │ NULL    │ NULL    │ NULL    │
│ passenger_count       │ BIGINT      │ YES     │ NULL    │ NULL    │ NULL    │
│ trip_distance         │ DOUBLE      │ YES     │ NULL    │ NULL    │ NULL    │
│ RatecodeID            │ BIGINT      │ YES     │ NULL    │ NULL    │ NULL    │
│ store_and_fwd_flag    │ VARCHAR     │ YES     │ NULL    │ NULL    │ NULL    │
│ PULocationID          │ INTEGER     │ YES     │ NULL    │ NULL    │ NULL    │
│ DOLocationID          │ INTEGER     │ YES     │ NULL    │ NULL    │ NULL    │
│ payment_type          │ BIGINT      │ YES     │ NULL    │ NULL    │ NULL    │
│ fare_amount           │ DOUBLE      │ YES     │ NULL    │ NULL    │ NULL    │
│ extra                 │ DOUBLE      │ YES     │ NULL    │ NULL    │ NULL    │
│ mta_tax               │ DOUBLE      │ YES     │ NULL    │ NULL    │ NULL    │
│ tip_amount            │ DOUBLE      │ YES     │ NULL    │ NULL    │ NULL    │
│ tolls_amount          │ DOUBLE      │ YES     │ NULL    │ NULL    │ NULL    │
│ improvement_surcharge │ DOUBLE      │ YES     │ NULL    │ NULL    │ NULL    │
│ total_amount          │ DOUBLE      │ YES     │ NULL    │ NULL    │ NULL    │
│ congestion_surcharge  │ DOUBLE      │ YES     │ NULL    │ NULL    │ NULL    │
│ Airport_fee           │ DOUBLE      │ YES     │ NULL    │ NULL    │ NULL    │
│ cbd_congestion_fee    │ DOUBLE      │ YES     │ NULL    │ NULL    │ NULL    │
├───────────────────────┴─────────────┴─────────┴─────────┴─────────┴─────────┤
│ 20 rows                                                           6 columns │
└─────────────────────────────────────────────────────────────────────────────┘
```
