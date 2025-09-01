# DuckDB Data Analysis Lab
**Duration:** 45 minutes  
**Level:** Intermediate  
**Prerequisites:** Basic Python knowledge, familiarity with SQL

## Learning Objectives
By the end of this lab, you will be able to:
- Set up and use DuckDB in Python
- Load and query CSV files directly with DuckDB
- Convert between CSV and Parquet formats
- Perform complex analytical queries
- Integrate DuckDB with pandas DataFrames
- Optimize query performance with columnar storage

## Setup (5 minutes)

### Install Required Packages

Install DuckDB in your environment:

- Official [installation](https://duckdb.org/docs/installation/)

Set up a Python virtual environment:

1. Using GitHub Codespace
2. Using `virtualenv`
3. Using `pipenv`

Next install the necessary packages:

```bash
pip install duckdb pandas numpy matplotlib
```

Finally, open an interactive Python session within your virtual environment:

```
python
```

### Create Sample Data
First, let's create some sample datasets to work with:

```python
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Create sample sales data
np.random.seed(42)
start_date = datetime(2023, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(365)]

sales_data = {
    'date': dates,
    'product_id': np.random.choice(['A001', 'A002', 'A003', 'B001', 'B002'], 365),
    'category': np.random.choice(['Electronics', 'Clothing', 'Books'], 365),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 365),
    'quantity': np.random.randint(1, 100, 365),
    'price': np.round(np.random.uniform(10, 500, 365), 2),
    'customer_id': np.random.randint(1000, 9999, 365)
}

# Save to CSV
sales_df = pd.DataFrame(sales_data)
sales_df.to_csv('sales_data.csv', index=False)

# Create customer data
customer_data = {
    'customer_id': range(1000, 10000),
    'customer_name': [f'Customer_{i}' for i in range(1000, 10000)],
    'age': np.random.randint(18, 80, 9000),
    'membership_tier': np.random.choice(['Bronze', 'Silver', 'Gold'], 9000),
    'signup_date': [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(9000)]
}

customers_df = pd.DataFrame(customer_data)
customers_df.to_csv('customers.csv', index=False)

print("Sample data created successfully!")
```

## Part 1: DuckDB Basics (10 minutes)

### Connecting to DuckDB
```python
# Create an in-memory DuckDB connection
conn = duckdb.connect()

# Alternative: Create a persistent database
# conn = duckdb.connect('my_database.db')

print("DuckDB connection established!")
```

### Querying CSV Files Directly
One of DuckDB's superpowers is the ability to query CSV files directly without loading them into memory first:

```python
# Query CSV file directly
result = conn.execute("""
    SELECT 
        category,
        COUNT(*) as transaction_count,
        SUM(quantity * price) as total_revenue,
        AVG(price) as avg_price
    FROM 'sales_data.csv'
    GROUP BY category
    ORDER BY total_revenue DESC
""").fetchall()

print("Revenue by Category:")
for row in result:
    print(f"{row[0]}: {row[2]:,.2f} (Avg Price: ${row[3]:.2f})")
```

### Working with Multiple CSV Files
```python
# Join data from multiple CSV files
join_query = """
    SELECT 
        s.date,
        s.product_id,
        s.quantity * s.price as revenue,
        c.customer_name,
        c.membership_tier
    FROM 'sales_data.csv' s
    JOIN 'customers.csv' c ON s.customer_id = c.customer_id
    WHERE s.quantity > 50
    ORDER BY revenue DESC
    LIMIT 10
"""

high_value_sales = conn.execute(join_query).fetchdf()
print("\nTop 10 High-Volume Sales:")
print(high_value_sales)
```

## Part 2: Working with Parquet Files (10 minutes)

### Converting CSV to Parquet
```python
# Convert CSV to Parquet using DuckDB
conn.execute("""
    COPY (SELECT * FROM 'sales_data.csv') 
    TO 'sales_data.parquet' (FORMAT PARQUET)
""")

conn.execute("""
    COPY (SELECT * FROM 'customers.csv') 
    TO 'customers.parquet' (FORMAT PARQUET)
""")

print("Files converted to Parquet format!")
```

### Comparing File Sizes
```python
import os

csv_size = os.path.getsize('sales_data.csv')
parquet_size = os.path.getsize('sales_data.parquet')

print(f"CSV file size: {csv_size:,} bytes")
print(f"Parquet file size: {parquet_size:,} bytes")
print(f"Space savings: {((csv_size - parquet_size) / csv_size * 100):.1f}%")
```

### Querying Parquet Files
```python
# Query Parquet files (typically faster than CSV)
monthly_sales = conn.execute("""
    SELECT 
        DATE_TRUNC('month', date) as month,
        region,
        SUM(quantity * price) as monthly_revenue
    FROM 'sales_data.parquet'
    GROUP BY DATE_TRUNC('month', date), region
    ORDER BY month, region
""").fetchdf()

print("\nMonthly Sales by Region:")
print(monthly_sales.head(10))
```

## Part 3: Advanced Analytics (10 minutes)

### Time Series Analysis
```python
# Daily sales trend with moving average
daily_trends = conn.execute("""
    SELECT 
        date,
        SUM(quantity * price) as daily_revenue,
        AVG(SUM(quantity * price)) OVER (
            ORDER BY date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) as weekly_moving_avg
    FROM 'sales_data.parquet'
    GROUP BY date
    ORDER BY date
""").fetchdf()

# Plot the trends
plt.figure(figsize=(12, 6))
plt.plot(daily_trends['date'], daily_trends['daily_revenue'], alpha=0.3, label='Daily Revenue')
plt.plot(daily_trends['date'], daily_trends['weekly_moving_avg'], label='7-Day Moving Average')
plt.title('Daily Sales Trends')
plt.xlabel('Date')
plt.ylabel('Revenue ($)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### Customer Segmentation
```python
# Advanced customer analysis
customer_analysis = conn.execute("""
    WITH customer_metrics AS (
        SELECT 
            c.customer_id,
            c.membership_tier,
            c.age,
            COUNT(s.date) as transaction_count,
            SUM(s.quantity * s.price) as total_spent,
            AVG(s.quantity * s.price) as avg_transaction,
            MAX(s.date) as last_purchase,
            MIN(s.date) as first_purchase
        FROM 'customers.parquet' c
        LEFT JOIN 'sales_data.parquet' s ON c.customer_id = s.customer_id
        GROUP BY c.customer_id, c.membership_tier, c.age
    ),
    customer_segments AS (
        SELECT *,
            CASE 
                WHEN total_spent > 10000 THEN 'High Value'
                WHEN total_spent > 5000 THEN 'Medium Value'
                WHEN total_spent > 0 THEN 'Low Value'
                ELSE 'No Purchases'
            END as value_segment
        FROM customer_metrics
    )
    SELECT 
        value_segment,
        membership_tier,
        COUNT(*) as customer_count,
        AVG(total_spent) as avg_spent,
        AVG(transaction_count) as avg_transactions
    FROM customer_segments
    GROUP BY value_segment, membership_tier
    ORDER BY avg_spent DESC
""").fetchdf()

print("\nCustomer Segmentation Analysis:")
print(customer_analysis)
```

### Product Performance Analysis
```python
# Product performance with statistical analysis
product_stats = conn.execute("""
    SELECT 
        product_id,
        category,
        COUNT(*) as sales_count,
        SUM(quantity) as total_quantity,
        SUM(quantity * price) as total_revenue,
        AVG(price) as avg_price,
        STDDEV(price) as price_stddev,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price) as median_price,
        MIN(price) as min_price,
        MAX(price) as max_price
    FROM 'sales_data.parquet'
    GROUP BY product_id, category
    ORDER BY total_revenue DESC
""").fetchdf()

print("\nProduct Performance Statistics:")
print(product_stats)
```

## Part 4: Integration with Pandas (5 minutes)

### DuckDB ↔ Pandas Integration
```python
# Convert DuckDB result to pandas DataFrame
df_from_duckdb = conn.execute("SELECT * FROM 'sales_data.parquet' LIMIT 1000").fetchdf()

# Register pandas DataFrame as a virtual table in DuckDB
conn.register('sales_df', df_from_duckdb)

# Query the registered DataFrame using SQL
result = conn.execute("""
    SELECT 
        region,
        AVG(quantity) as avg_quantity,
        COUNT(*) as transaction_count
    FROM sales_df
    GROUP BY region
""").fetchall()

print("Analysis of Pandas DataFrame via DuckDB:")
for row in result:
    print(f"{row[0]}: Avg Qty {row[1]:.1f}, Transactions: {row[2]}")
```

### Performance Comparison
```python
import time

# Pandas approach
start_time = time.time()
pandas_result = df_from_duckdb.groupby('region').agg({
    'quantity': 'mean',
    'price': 'count'
}).reset_index()
pandas_time = time.time() - start_time

# DuckDB approach
start_time = time.time()
duckdb_result = conn.execute("""
    SELECT 
        region,
        AVG(quantity) as avg_quantity,
        COUNT(*) as transaction_count
    FROM sales_df
    GROUP BY region
""").fetchdf()
duckdb_time = time.time() - start_time

print(f"\nPerformance Comparison:")
print(f"Pandas: {pandas_time:.4f} seconds")
print(f"DuckDB: {duckdb_time:.4f} seconds")
print(f"DuckDB is {pandas_time/duckdb_time:.1f}x faster")
```

## Part 5: Best Practices & Optimization (5 minutes)

### Query Optimization Tips
```python
# 1. Use column selection instead of SELECT *
efficient_query = """
    SELECT product_id, SUM(quantity * price) as revenue
    FROM 'sales_data.parquet'
    WHERE date >= '2023-06-01'
    GROUP BY product_id
"""

# 2. Use appropriate data types and filters early
optimized_query = """
    SELECT 
        DATE_TRUNC('week', date) as week,
        SUM(quantity * price) as weekly_revenue
    FROM 'sales_data.parquet'
    WHERE date BETWEEN '2023-01-01' AND '2023-12-31'
        AND quantity > 10
    GROUP BY DATE_TRUNC('week', date)
    ORDER BY week
"""

weekly_revenue = conn.execute(optimized_query).fetchdf()
print("Weekly Revenue Trends:")
print(weekly_revenue.head())
```

### Memory Management
```python
# For large datasets, use streaming or chunked processing
def process_large_dataset(file_path, chunk_size=10000):
    """Process large datasets in chunks to manage memory"""
    
    # Get total row count
    total_rows = conn.execute(f"SELECT COUNT(*) FROM '{file_path}'").fetchone()[0]
    
    results = []
    for offset in range(0, total_rows, chunk_size):
        chunk_query = f"""
            SELECT region, SUM(quantity * price) as revenue
            FROM '{file_path}'
            GROUP BY region
            LIMIT {chunk_size} OFFSET {offset}
        """
        chunk_result = conn.execute(chunk_query).fetchall()
        results.extend(chunk_result)
    
    return results

# Example usage (commented out for lab)
# large_result = process_large_dataset('sales_data.parquet')
```

### Creating Views for Reusable Queries
```python
# Create a view for commonly used aggregations
conn.execute("""
    CREATE VIEW monthly_summary AS
    SELECT 
        DATE_TRUNC('month', date) as month,
        product_id,
        category,
        region,
        SUM(quantity) as total_quantity,
        SUM(quantity * price) as total_revenue,
        COUNT(*) as transaction_count
    FROM 'sales_data.parquet'
    GROUP BY DATE_TRUNC('month', date), product_id, category, region
""")

# Use the view in queries
view_result = conn.execute("""
    SELECT 
        month,
        category,
        SUM(total_revenue) as category_revenue
    FROM monthly_summary
    WHERE month >= '2023-06-01'
    GROUP BY month, category
    ORDER BY month, category_revenue DESC
""").fetchdf()

print("\nMonthly Category Revenue:")
print(view_result.head(10))
```

## Wrap-up & Next Steps

### Key Takeaways
1. **DuckDB excels at analytical workloads** - Perfect for data analysis, aggregations, and complex queries
2. **Direct file querying** - No need to load entire datasets into memory
3. **Parquet optimization** - Significant performance and storage benefits
4. **Pandas integration** - Seamless workflow between SQL and Python
5. **SQL simplicity** - Leverage existing SQL knowledge for data analysis

### Clean Up
```python
# Close the connection
conn.close()

# Clean up files (optional)
import os
files_to_remove = ['sales_data.csv', 'customers.csv', 'sales_data.parquet', 'customers.parquet']
for file in files_to_remove:
    if os.path.exists(file):
        os.remove(file)
        
print("Lab completed successfully!")
```

## Additional Exercises (Homework)

1. **Data Pipeline Challenge**: Create a pipeline that reads multiple CSV files, cleans the data, and outputs optimized Parquet files
2. **Time Series Forecasting**: Use DuckDB's window functions to create lagged features for time series analysis
3. **Data Quality Assessment**: Write queries to identify missing values, outliers, and data quality issues
4. **Performance Benchmarking**: Compare DuckDB performance against other tools (SQLite, pandas, etc.) on larger datasets

## Resources for Further Learning
- [DuckDB Documentation](https://duckdb.org/docs/)
- [DuckDB Python API Reference](https://duckdb.org/docs/api/python/overview)
- [SQL for Data Analysis with DuckDB](https://duckdb.org/docs/sql/introduction)
- [Parquet File Format Specification](https://parquet.apache.org/docs/)