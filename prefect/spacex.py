"""
SpaceX Launch Data ETL Pipeline with Prefect
=============================================
An ETL project using real SpaceX launch data

Run this flow in the remote Prefect Server (above)
and identify FOUR things:

1. How does this flow handle data storage? What are the potential advantages/disadvantages of this?
2. How many FOR loops exist in this flow?
3. What line determines how many crew are onbaord each launch? How is this derived?
4. Where in this flow is there essentially a JOIN between launches and rockets? Hint: It helps get the name of the rocket.

"""

import requests
import pandas as pd
from prefect import task, flow
from datetime import datetime
import json


@task(retries=2, retry_delay_seconds=10)
def extract_spacex_launches():
    """
    Extract all SpaceX launch data from the public API
    
    Returns:
        list: Raw launch data from SpaceX API
    """
    url = "https://api.spacexdata.com/v4/launches"
    
    print("🚀 Fetching SpaceX launch data from API...")
    response = requests.get(url)
    response.raise_for_status()
    
    data = response.json()
    print(f"✓ Extracted {len(data)} launches from SpaceX API")

    return data

@task
def extract_rockets_reference():
    """
    Extract rocket reference data to enrich launch information
    
    Returns:
        dict: Mapping of rocket IDs to rocket names
    """
    url = "https://api.spacexdata.com/v4/rockets"
    
    print("🚀 Fetching rocket reference data...")
    response = requests.get(url)
    response.raise_for_status()
    
    rockets = response.json()
    rocket_map = {r['id']: r['name'] for r in rockets}
    
    print(f"✓ Extracted {len(rocket_map)} rocket types")
    return rocket_map


@task
def transform_launch_data(raw_launches, rocket_map):
    """
    Transform raw SpaceX launch data into analysis-ready format
    
    Args:
        raw_launches (list): Raw launch data from API
        rocket_map (dict): Rocket ID to name mapping
    
    Returns:
        pd.DataFrame: Transformed launch data
    """
    print("🔧 Transforming launch data...")
    
    # Extract relevant fields from nested JSON
    launches = []
    for launch in raw_launches:
        launches.append({
            'flight_number': launch.get('flight_number'),
            'name': launch.get('name'),
            'date': launch.get('date_utc'),
            'rocket_id': launch.get('rocket'),
            'success': launch.get('success'),
            'details': launch.get('details', ''),
            'crew_count': len(launch.get('crew', [])),
            'payload_count': len(launch.get('payloads', [])),
            'launchpad': launch.get('launchpad'),
            'upcoming': launch.get('upcoming', False),
        })
    
    df = pd.DataFrame(launches)
    
    # Data type conversions
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.day_name()
    
    # Enrich with rocket names
    df['rocket_name'] = df['rocket_id'].map(rocket_map)
    
    # Handle missing success values (upcoming launches)
    df['success_status'] = df['success'].apply(
        lambda x: 'Success' if x == True 
        else 'Failure' if x == False 
        else 'Upcoming/Unknown'
    )
    
    # Add mission type indicator (crude but educational)
    df['has_crew'] = df['crew_count'] > 0
    
    # Filter to completed launches only for analysis
    completed_df = df[df['upcoming'] == False].copy()
    
    print(f"✓ Transformed {len(completed_df)} completed launches")
    print(f"  Date range: {completed_df['date'].min().date()} to {completed_df['date'].max().date()}")
    print(f"  Success rate: {(completed_df['success'].sum() / len(completed_df) * 100):.1f}%")
    print(f"  Rocket types: {completed_df['rocket_name'].nunique()}")
    
    return completed_df


@task
def calculate_metrics(df):
    """
    Calculate summary metrics for business intelligence
    
    Args:
        df (pd.DataFrame): Transformed launch data
    
    Returns:
        dict: Key metrics and insights
    """
    print("📊 Calculating metrics...")
    
    metrics = {
        'total_launches': len(df),
        'successful_launches': df['success'].sum(),
        'failed_launches': (df['success'] == False).sum(),
        'success_rate_pct': (df['success'].sum() / len(df) * 100),
        'crewed_missions': df['has_crew'].sum(),
        'most_used_rocket': df['rocket_name'].mode()[0] if not df.empty else None,
        'launches_by_year': df['year'].value_counts().to_dict(),
        'launches_by_rocket': df['rocket_name'].value_counts().to_dict(),
    }
    
    print(f"\n📈 Key Metrics:")
    print(f"   Total Launches: {metrics['total_launches']}")
    print(f"   Success Rate: {metrics['success_rate_pct']:.1f}%")
    print(f"   Crewed Missions: {metrics['crewed_missions']}")
    print(f"   Most Used Rocket: {metrics['most_used_rocket']}")
    
    return metrics


@task
def load_launch_data(df, metrics, filename="spacex_launches.csv", metrics_file="spacex_metrics.json"):
    """
    Load transformed data and metrics to files
    
    Args:
        df (pd.DataFrame): Transformed launch data
        metrics (dict): Calculated metrics
        filename (str): Output CSV filename
        metrics_file (str): Output JSON filename for metrics
    
    Returns:
        tuple: (data_filename, metrics_filename)
    """
    print(f"💾 Loading data to files...")
    
    # Save main dataset
    df.to_csv(filename, index=False)
    print(f"✓ Saved {len(df)} records to {filename}")
    
    # Save metrics
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"✓ Saved metrics to {metrics_file}")
    
    # Display preview
    print(f"\n📋 Data Preview:")
    print(df[['flight_number', 'name', 'date', 'rocket_name', 'success_status']].head(10))
    
    return filename, metrics_file


@flow(name="xtm9px", log_prints=True)
def spacex_etl_flow():
    """
    Complete ETL pipeline for SpaceX launch data
    
    This flow demonstrates:
    - API data extraction with retries
    - Data enrichment from multiple sources
    - Complex transformations
    - Metric calculation
    - Multi-file output
    """
    print("="*60)
    print("🚀 SpaceX Launch Data ETL Pipeline")
    print("="*60)
    
    # Extract
    raw_launches = extract_spacex_launches()
    rocket_map = extract_rockets_reference()
    
    # Transform
    transformed_data = transform_launch_data(raw_launches, rocket_map)
    metrics = calculate_metrics(transformed_data)
    
    # Load
    data_file, metrics_file = load_launch_data(transformed_data, metrics)
    
    print("\n" + "="*60)
    print(f"✅ Pipeline Complete!")
    print(f"   📁 Launch Data: {data_file}")
    print(f"   📁 Metrics: {metrics_file}")
    print("="*60)
    
    return data_file, metrics_file

if __name__ == "__main__":
    spacex_etl_flow()
