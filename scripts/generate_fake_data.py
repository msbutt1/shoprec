"""Generate fake product purchase data for testing and development.

This module provides functionality to create synthetic ecommerce purchase data
for testing recommendation systems. It generates CSV files with simulated
user-product interactions including timestamps.

Example:
    Run the script directly to generate default data:
        $ python scripts/generate_fake_data.py

    Or import and use programmatically:
        from scripts.generate_fake_data import generate_fake_purchases
        df = generate_fake_purchases(num_users=100, num_products=200)
"""

import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

# Default configuration constants
DEFAULT_NUM_USERS = 50
DEFAULT_NUM_PRODUCTS = 100
DEFAULT_NUM_PURCHASES = 1000
DEFAULT_DAYS_BACK = 90
SECONDS_PER_DAY = 86400


def generate_fake_purchases(
    num_users: int = DEFAULT_NUM_USERS,
    num_products: int = DEFAULT_NUM_PRODUCTS,
    num_purchases: int = DEFAULT_NUM_PURCHASES,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """Generate synthetic purchase data for recommendation system testing.

    Creates a DataFrame with simulated user-product purchase interactions
    with random timestamps distributed across a specified date range.

    Args:
        num_users: Number of unique users to simulate. Must be positive.
        num_products: Number of unique products available. Must be positive.
        num_purchases: Total number of purchase records to generate.
            Must be positive.
        start_date: Start date for purchase timestamps. If None, defaults to
            90 days before the current date.
        end_date: End date for purchase timestamps. If None, defaults to
            the current date.

    Returns:
        A pandas DataFrame with the following columns:
            - user_id: Integer user identifier (1 to num_users)
            - product_id: Integer product identifier (1 to num_products)
            - timestamp: Datetime of the purchase event

        The DataFrame is sorted by timestamp in ascending order.

    Raises:
        ValueError: If any numeric parameter is non-positive or if
            start_date is after end_date.
    """
    # Validate inputs
    if num_users <= 0 or num_products <= 0 or num_purchases <= 0:
        raise ValueError(
            "num_users, num_products, and num_purchases must be positive"
        )

    # Set default date range if not provided
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=DEFAULT_DAYS_BACK)

    # Validate date range
    if start_date >= end_date:
        raise ValueError("start_date must be before end_date")

    # Generate purchase records
    purchases = []
    time_range = end_date - start_date
    days_range = time_range.days

    for _ in range(num_purchases):
        user_id = random.randint(1, num_users)
        product_id = random.randint(1, num_products)

        # Generate random timestamp within the specified date range
        random_days = random.randrange(days_range)
        random_seconds = random.randrange(SECONDS_PER_DAY)
        timestamp = start_date + timedelta(
            days=random_days, seconds=random_seconds
        )

        purchases.append({
            'user_id': user_id,
            'product_id': product_id,
            'timestamp': timestamp,
        })

    # Create DataFrame and sort by timestamp for realistic ordering
    df = pd.DataFrame(purchases)
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df


def main() -> None:
    """Main entry point for the data generation script.

    Generates fake purchase data with default parameters and saves it to
    data/fake_purchases.csv. Prints summary statistics upon completion.
    """
    # Configuration
    num_users = DEFAULT_NUM_USERS
    num_products = DEFAULT_NUM_PRODUCTS
    num_purchases = DEFAULT_NUM_PURCHASES
    
    # Generate synthetic purchase data
    print(f"Generating {num_purchases} fake purchases...")
    print(f"Users: {num_users}, Products: {num_products}")

    try:
        df = generate_fake_purchases(
            num_users=num_users,
            num_products=num_products,
            num_purchases=num_purchases,
        )
    except ValueError as e:
        print(f"Error generating data: {e}")
        return

    # Ensure data directory exists
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)

    # Save to CSV file
    output_path = data_dir / 'fake_purchases.csv'
    df.to_csv(output_path, index=False)

    # Print results summary
    print(f"\nData generated successfully!")
    print(f"Saved to: {output_path}")
    print(f"\nData preview:")
    print(df.head(10))
    print(f"\nData summary:")
    print(f"  Total purchases: {len(df)}")
    print(f"  Unique users: {df['user_id'].nunique()}")
    print(f"  Unique products: {df['product_id'].nunique()}")
    print(
        f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}"
    )


if __name__ == '__main__':
    main()

