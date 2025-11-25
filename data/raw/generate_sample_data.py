"""
Generate sample Online Retail-like data for testing if the original file is not available.
This creates a CSV file compatible with the analysis notebooks.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_transactions = 5000
n_customers = 500
n_products = 100

# Date range
start_date = datetime(2010, 12, 1)
end_date = datetime(2011, 12, 1)

# Generate data
dates = [start_date + timedelta(days=int(x)) for x in np.random.rand(n_transactions) * (end_date - start_date).days]
customer_ids = np.random.choice(range(12000, 12000 + n_customers), n_transactions)
product_codes = [f'PROD{i:04d}' for i in np.random.choice(range(n_products), n_transactions)]
quantities = np.random.randint(1, 50, n_transactions)
prices = np.random.uniform(0.5, 50, n_transactions)
countries = np.random.choice(['United Kingdom', 'Germany', 'France', 'Netherlands', 'USA', 'Sweden'], n_transactions)
invoices = [f'INV{i:06d}' for i in np.random.choice(range(10000), n_transactions)]

# Create dataframe
df = pd.DataFrame({
    'InvoiceNo': invoices,
    'StockCode': product_codes,
    'Description': ['Product ' + code for code in product_codes],
    'Quantity': quantities,
    'InvoiceDate': dates,
    'UnitPrice': prices,
    'CustomerID': customer_ids,
    'Country': countries
})

# Save to CSV
output_path = 'Online Retail.csv'
df.to_csv(output_path, index=False)
print(f"Sample data generated: {output_path}")
print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
