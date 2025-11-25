#!/usr/bin/env python
"""
Quick verification that the Streamlit dashboard is ready to use
"""
import sys
from pathlib import Path
import pickle

print("\n" + "="*70)
print("🧠 CUSTOMER SEGMENTATION DASHBOARD - VERIFICATION")
print("="*70 + "\n")

checks_passed = 0
checks_total = 0

# Check 1: Models exist
checks_total += 1
print("[1] Checking required models...")
model_path = Path("models/kmeans_model.pkl")
scaler_path = Path("models/scaler.pkl")

if model_path.exists() and scaler_path.exists():
    with open(model_path, "rb") as f:
        kmeans = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    if kmeans.n_clusters == 4 and kmeans.n_features_in_ == 3:
        print(f"    ✅ K-Means model (4 clusters, 3 features)")
        print(f"    ✅ StandardScaler loaded")
        checks_passed += 1
    else:
        print(f"    ❌ Model configuration incorrect: {kmeans.n_clusters} clusters, {kmeans.n_features_in_} features")
else:
    print(f"    ❌ Models not found!")

# Check 2: App file exists
checks_total += 1
print("\n[2] Checking Streamlit app...")
app_path = Path("src/app.py")
if app_path.exists():
    # Check syntax
    try:
        with open(app_path, "r") as f:
            compile(f.read(), "src/app.py", "exec")
        print(f"    ✅ App file exists and has no syntax errors")
        checks_passed += 1
    except SyntaxError as e:
        print(f"    ❌ Syntax error in app: {e}")
else:
    print(f"    ❌ App file not found at {app_path}")

# Check 3: Sample data exists
checks_total += 1
print("\n[3] Checking sample data...")
data_path = Path("data/raw/Online Retail.csv")
if data_path.exists():
    import pandas as pd
    try:
        df = pd.read_csv(data_path, nrows=1)
        required_cols = {"InvoiceNo", "CustomerID", "InvoiceDate", "Quantity", "UnitPrice"}
        actual_cols = set(df.columns)
        
        if required_cols.issubset(actual_cols):
            print(f"    ✅ Sample data available with all required columns")
            checks_passed += 1
        else:
            missing = required_cols - actual_cols
            print(f"    ⚠️  Sample data missing columns: {missing}")
    except Exception as e:
        print(f"    ❌ Error reading sample data: {e}")
else:
    print(f"    ⚠️  Sample data not found (optional for testing)")
    checks_passed += 1  # This is optional

# Check 4: Required packages installed
checks_total += 1
print("\n[4] Checking required packages...")
required_packages = ["streamlit", "pandas", "numpy", "sklearn", "matplotlib", "seaborn"]
missing_packages = []

for pkg in required_packages:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        missing_packages.append(pkg)

if not missing_packages:
    print(f"    ✅ All required packages installed")
    checks_passed += 1
else:
    print(f"    ❌ Missing packages: {', '.join(missing_packages)}")
    print(f"       Run: pip install {' '.join(missing_packages)}")

# Summary
print("\n" + "="*70)
print(f"VERIFICATION RESULT: {checks_passed}/{checks_total} checks passed")
print("="*70 + "\n")

if checks_passed == checks_total:
    print("✅ DASHBOARD IS READY TO USE!")
    print("\nTo start the dashboard, run:")
    print("  streamlit run src/app.py")
    sys.exit(0)
else:
    print("⚠️  Some checks failed. Please fix issues above.")
    print("\nFor help, see: STREAMLIT_BUG_FIX_REPORT.md")
    sys.exit(1)
