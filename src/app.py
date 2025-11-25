import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
    }
    .section-header {
        color: #2ca02c;
        font-size: 24px;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ================================
# Load Saved Model + Scaler
# ================================

@st.cache_resource
def load_model_and_scaler():
    """Load pre-trained K-Means model and StandardScaler"""
    model_path = Path("models/kmeans_model.pkl")
    scaler_path = Path("models/scaler.pkl")
    
    if not model_path.exists():
        st.error(f"❌ Model not found at {model_path}")
        st.stop()
    if not scaler_path.exists():
        st.error(f"❌ Scaler not found at {scaler_path}")
        st.stop()
    
    with open(model_path, "rb") as f:
        kmeans = pickle.load(f)
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    return kmeans, scaler


# ================================
# Helper Functions
# ================================

def clean_data(df):
    """Clean and preprocess customer transaction data"""
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Remove rows with missing CustomerID
    df = df.dropna(subset=["CustomerID"])
    
    # Remove negative quantities (returns/cancellations)
    df = df[df["Quantity"] > 0]
    
    # Remove cancelled invoices (start with 'C')
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    
    # Remove non-positive prices
    df = df[df["UnitPrice"] > 0]
    
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Convert date column
    if "invoicedate" in df.columns:
        df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors='coerce')
        df = df.dropna(subset=["invoicedate"])
    
    # Convert CustomerID to int
    df["customerid"] = df["customerid"].astype(int)
    
    # Calculate total price per item
    if "quantity" in df.columns and "unitprice" in df.columns:
        df["totalprice"] = df["quantity"] * df["unitprice"]
    
    return df.reset_index(drop=True)


def generate_rfm(df):
    """Generate RFM (Recency, Frequency, Monetary) features"""
    if "invoicedate" not in df.columns or "customerid" not in df.columns:
        st.error("❌ Data must contain 'InvoiceDate' and 'CustomerID' columns")
        return None
    
    # Reference date is day after last transaction
    reference_date = df["invoicedate"].max() + timedelta(days=1)
    
    # Calculate RFM
    rfm = df.groupby("customerid").agg({
        "invoicedate": lambda x: (reference_date - x.max()).days,  # Recency
        "invoicedate": "nunique",  # Frequency (count of invoices)
        "totalprice": "sum"  # Monetary (total spent)
    }).rename(columns={
        "invoicedate": "Recency",
        "invoicedate": "Frequency",
        "totalprice": "Monetary"
    })
    
    # Use proper aggregation
    rfm = df.groupby("customerid").agg({
        "invoicedate": lambda x: (reference_date - x.max()).days,
    })
    rfm.columns = ["Recency"]
    
    rfm["Frequency"] = df.groupby("customerid")["invoicedate"].nunique()
    rfm["Monetary"] = df.groupby("customerid")["totalprice"].sum()
    
    # Log transform Monetary to handle skew
    rfm["Monetary_log"] = np.log1p(rfm["Monetary"])
    
    return rfm


def scale_features(rfm, scaler):
    """Scale RFM features using pre-fitted scaler"""
    features = rfm[["Recency", "Frequency", "Monetary"]]
    scaled = scaler.transform(features)
    
    scaled_df = pd.DataFrame(
        scaled,
        index=rfm.index,
        columns=["Recency_scaled", "Frequency_scaled", "Monetary_scaled"]
    )
    return scaled_df


def assign_clusters(scaled_df, kmeans):
    """Assign customers to clusters"""
    scaled_df = scaled_df.copy()
    scaled_df["Cluster"] = kmeans.predict(scaled_df[["Recency_scaled", "Frequency_scaled", "Monetary_scaled"]])
    return scaled_df


def map_personas(df):
    """Map cluster numbers to persona names"""
    persona_map = {
        0: "Champions",
        1: "At-Risk",
        2: "Loyal",
        3: "Growth"
    }
    df = df.copy()
    df["Persona"] = df["Cluster"].map(persona_map)
    return df


def get_persona_description(persona):
    """Return detailed description for each persona"""
    descriptions = {
        "Champions": {
            "emoji": "🏆",
            "description": "Recent, frequent, high-value buyers",
            "strategy": "VIP retention, referral programs, exclusive benefits",
            "priority": "🔴 HIGHEST"
        },
        "At-Risk": {
            "emoji": "⚠️",
            "description": "Previously valuable but inactive",
            "strategy": "Win-back campaigns, surveys, reactivation offers",
            "priority": "🔴 HIGH"
        },
        "Loyal": {
            "emoji": "💪",
            "description": "Consistent, engaged buyers with growth potential",
            "strategy": "Personalized recommendations, loyalty rewards, engagement campaigns",
            "priority": "🟠 HIGH"
        },
        "Growth": {
            "emoji": "🌱",
            "description": "New or occasional buyers",
            "strategy": "Onboarding, education, cross-selling",
            "priority": "🟡 MEDIUM"
        }
    }
    return descriptions.get(persona, {})


# ================================
# Main Streamlit App
# ================================

st.markdown("<h1 class='main-header'>🧠 Customer Segmentation Dashboard</h1>", unsafe_allow_html=True)
st.markdown("Upload your transaction data and automatically segment customers using RFM analysis + K-Means clustering")

# Load model and scaler
try:
    kmeans, scaler = load_model_and_scaler()
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

# ================================
# Sidebar - File Upload
# ================================

st.sidebar.markdown("## 📤 Upload Data")
st.sidebar.markdown("Upload a CSV file with customer transaction data")

uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="Requires columns: InvoiceNo, CustomerID, InvoiceDate, Quantity, UnitPrice"
)

# ================================
# Main Content
# ================================

if uploaded_file is not None:
    
    # Read data
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df)} transactions from {uploaded_file.name}")
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🧹 Processing", "🎯 Segmentation", "📈 Analysis"])
    
    # ================================
    # TAB 1: Overview
    # ================================
    with tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Transactions", len(df))
        with col2:
            st.metric("Unique Customers", df["CustomerID"].nunique() if "CustomerID" in df.columns else "N/A")
        with col3:
            st.metric("Date Range", f"{df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}" if "InvoiceDate" in df.columns else "N/A")
        
        st.subheader("📄 Raw Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("📋 Column Information")
        st.dataframe(df.dtypes.astype(str).rename("Data Type"), use_container_width=True)
    
    # ================================
    # TAB 2: Data Processing
    # ================================
    with tab2:
        st.subheader("🧹 Data Cleaning")
        st.write("Applying data quality filters...")
        
        # Clean data
        df_clean = clean_data(df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Rows", len(df))
        with col2:
            st.metric("Cleaned Rows", len(df_clean))
        
        st.info(f"ℹ️ Removed {len(df) - len(df_clean)} rows ({100 * (len(df) - len(df_clean)) / len(df):.1f}%)")
        
        st.subheader("✅ Cleaned Data Preview")
        st.dataframe(df_clean.head(10), use_container_width=True)
        
        # Generate RFM
        st.subheader("📊 RFM Calculation")
        st.write("Computing Recency, Frequency, and Monetary (RFM) metrics...")
        
        rfm = generate_rfm(df_clean)
        
        if rfm is not None:
            st.dataframe(rfm.head(10), use_container_width=True)
            
            # RFM Statistics
            st.subheader("📈 RFM Statistics")
            st.dataframe(rfm.describe().round(2), use_container_width=True)
    
    # ================================
    # TAB 3: Segmentation
    # ================================
    with tab3:
        st.subheader("🎯 Clustering & Segmentation")
        st.write("Scaling features and assigning customers to clusters...")
        
        # Scale features
        scaled_df = scale_features(rfm, scaler)
        
        # Assign clusters
        segmented = assign_clusters(scaled_df, kmeans)
        
        # Map personas
        segmented_with_personas = map_personas(segmented)
        
        # Combine with original RFM
        result_df = rfm.copy()
        result_df["Cluster"] = segmented["Cluster"]
        result_df["Persona"] = segmented_with_personas["Persona"]
        
        st.dataframe(result_df.head(15), use_container_width=True)
        
        # Cluster distribution
        st.subheader("📊 Cluster Distribution")
        cluster_counts = result_df["Cluster"].value_counts().sort_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            cluster_counts.plot(kind='bar', color='steelblue', ax=ax)
            ax.set_title("Customers by Cluster", fontsize=14, fontweight='bold')
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Count")
            plt.xticks(rotation=0)
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            cluster_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax.set_title("Cluster Distribution", fontsize=14, fontweight='bold')
            ax.set_ylabel("")
            st.pyplot(fig, use_container_width=True)
        
        # Download segmented data
        st.subheader("📥 Download Results")
        
        csv_data = result_df.to_csv()
        st.download_button(
            label="Download Segmented Customers (CSV)",
            data=csv_data,
            file_name="segmented_customers.csv",
            mime="text/csv",
            help="Save the segmented customer data for use in your systems"
        )
    
    # ================================
    # TAB 4: Analysis
    # ================================
    with tab4:
        st.subheader("📈 RFM Analysis by Cluster")
        
        # Recalculate for analysis
        scaled_df_temp = scale_features(rfm, scaler)
        segmented_temp = assign_clusters(scaled_df_temp, kmeans)
        
        # Add cluster to RFM
        analysis_df = rfm.copy()
        analysis_df["Cluster"] = segmented_temp["Cluster"]
        
        # Calculate cluster statistics
        cluster_stats = analysis_df.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
        
        st.subheader("📊 Average RFM by Cluster")
        st.dataframe(cluster_stats.round(2), use_container_width=True)
        
        # Heatmap
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(
            cluster_stats.T,
            annot=True,
            fmt=".0f",
            cmap="RdYlGn_r",
            ax=ax,
            cbar_kws={"label": "Value"}
        )
        ax.set_title("Average RFM Metrics by Cluster", fontsize=14, fontweight='bold')
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Metric")
        st.pyplot(fig, use_container_width=True)
        
        # Persona insights
        st.subheader("🎯 Segment Personas & Strategies")
        
        segmented_with_personas_temp = map_personas(segmented_temp)
        analysis_df["Persona"] = segmented_with_personas_temp["Persona"]
        
        for persona in ["Champions", "At-Risk", "Loyal", "Growth"]:
            persona_info = get_persona_description(persona)
            
            if persona in analysis_df["Persona"].values:
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.markdown(f"### {persona_info['emoji']} {persona}")
                    st.metric(
                        "Segment Size",
                        len(analysis_df[analysis_df["Persona"] == persona])
                    )
                
                with col2:
                    st.markdown(f"**Priority:** {persona_info['priority']}")
                    st.markdown(f"**Profile:** {persona_info['description']}")
                    st.markdown(f"**Strategy:** {persona_info['strategy']}")
                
                st.markdown("---")

else:
    # Landing page
    st.markdown("""
    ## 📋 How to Use This Dashboard
    
    1. **Upload CSV** — Use the sidebar to upload your customer transaction data
    2. **View Overview** — See summary statistics of your data
    3. **Process Data** — Automatically clean and calculate RFM metrics
    4. **Segment Customers** — Apply K-Means clustering to assign personas
    5. **Download Results** — Export segmented customer data for your systems
    6. **Analyze** — Explore segment characteristics and strategies
    
    ### 📊 Required Data Format
    
    Your CSV must contain these columns:
    - `InvoiceNo` — Unique transaction identifier
    - `CustomerID` — Unique customer identifier
    - `InvoiceDate` — Transaction date (any standard format)
    - `Quantity` — Number of items purchased
    - `UnitPrice` — Price per item
    
    ### 🎯 Customer Personas
    
    The dashboard will automatically segment your customers into 4 personas:
    
    - **🏆 Champions** — High-value, frequent buyers (focus on retention)
    - **⚠️ At-Risk** — Previously valuable but inactive (reactivation needed)
    - **💪 Loyal** — Consistent buyers with growth potential (upsell opportunities)
    - **🌱 Growth** — New or occasional buyers (nurture & onboard)
    
    ### 📁 Example Data
    
    ```
    InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country
    536365,85123A,WHITE HANGING HEART T-LIGHT,6,1/12/2010,2.55,17850,United Kingdom
    536365,71053,WHITE METAL LANTERN,2,1/12/2010,3.39,17850,United Kingdom
    ...
    ```
    
    ---
    
    **Ready to segment your customers? Upload a CSV to get started!** 👆
    """)
    
    st.info("💡 No data? Try the sample dataset from the Online Retail dataset (UCI ML Repository)")

# ================================
# Footer
# ================================

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🧠 Customer Segmentation Dashboard v1.0</p>
    <p><small>Built with Streamlit | Powered by K-Means Clustering & RFM Analysis</small></p>
</div>
""", unsafe_allow_html=True)
