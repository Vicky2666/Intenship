import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------
# Load dataset with encoding fix
# -----------------------------
file_path = "superstore.csv"

if not os.path.exists(file_path):
    st.error(f"❌ File '{file_path}' not found in the current directory.")
    st.stop()

try:
    df = pd.read_csv(file_path, encoding="latin1")
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding="cp1252")

# Convert 'Order Date' to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df.dropna(inplace=True)

# -----------------------------
# Streamlit Dashboard
# -----------------------------
st.set_page_config(page_title="Superstore Dashboard", layout="wide")

# Big Title
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>📊 Superstore Sales & Profit Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar filters
st.sidebar.header("Filter Options")

# Category & Segment filter
categories = st.sidebar.multiselect("Select Category:", df['Category'].unique(), default=df['Category'].unique())
segments = st.sidebar.multiselect("Select Segment:", df['Segment'].unique(), default=df['Segment'].unique())

# Date filter
min_date, max_date = df['Order Date'].min(), df['Order Date'].max()
date_range = st.sidebar.date_input("Select Date Range:", [min_date, max_date])

# Apply filters
filtered_df = df[
    (df['Category'].isin(categories)) &
    (df['Segment'].isin(segments)) &
    (df['Order Date'] >= pd.to_datetime(date_range[0])) &
    (df['Order Date'] <= pd.to_datetime(date_range[1]))
]

# -----------------------------
# KPIs
# -----------------------------
total_sales = filtered_df['Sales'].sum()
total_profit = filtered_df['Profit'].sum()

col1, col2 = st.columns(2)
col1.metric("💰 Total Sales", f"${total_sales:,.2f}")
col2.metric("📈 Total Profit", f"${total_profit:,.2f}")

# -----------------------------
# Plots
# -----------------------------
st.subheader("Monthly Sales Trend")
monthly_sales = filtered_df.groupby(filtered_df['Order Date'].dt.to_period('M'))['Sales'].sum()
fig, ax = plt.subplots(figsize=(10, 4))
monthly_sales.plot(ax=ax, marker='o')
plt.xticks(rotation=45)
st.pyplot(fig)

st.subheader("Sales by Category & Sub-Category")
cat_sales = filtered_df.groupby(['Category', 'Sub-Category'])['Sales'].sum().unstack().fillna(0)
fig, ax = plt.subplots(figsize=(10, 5))
cat_sales.T.plot(kind='bar', stacked=True, ax=ax)
plt.xticks(rotation=60)
st.pyplot(fig)

st.subheader("Profit by Sub-Category")
profit_by_subcat = filtered_df.groupby('Sub-Category')['Profit'].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
profit_by_subcat.plot(kind='bar', ax=ax, color="skyblue")
plt.xticks(rotation=60)
st.pyplot(fig)

st.subheader("Profit by Segment")
segment_profit = filtered_df.groupby('Segment')['Profit'].sum()
fig, ax = plt.subplots()
segment_profit.plot(kind='pie', autopct='%1.1f%%', ax=ax)
ax.set_ylabel("")
st.pyplot(fig)

# -----------------------------
# Data Summary
# -----------------------------
st.subheader("Category Sales & Profit Summary")
summary = filtered_df.groupby('Category')[['Sales', 'Profit']].sum()
st.dataframe(summary)

# -----------------------------
# Download Option
# -----------------------------
st.subheader("📥 Download Filtered Data")
csv_data = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV",
    data=csv_data,
    file_name="filtered_superstore.csv",
    mime="text/csv"
)

st.success("✅ Store Sales and Profit Analysis Completed!")
