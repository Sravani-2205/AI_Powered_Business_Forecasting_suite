
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from forecast_runner import run_all_forecasts
from utils import simulate_sales
import io
from datetime import datetime, timedelta
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score


# App configuration
st.set_page_config(
    page_title="AI Business Forecasting Tool",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #f0f8ff;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #000000;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        border-radius: 5px;
        margin: 1rem 0;
        color: #ffffff
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "🏠 Welcome"
if "comparison_scenarios" not in st.session_state:
    st.session_state.comparison_scenarios = []
if "forecast_history" not in st.session_state:
    st.session_state.forecast_history = []

# Sidebar navigation with icons
st.sidebar.markdown("## NAVIGATION")
selected_page = st.sidebar.radio("Choose a section", [
    "🏠 Welcome",
    "📤 Upload Data",
    "📊 Forecast Dashboard",
    "🔮 What-If Simulation",
    "⚖️ Scenario Comparison",
    "📈 Trend Analysis",
    "🎯 Goal Tracker",
    "💡 Business Insights",
    "📚 User Guide"
])

if selected_page != st.session_state.page:
    st.session_state.page = selected_page
    st.rerun()

page = st.session_state.page

# Sidebar statistics (if data is loaded)
try:
    df = pd.read_csv("enriched_sales_data.csv", parse_dates=["Date"])
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Quick Stats")
    st.sidebar.metric("Total Records", len(df))
    st.sidebar.metric("Avg Sales", f"₹{df['Sales'].mean():.2f}")
    if "Profit" in df.columns:
        st.sidebar.metric("Avg Profit", f"₹{df['Profit'].mean():.2f}")
    st.sidebar.metric("Date Range", f"{(df['Date'].max() - df['Date'].min()).days} days")
except:
    pass

# Sidebar tips
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Quick Tip")
tips = {
    "🏠 Welcome": "Start here to learn about all available features.",
    "📤 Upload Data": "Supports CSV files with Date and Sales columns.",
    "📊 Forecast Dashboard": "View multiple forecasting models side-by-side.",
    "🔮 What-If Simulation": "Test different business scenarios interactively.",
    "⚖️ Scenario Comparison": "Compare up to 5 scenarios simultaneously.",
    "📈 Trend Analysis": "Deep dive into patterns and anomalies.",
    "🎯 Goal Tracker": "Set targets and track your progress.",
    "💡 Business Insights": "Get AI-powered recommendations.",
    "📚 User Guide": "Learn how to use each feature effectively."
}
st.sidebar.info(tips.get(page, "Navigate through the app using the menu."))

# Progress indicator
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Progress")
progress_steps = {
    "🏠 Welcome": 0,
    "📤 Upload Data": 20,
    "📊 Forecast Dashboard": 40,
    "🔮 What-If Simulation": 60,
    "⚖️ Scenario Comparison": 70,
    "📈 Trend Analysis": 80,
    "🎯 Goal Tracker": 85,
    "💡 Business Insights": 90,
    "📚 User Guide": 100
}
st.sidebar.progress(progress_steps.get(page, 0) / 100)

# Safely retrieve forecast results
results = st.session_state.get("results")

# WELCOME PAGE
if page == "🏠 Welcome":
    st.markdown('<div class="main-header">  AI BUSINESS FORECASTING TOOL</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image("forecast_flow_logo.png", width=500)
        except:
            st.markdown("### 📊 Advanced Sales Forecasting Platform")
    
    st.markdown("---")
    
    # Feature highlights
    st.markdown("## ✨ Smart Features for Smarter Decision")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Core Features
        - **Multi-Model Forecasting**: Prophet, LSTM, AutoML
        - **Interactive What-If Analysis**: Test scenarios
        - **Smart Business Insights**: AI-powered recommendations
        - **Visual Analytics**: Beautiful charts and graphs
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 Additional Features
        - **Scenario Comparison**: Compare multiple scenarios
        - **Trend Analysis**: Detect patterns and anomalies
        - **Goal Tracker**: Set and monitor targets
        - **Export Reports**: Download comprehensive PDFs
        """)
    
    with col3:
        st.markdown("""
        ### 💼 Business Value
        - **Better Planning**: Make data-driven decisions
        - **Risk Management**: Identify potential issues
        - **Resource Optimization**: Allocate budget wisely
        - **Performance Tracking**: Monitor KPIs continuously
        """)
    
    st.markdown("---")
    
    # Quick start guide
    st.markdown("## 🚀 Quick Start Guide")
    
    steps = [
        ("1️⃣", "Upload Data", "Import your sales CSV file with Date and Sales columns"),
        ("2️⃣", "Run Forecasts", "Generate predictions using multiple AI models"),
        ("3️⃣", "Analyze Results", "Explore forecasts, trends, and insights"),
        ("4️⃣", "Test Scenarios", "Simulate different business conditions"),
        ("5️⃣", "Take Action", "Implement data-driven strategies")
    ]
    
    for emoji, title, desc in steps:
        st.markdown(f"**{emoji} {title}**: {desc}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎬 Get Started →", use_container_width=True):
            st.session_state.page = "📤 Upload Data"
            st.rerun()
    with col2:
        if st.button("📚 View User Guide", use_container_width=True):
            st.session_state.page = "📚 User Guide"
            st.rerun()

# UPLOAD DATA (Enhanced)
elif page == "📤 Upload Data":
    st.header("📤 Upload Your Sales Data")
    
    # Instructions in tabs
    tab1, tab2, tab3 = st.tabs(["📋 Requirements", "📝 Sample Format", "💡 Tips"])
    
    with tab1:
        st.markdown("""
        ### Required Columns:
        - **Date**: Any date format (YYYY-MM-DD, MM/DD/YYYY, etc.)
        - **Sales**: Numeric sales values
        
        ### Optional Columns (Enhance Predictions):
        - **Profit**: Profit figures
        - **Category**: Product categories
        - **Region**: Geographic regions
        - **Promotion**: Promotional activities (0/1)
        - **Holiday**: Holiday indicator (0/1)
        - **InflationRate**: Inflation percentage
        - **AdSpend**: Advertising budget
        """)
    
    with tab2:
        sample_data = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Sales': [15000, 18000, 16500],
            'Profit': [4500, 5400, 4950],
            'Promotion': [0, 1, 0]
        })
        st.dataframe(sample_data)
        st.download_button(
            "📥 Download Sample CSV",
            sample_data.to_csv(index=False),
            "sample_sales_data.csv",
            "text/csv"
        )
    
    with tab3:
        st.markdown("""
        - Include at least **90 days** of historical data for better accuracy
        - Ensure dates are in chronological order
        - Remove any duplicate entries
        - Fill missing values or use consistent NULL indicators
        - More optional columns = better predictions
        """)
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📥 Choose your CSV file", type="csv", help="Upload a CSV file with your sales data")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            required_columns = {"Date", "Sales"}
            
            if not required_columns.issubset(df.columns):
                st.error("❌ Your file must include 'Date' and 'Sales' columns.")
            else:
                # Convert Date column to datetime before saving
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                
                # Check if date conversion was successful
                if df['Date'].isnull().any():
                    st.error("❌ Some dates couldn't be parsed. Please check your date format.")
                else:
                    # Save the processed dataframe
                    df.to_csv("enriched_sales_data.csv", index=False)
                    
                    st.success("✅ File uploaded successfully!")
                
                # Data preview with statistics
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("📊 Data Preview")
                    st.dataframe(df.head(10), use_container_width=True)
                
                with col2:
                    st.subheader("📈 Quick Stats")
                    st.metric("Total Records", len(df))
                    st.metric("Date Range", f"{(df['Date'].max() - df['Date'].min()).days} days")
                    st.metric("Avg Sales", f"₹{df['Sales'].mean():.2f}")
                    st.metric("Max Sales", f"₹{df['Sales'].max():.2f}")
                    st.metric("Min Sales", f"₹{df['Sales'].min():.2f}")
                
                # Data quality check
                st.subheader("🔍 Data Quality Check")
                quality_col1, quality_col2, quality_col3 = st.columns(3)
                
                with quality_col1:
                    missing = df.isnull().sum().sum()
                    if missing == 0:
                        st.success(f"✅ No missing values")
                    else:
                        st.warning(f"⚠️ {missing} missing values found")
                
                with quality_col2:
                    duplicates = df.duplicated().sum()
                    if duplicates == 0:
                        st.success("✅ No duplicate rows")
                    else:
                        st.warning(f"⚠️ {duplicates} duplicate rows")
                
                with quality_col3:
                    optional_cols = ['Profit', 'Category', 'Region', 'Promotion', 'Holiday', 'InflationRate', 'AdSpend']
                    available = sum([1 for col in optional_cols if col in df.columns])
                    st.info(f"ℹ️ {available}/{len(optional_cols)} optional columns")
                
                    # Run forecasts
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Run Forecasts"):
                        with st.spinner("⏳ Running forecasts..."):
                            try:
                                st.session_state.results = run_all_forecasts("enriched_sales_data.csv")
                                st.success("✅ Forecasts are ready!")
                                st.balloons()
                            except Exception as e:
                                st.error(f"❌ Forecast error: {e}")
                                st.info("💡 Check that your data has been preprocessed correctly")
                                                                   
                with col2:
                    if st.button("➡️ View Dashboard", use_container_width=True):
                        if results:
                            st.session_state.page = "📊 Forecast Dashboard"
                            st.rerun()
                        else:
                            st.warning("⚠️ Please run forecasts first")
        
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            st.info("💡 Check that your file is a valid CSV with proper formatting")

# FORECAST DASHBOARD (Enhanced)
elif page == "📊 Forecast Dashboard":
    st.header("📊 Forecast Dashboard")
    
    if results is not None:
        # Summary metrics at top
        st.subheader("📈 Forecast Summary")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        try:
            with metric_col1:
                prophet_avg = results["Prophet"]["yhat"].mean()
                st.metric("Prophet Avg", f"₹{prophet_avg:.2f}", delta=f"{(prophet_avg/results['Prophet']['yhat'].iloc[0]-1)*100:.1f}%")
            
            with metric_col2:
                lstm_pred = float(results["LSTM"])
                st.metric("LSTM Next Month", f"₹{lstm_pred:.2f}")
            
            with metric_col3:
                automl_mae = results["AutoML"]["mae"]
                st.metric("AutoML Accuracy", f"MAE: {automl_mae:.2f}", delta="Lower is better", delta_color="inverse")
            
            with metric_col4:
                forecast_days = len(results["Prophet"])
                st.metric("Forecast Horizon", f"{forecast_days} days")
        except:
            st.warning("⚠️ Some metrics unavailable")
        
        st.markdown("---")
        
        # Tabbed interface for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Prophet Analysis", "🧠 LSTM Forecast", "🤖 AutoML Insights", "📊 Model Comparison"])
        
        with tab1:
            st.subheader("📈 Prophet Time Series Forecast")
            try:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(results["Prophet"]["ds"], results["Prophet"]["yhat"], label="Forecast", color="#1f77b4", linewidth=2)
                
                if "yhat_lower" in results["Prophet"].columns and "yhat_upper" in results["Prophet"].columns:
                    ax.fill_between(results["Prophet"]["ds"], 
                                   results["Prophet"]["yhat_lower"], 
                                   results["Prophet"]["yhat_upper"], 
                                   alpha=0.3, color="#1f77b4", label="Confidence Interval")
                
                ax.set_xlabel("Date", fontsize=12)
                ax.set_ylabel("Sales (₹)", fontsize=12)
                ax.set_title("Prophet Forecast with Confidence Intervals", fontsize=14, fontweight="bold")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                st.caption("📊 Prophet captures seasonality, trends, and holidays. The shaded area shows prediction confidence.")
                
                # Prophet statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Max Forecast", f"₹{results['Prophet']['yhat'].max():.2f}")
                    st.metric("Min Forecast", f"₹{results['Prophet']['yhat'].min():.2f}")
                with col2:
                    volatility = results['Prophet']['yhat'].std()
                    st.metric("Volatility (Std Dev)", f"₹{volatility:.2f}")
                    st.metric("Forecast Range", f"₹{results['Prophet']['yhat'].max() - results['Prophet']['yhat'].min():.2f}")
            
            except Exception as e:
                st.error(f"⚠️ Prophet visualization error: {e}")
        
        with tab2:
            st.subheader("🧠 LSTM Deep Learning Forecast")
            try:
                lstm_value = float(results["LSTM"])
                
                # Visual gauge for LSTM
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.barh(["Next Month Prediction"], [lstm_value], color="#ff7f0e", height=0.5)
                ax.set_xlabel("Predicted Sales (₹)", fontsize=12)
                ax.set_title("LSTM Next Month Forecast", fontsize=14, fontweight="bold")
                ax.grid(True, axis='x', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                st.caption("🔮 LSTM uses recent patterns and momentum to predict short-term sales.")
                
                # Comparison with historical average
                try:
                    df = pd.read_csv("enriched_sales_data.csv")
                    hist_avg = df["Sales"].mean()
                    diff = lstm_value - hist_avg
                    diff_pct = (diff / hist_avg) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("LSTM Prediction", f"₹{lstm_value:.2f}")
                    with col2:
                        st.metric("Historical Avg", f"₹{hist_avg:.2f}")
                    with col3:
                        st.metric("Difference", f"₹{diff:.2f}", delta=f"{diff_pct:.1f}%")
                except:
                    pass
            
            except Exception as e:
                st.error(f"⚠️ LSTM visualization error: {e}")
        
        with tab3:
            st.subheader("🤖 AutoML Multi-Model Insights")
            try:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(results["AutoML"]["predictions"], label="AutoML Prediction", color="#2ca02c", linewidth=2, marker='o')
                ax.set_xlabel("Time Period", fontsize=12)
                ax.set_ylabel("Sales (₹)", fontsize=12)
                ax.set_title("AutoML Ensemble Forecast", fontsize=14, fontweight="bold")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                st.metric("Mean Absolute Error", f"{results['AutoML']['mae']:.2f}")
                st.caption("Lower MAE indicates better accuracy") 
                
                st.caption("🧠 AutoML combines multiple algorithms to optimize prediction accuracy.")
            
            except Exception as e:
                st.error(f"⚠️ AutoML visualization error: {e}")
        
        with tab4:
            st.subheader("📊 Model Comparison")
            try:
                comparison_cols = st.columns(3)
                
                with comparison_cols[0]:
                    st.markdown("### Prophet")
                    st.line_chart(results["Prophet"].set_index("ds")["yhat"])
                    st.caption("Best for: Long-term trends, seasonality")
                
                with comparison_cols[1]:
                    st.markdown("### LSTM")
                    # Create a simple visualization for LSTM
                    lstm_df = pd.DataFrame({
                        'Period': ['Next Month'],
                        'Prediction': [float(results["LSTM"])]
                    })
                    st.bar_chart(lstm_df.set_index('Period'))
                    st.caption("Best for: Short-term, momentum-based")
                
                with comparison_cols[2]:
                    st.markdown("### AutoML")
                    st.line_chart(results["AutoML"]["predictions"])
                    st.caption("Best for: Overall accuracy, robustness")
                
                # Model recommendation
                st.markdown("---")
                st.subheader("🎯 Model Recommendation")
                
                try:
                    automl_mae = results['AutoML']['mae']
                    if automl_mae < 1000:
                        recommended = "AutoML"
                        reason = "Lowest error rate and most consistent predictions"
                    elif len(results["Prophet"]) > 30:
                        recommended = "Prophet"
                        reason = "Best for capturing long-term trends and seasonal patterns"
                    else:
                        recommended = "LSTM"
                        reason = "Most responsive to recent changes"
                    
                    st.success(f"✅ Recommended Model: **{recommended}**")
                    st.info(f"💡 Reason: {reason}")
                except:
                    st.info("Compare models based on your specific needs and time horizon")
            
            except Exception as e:
                st.error(f"⚠️ Comparison error: {e}")
        
        st.markdown("---")
        
        # Navigation buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔮 Try What-If Scenarios", use_container_width=True):
                st.session_state.page = "🔮 What-If Simulation"
                st.rerun()
        with col2:
            if st.button("📈 Trend Analysis", use_container_width=True):
                st.session_state.page = "📈 Trend Analysis"
                st.rerun()
        with col3:
            if st.button("💡 View Insights", use_container_width=True):
                st.session_state.page = "💡 Business Insights"
                st.rerun()
    
    else:
        st.warning("⚠️ No forecast data available. Please upload data and run forecasts first.")
        if st.button("← Go to Upload", use_container_width=True):
            st.session_state.page = "📤 Upload Data"
            st.rerun()

# WHAT-IF SIMULATION (Enhanced)
elif page == "🔮 What-If Simulation":
    st.header("🔮 What-If Scenario Simulator")
    
    if results:
        st.markdown("### 🎯 Design Your Business Scenario")
        st.info("💡 Adjust the parameters below to see how different strategies might impact your sales.")
        
        # Organize inputs in columns
        st.subheader("💰 Financial Factors")
        fin_col1, fin_col2, fin_col3 = st.columns(3)
        
        with fin_col1:
            ad_spend = st.slider("Ad Budget (₹)", 1000, 50000, 5000, 500, 
                               help="Monthly advertising spend")
        with fin_col2:
            discount = st.slider("Discount Rate (%)", 0.0, 50.0, 10.0, 1.0,
                               help="Average discount offered")
        with fin_col3:
            inflation = st.slider("Inflation Rate (%)", 2.0, 15.0, 5.5, 0.5,
                               help="Economic inflation impact")
        
        st.subheader("👥 Operational Factors")
        op_col1, op_col2, op_col3 = st.columns(3)
        
        with op_col1:
            staffing = st.slider("Staffing Level", 1, 100, 25, 1,
                               help="Number of staff members")
        with op_col2:
            online_campaign = st.slider("Digital Marketing (0-100)", 0, 100, 50, 5,
                                      help="Online campaign effectiveness")
        with op_col3:
            inventory = st.slider("Inventory Level (%)", 50, 150, 100, 5,
                                help="Stock availability percentage")
        
        st.subheader("📅 Seasonal & External Factors")
        ext_col1, ext_col2, ext_col3 = st.columns(3)
        
        with ext_col1:
            holiday_boost = st.checkbox("🎉 Holiday/Festival Season",
                                       help="High-demand period")
        with ext_col2:
            competitor_activity = st.slider("Competitor Activity", 0, 100, 50, 5,
                                          help="Market competition level")
        with ext_col3:
            weather_impact = st.selectbox("Weather Impact",
                                         ["Favorable", "Neutral", "Unfavorable"],
                                         help="Weather effect on sales")
        
        # Calculate simulation
        base_sales = float(results["LSTM"])
        simulated = simulate_sales(ad_spend, inflation, base_sales)
        
        # Apply all factors
        simulated *= (1 + discount / 100 * 0.3)
        simulated *= (1 + staffing / 100 * 0.2)
        simulated *= (1 + online_campaign / 100 * 0.25)
        simulated *= (inventory / 100)
        simulated *= (1 - competitor_activity / 200)
        
        if holiday_boost:
            simulated *= 1.20
        
        weather_multiplier = {"Favorable": 1.1, "Neutral": 1.0, "Unfavorable": 0.9}
        simulated *= weather_multiplier[weather_impact]
        
        # Results display
        st.markdown("---")
        st.subheader("📊 Simulation Results")
        
        result_col1, result_col2, result_col3, result_col4 = st.columns(4)
        
        with result_col1:
            st.metric("Base Forecast", f"₹{base_sales:.2f}")
        with result_col2:
            st.metric("Simulated Sales", f"₹{simulated:.2f}")
        with result_col3:
            diff = simulated - base_sales
            st.metric("Difference", f"₹{diff:.2f}", delta=f"{(diff/base_sales)*100:.1f}%")
        with result_col4:
            roi = (simulated - base_sales) / (ad_spend / 1000) if ad_spend > 0 else 0
            st.metric("ROI Estimate", f"{roi:.2f}x")
        
        # Visual comparison
        fig, ax = plt.subplots(figsize=(10, 5))
        scenarios = ['Base Forecast', 'With Current\nSettings']
        values = [base_sales, simulated]
        colors = ['#1f77b4', '#ff7f0e' if simulated > base_sales else '#d62728']
        
        bars = ax.bar(scenarios, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylabel('Sales (₹)', fontsize=12, fontweight='bold')
        ax.set_title('Scenario Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'₹{height:.0f}',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Factor breakdown
        with st.expander("📋 Detailed Factor Breakdown", expanded=True):
            factors = {
                "💰 Ad Budget": f"₹{ad_spend:,}",
                "🏷️ Discount Rate": f"{discount}%",
                "📈 Inflation": f"{inflation}%",
                "👥 Staffing": f"{staffing} people",
                "🌐 Digital Marketing": f"{online_campaign}/100",
                "📦 Inventory Level": f"{inventory}%",
                "🏪 Competitor Activity": f"{competitor_activity}/100",
                "🌤️ Weather": weather_impact,
                "🎉 Holiday Boost": "✅ Yes" if holiday_boost else "❌ No"
            }
            
            for factor, value in factors.items():
                st.markdown(f"**{factor}**: {value}")
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 AI Recommendations")
        
        recommendations = []
        
        if simulated < base_sales:
            recommendations.append("⚠️ **Current settings may reduce sales.** Consider adjusting your strategy.")
        else:
            recommendations.append("✅ **Positive impact detected!** This scenario looks promising.")
        
        if ad_spend < 5000:
            recommendations.append("💰 **Low ad budget:** Consider increasing marketing spend for better reach.")
        
        if discount > 30:
            recommendations.append("🏷️ **High discount:** Monitor profit margins carefully.")
        
        if staffing < 20:
            recommendations.append("👥 **Understaffed:** May affect customer service quality.")
        
        if online_campaign < 40:
            recommendations.append("🌐 **Weak digital presence:** Boost online marketing efforts.")
        
        if inventory < 80:
            recommendations.append("📦 **Low inventory:** Risk of stockouts during high demand.")
        
        for rec in recommendations:
            st.markdown(f"- {rec}")
        
        # Save scenario
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            scenario_name = st.text_input("Scenario Name", f"Scenario {len(st.session_state.comparison_scenarios) + 1}")
            if st.button("💾 Save This Scenario", use_container_width=True):
                scenario_data = {
                    'name': scenario_name,
                    'timestamp': datetime.now(),
                    'base_sales': base_sales,
                    'simulated_sales': simulated,
                    'parameters': {
                        'ad_spend': ad_spend,
                        'discount': discount,
                        'inflation': inflation,
                        'staffing': staffing,
                        'online_campaign': online_campaign,
                        'inventory': inventory,
                        'holiday_boost': holiday_boost,
                        'competitor_activity': competitor_activity,
                        'weather_impact': weather_impact
                    }
                }
                st.session_state.comparison_scenarios.append(scenario_data)
                st.success(f"✅ Scenario '{scenario_name}' saved!")
        
        with col2:
            if st.button("⚖️ Compare Scenarios →", use_container_width=True):
                st.session_state.page = "⚖️ Scenario Comparison"
                st.rerun()
    
    else:
        st.warning("⚠️ Please run forecasts first.")
        if st.button("← Go to Upload", use_container_width=True):
            st.session_state.page = "📤 Upload Data"
            st.rerun()

# SCENARIO COMPARISON (NEW)
elif page == "⚖️ Scenario Comparison":
    st.header("⚖️ Scenario Comparison")
    
    if len(st.session_state.comparison_scenarios) == 0:
        st.info("💡 No saved scenarios yet. Create scenarios in the What-If Simulation page first.")
        if st.button("← Go to What-If Simulation", use_container_width=True):
            st.session_state.page = "🔮 What-If Simulation"
            st.rerun()
    else:
        st.markdown(f"### Comparing {len(st.session_state.comparison_scenarios)} Scenario(s)")
        
        # Scenario selector
        if len(st.session_state.comparison_scenarios) > 5:
            st.warning("⚠️ Showing only the 5 most recent scenarios for clarity")
        
        scenarios_to_show = st.session_state.comparison_scenarios[-5:]
        
        # Comparison chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        names = [s['name'] for s in scenarios_to_show]
        base_values = [s['base_sales'] for s in scenarios_to_show]
        sim_values = [s['simulated_sales'] for s in scenarios_to_show]
        
        x = np.arange(len(names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, base_values, width, label='Base Forecast', color='#1f77b4', alpha=0.7)
        bars2 = ax.bar(x + width/2, sim_values, width, label='Simulated', color='#ff7f0e', alpha=0.7)
        
        ax.set_xlabel('Scenarios', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sales (₹)', fontsize=12, fontweight='bold')
        ax.set_title('Scenario Comparison: Base vs Simulated Sales', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Detailed comparison table
        st.subheader("📊 Detailed Comparison")
        
        comparison_data = []
        for scenario in scenarios_to_show:
            diff = scenario['simulated_sales'] - scenario['base_sales']
            diff_pct = (diff / scenario['base_sales']) * 100
            
            comparison_data.append({
                'Scenario': scenario['name'],
                'Base Sales': f"₹{scenario['base_sales']:.2f}",
                'Simulated Sales': f"₹{scenario['simulated_sales']:.2f}",
                'Difference': f"₹{diff:.2f}",
                'Change %': f"{diff_pct:.1f}%",
                'Timestamp': scenario['timestamp'].strftime("%Y-%m-%d %H:%M")
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Best scenario recommendation
        st.markdown("---")
        st.subheader("🏆 Best Scenario")
        
        best_scenario = max(scenarios_to_show, key=lambda x: x['simulated_sales'])
        worst_scenario = min(scenarios_to_show, key=lambda x: x['simulated_sales'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"✅ **Best Performing:** {best_scenario['name']}")
            st.metric("Projected Sales", f"₹{best_scenario['simulated_sales']:.2f}")
            
            with st.expander("View Parameters"):
                for key, value in best_scenario['parameters'].items():
                    st.markdown(f"- **{key.replace('_', ' ').title()}**: {value}")
        
        with col2:
            st.error(f"⚠️ **Lowest Performing:** {worst_scenario['name']}")
            st.metric("Projected Sales", f"₹{worst_scenario['simulated_sales']:.2f}")
            
            with st.expander("View Parameters"):
                for key, value in worst_scenario['parameters'].items():
                    st.markdown(f"- **{key.replace('_', ' ').title()}**: {value}")
        
        # Clear scenarios
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear All Scenarios", use_container_width=True):
                st.session_state.comparison_scenarios = []
                st.rerun()
        
        with col2:
            if st.button("➕ Create New Scenario", use_container_width=True):
                st.session_state.page = "🔮 What-If Simulation"
                st.rerun()

# TREND ANALYSIS (NEW)
elif page == "📈 Trend Analysis":
    st.header("📈 Advanced Trend Analysis")
    
    try:
        df = pd.read_csv("enriched_sales_data.csv")
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Time period selector
        st.subheader("🗓️ Select Analysis Period")
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input("Start Date", df['Date'].min())
        with col2:
            end_date = st.date_input("End Date", df['Date'].max())
        
        # Filter data
        mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
        filtered_df = df[mask]
        
        if len(filtered_df) == 0:
            st.warning("⚠️ No data in selected period")
        else:
            # Tabs for different analyses
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📉 Patterns", "🔍 Anomalies", "📅 Seasonality"])
            
            with tab1:
                st.subheader("Sales Trend Overview")
                
                # Main trend chart
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(filtered_df['Date'], filtered_df['Sales'], color='#1f77b4', linewidth=2)
                
                # Add moving average
                if len(filtered_df) > 7:
                    filtered_df['MA7'] = filtered_df['Sales'].rolling(window=7).mean()
                    ax.plot(filtered_df['Date'], filtered_df['MA7'], color='red', 
                           linewidth=2, linestyle='--', label='7-Day Moving Average')
                
                ax.set_xlabel('Date', fontsize=12, fontweight='bold')
                ax.set_ylabel('Sales (₹)', fontsize=12, fontweight='bold')
                ax.set_title('Sales Trend with Moving Average', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Statistics
                st.markdown("### 📊 Key Statistics")
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                
                with stat_col1:
                    st.metric("Mean Sales", f"₹{filtered_df['Sales'].mean():.2f}")
                with stat_col2:
                    st.metric("Median Sales", f"₹{filtered_df['Sales'].median():.2f}")
                with stat_col3:
                    st.metric("Std Deviation", f"₹{filtered_df['Sales'].std():.2f}")
                with stat_col4:
                    growth = ((filtered_df['Sales'].iloc[-1] / filtered_df['Sales'].iloc[0]) - 1) * 100
                    st.metric("Period Growth", f"{growth:.1f}%")
            
            with tab2:
                st.subheader("Pattern Recognition")
                
                # Day of week analysis
                if len(filtered_df) > 14:
                    filtered_df['DayOfWeek'] = filtered_df['Date'].dt.day_name()
                    dow_avg = filtered_df.groupby('DayOfWeek')['Sales'].mean().reindex(
                        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    )
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    dow_avg.plot(kind='bar', ax=ax, color='#2ca02c', alpha=0.7)
                    ax.set_xlabel('Day of Week', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Average Sales (₹)', fontsize=12, fontweight='bold')
                    ax.set_title('Sales by Day of Week', fontsize=14, fontweight='bold')
                    ax.grid(True, axis='y', alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    best_day = dow_avg.idxmax()
                    worst_day = dow_avg.idxmin()
                    st.info(f"💡 **Best Day:** {best_day} (₹{dow_avg[best_day]:.2f}) | **Slowest Day:** {worst_day} (₹{dow_avg[worst_day]:.2f})")
                
                # Month analysis
                if len(filtered_df) > 30:
                    filtered_df['Month'] = filtered_df['Date'].dt.month_name()
                    month_avg = filtered_df.groupby('Month')['Sales'].mean()
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    month_avg.plot(kind='bar', ax=ax, color='#ff7f0e', alpha=0.7)
                    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Average Sales (₹)', fontsize=12, fontweight='bold')
                    ax.set_title('Sales by Month', fontsize=14, fontweight='bold')
                    ax.grid(True, axis='y', alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
            
            with tab3:
                st.subheader("Anomaly Detection")
                
                # Calculate z-scores
                mean_sales = filtered_df['Sales'].mean()
                std_sales = filtered_df['Sales'].std()
                filtered_df['zscore'] = (filtered_df['Sales'] - mean_sales) / std_sales
                
                anomalies = filtered_df[abs(filtered_df['zscore']) > 2]
                
                if len(anomalies) > 0:
                    st.warning(f"⚠️ Found {len(anomalies)} unusual sales days")
                    
                    # Plot with anomalies highlighted
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(filtered_df['Date'], filtered_df['Sales'], color='#1f77b4', linewidth=2, label='Normal')
                    ax.scatter(anomalies['Date'], anomalies['Sales'], color='red', s=100, 
                              marker='o', label='Anomalies', zorder=5)
                    
                    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Sales (₹)', fontsize=12, fontweight='bold')
                    ax.set_title('Sales Anomalies Detection', fontsize=14, fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Show anomaly details
                    st.dataframe(anomalies[['Date', 'Sales', 'zscore']].sort_values('zscore', ascending=False))
                else:
                    st.success("✅ No significant anomalies detected in this period")
            
            with tab4:
                st.subheader("Seasonality Analysis")
                
                if len(filtered_df) > 60:
                    # Decomposition visualization
                    st.info("💡 This analysis breaks down your sales into trend, seasonal, and random components")
                    
                    # Simple seasonal decomposition
                    filtered_df['Quarter'] = filtered_df['Date'].dt.quarter
                    quarterly = filtered_df.groupby('Quarter')['Sales'].mean()
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    quarterly.plot(kind='bar', ax=ax, color='#9467bd', alpha=0.7)
                    ax.set_xlabel('Quarter', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Average Sales (₹)', fontsize=12, fontweight='bold')
                    ax.set_title('Quarterly Sales Pattern', fontsize=14, fontweight='bold')
                    ax.grid(True, axis='y', alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    best_quarter = quarterly.idxmax()
                    st.success(f"🏆 Best Quarter: Q{best_quarter} with average sales of ₹{quarterly[best_quarter]:.2f}")
                else:
                    st.warning("⚠️ Need at least 60 days of data for seasonality analysis")
    
    except FileNotFoundError:
        st.error("❌ No data file found. Please upload data first.")
        if st.button("← Go to Upload", use_container_width=True):
            st.session_state.page = "📤 Upload Data"
            st.rerun()
    except Exception as e:
        st.error(f"⚠️ Analysis error: {e}")

# GOAL TRACKER (NEW)
elif page == "🎯 Goal Tracker":
    st.header("🎯 Sales Goal Tracker")
    
    try:
        df = pd.read_csv("enriched_sales_data.csv", parse_dates=["Date"])
        
        st.markdown("### Set Your Sales Target")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            target_period = st.selectbox("Time Period", ["Monthly", "Quarterly", "Yearly"])
        
        with col2:
            current_avg = df['Sales'].mean()
            target_sales = st.number_input("Sales Target (₹)", min_value=0.0, 
                                          value=float(current_avg * 1.2), step=1000.0)
        
        with col3:
            target_growth = ((target_sales / current_avg) - 1) * 100
            st.metric("Target Growth", f"{target_growth:.1f}%")
        
        # Current performance
        st.markdown("---")
        st.subheader("📊 Current Performance")
        
        recent_avg = df['Sales'].tail(30).mean() if len(df) >= 30 else df['Sales'].mean()
        progress = (recent_avg / target_sales) * 100
        
        # Progress bar
        st.progress(min(progress / 100, 1.0))
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Avg", f"₹{recent_avg:.2f}")
        with col2:
            st.metric("Target", f"₹{target_sales:.2f}")
        with col3:
            gap = target_sales - recent_avg
            st.metric("Gap", f"₹{gap:.2f}")
        with col4:
            st.metric("Progress", f"{progress:.1f}%")
        
        # Goal insights
        st.markdown("---")
        st.subheader("💡 Goal Insights")
        
        if progress >= 100:
            st.success("🎉 Congratulations! You've reached your target!")
        elif progress >= 80:
            st.info("📈 Great progress! You're almost there!")
        elif progress >= 50:
            st.warning("⚠️ Halfway there. Keep pushing!")
        else:
            st.error("🚨 Significant effort needed to reach target")
        
        # Action plan
        st.markdown("### 📋 Recommended Actions")
        
        if gap > 0:
            daily_increase = gap / 30
            st.markdown(f"- **Daily Target Increase**: ₹{daily_increase:.2f} per day")
            st.markdown(f"- **Weekly Target Increase**: ₹{daily_increase * 7:.2f} per week")
            
            if results:
                lstm_pred = float(results["LSTM"])
                if lstm_pred >= target_sales:
                    st.success("✅ Your forecast suggests you're on track to meet this goal!")
                else:
                    shortfall = target_sales - lstm_pred
                    st.warning(f"⚠️ Forecast shows a potential shortfall of ₹{shortfall:.2f}")
        
        # Historical goal comparison
        if len(df) > 60:
            st.markdown("---")
            st.subheader("📈 Goal vs Actual Trend")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df['Date'], df['Sales'], label='Actual Sales', color='#1f77b4', linewidth=2)
            ax.axhline(y=target_sales, color='red', linestyle='--', linewidth=2, label=f'Target: ₹{target_sales:.0f}')
            ax.fill_between(df['Date'], 0, target_sales, alpha=0.1, color='red')
            
            ax.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax.set_ylabel('Sales (₹)', fontsize=12, fontweight='bold')
            ax.set_title('Sales Performance vs Target', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
    
    except FileNotFoundError:
        st.error("❌ No data file found. Please upload data first.")
        if st.button("← Go to Upload", use_container_width=True):
            st.session_state.page = "📤 Upload Data"
            st.rerun()
    except Exception as e:
        st.error(f"⚠️ Error: {e}")

# BUSINESS INSIGHTS (Enhanced)
elif page == "💡 Business Insights":
    st.header("💡 AI-Powered Business Insights")
    
    try:
        df = pd.read_csv("enriched_sales_data.csv", parse_dates=["Date"])
        
        # Executive Summary
        st.subheader("📋 Executive Summary")
        
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            total_sales = df['Sales'].sum()
            st.metric("Total Sales", f"₹{total_sales:,.2f}")
        
        with summary_col2:
            if 'Profit' in df.columns:
                total_profit = df['Profit'].sum()
                profit_margin = (total_profit / total_sales) * 100
                st.metric("Total Profit", f"₹{total_profit:,.2f}", delta=f"{profit_margin:.1f}% margin")
            else:
                st.metric("Avg Sales", f"₹{df['Sales'].mean():.2f}")
        
        with summary_col3:
            growth = ((df['Sales'].iloc[-1] / df['Sales'].iloc[0]) - 1) * 100
            st.metric("Overall Growth", f"{growth:.1f}%")
        
        st.markdown("---")
        
        # Tabs for different insights
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "🔗 Correlations", "💰 Profitability", "🎯 Recommendations"])
        
        with tab1:
            st.subheader("Sales & Profit Performance")
            
            # Dual axis chart
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            color = '#1f77b4'
            ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Sales (₹)', color=color, fontsize=12, fontweight='bold')
            ax1.plot(df['Date'], df['Sales'], color=color, linewidth=2, label='Sales')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, alpha=0.3)
            
            if 'Profit' in df.columns:
                ax2 = ax1.twinx()
                color = '#ff7f0e'
                ax2.set_ylabel('Profit (₹)', color=color, fontsize=12, fontweight='bold')
                ax2.plot(df['Date'], df['Profit'], color=color, linewidth=2, linestyle='--', label='Profit')
                ax2.tick_params(axis='y', labelcolor=color)
            
            plt.title('Sales & Profit Trend', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Performance metrics
            st.markdown("### 📈 Period-over-Period Comparison")
            
            if len(df) >= 60:
                recent_30 = df['Sales'].tail(30).mean()
                previous_30 = df['Sales'].iloc[-60:-30].mean()
                change = ((recent_30 / previous_30) - 1) * 100
                
                perf_col1, perf_col2, perf_col3 = st.columns(3)
                
                with perf_col1:
                    st.metric("Last 30 Days", f"₹{recent_30:.2f}")
                with perf_col2:
                    st.metric("Previous 30 Days", f"₹{previous_30:.2f}")
                with perf_col3:
                    st.metric("Change", f"{change:.1f}%", delta=f"{change:.1f}%")
        
        with tab2:
            st.subheader("Factor Correlation Analysis")
            
            # Correlation heatmap
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, cmap='RdYlGn', center=0, 
                           square=True, linewidths=1, ax=ax, 
                           cbar_kws={"shrink": 0.8}, fmt='.2f')
                ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                
                st.caption("🧮 Values close to +1 (green) indicate strong positive relationships. Values close to -1 (red) indicate inverse relationships.")
                
                # Top correlations with Sales
                if 'Sales' in corr.columns:
                    st.markdown("### 🔍 Top Factors Influencing Sales")
                    sales_corr = corr['Sales'].drop('Sales').abs().sort_values(ascending=False)
                    
                    for i, (factor, value) in enumerate(sales_corr.head(5).items(), 1):
                        impact = "Strong" if value > 0.7 else "Moderate" if value > 0.4 else "Weak"
                        direction = "positive" if corr['Sales'][factor] > 0 else "negative"
                        st.markdown(f"{i}. **{factor}**: {impact} {direction} correlation ({value:.2f})")
        
        with tab3:
            if 'Profit' in df.columns:
                st.subheader("Profitability Analysis")
                
                # Profit metrics
                avg_profit = df['Profit'].mean()
                avg_sales = df['Sales'].mean()
                avg_margin = (avg_profit / avg_sales) * 100
                
                profit_col1, profit_col2, profit_col3 = st.columns(3)
                
                with profit_col1:
                    st.metric("Avg Profit", f"₹{avg_profit:.2f}")
                with profit_col2:
                    st.metric("Avg Margin", f"{avg_margin:.1f}%")
                with profit_col3:
                    best_margin = ((df['Profit'] / df['Sales']) * 100).max()
                    st.metric("Best Margin", f"{best_margin:.1f}%")
                
                # Profit distribution
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(df['Profit'], bins=30, color='#2ca02c', alpha=0.7, edgecolor='black')
                ax.axvline(avg_profit, color='red', linestyle='--', linewidth=2, label=f'Average: ₹{avg_profit:.2f}')
                ax.set_xlabel('Profit (₹)', fontsize=12, fontweight='bold')
                ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
                ax.set_title('Profit Distribution', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, axis='y', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Profit insights
                low_profit_days = len(df[df['Profit'] < avg_profit * 0.5])
                if low_profit_days > 0:
                    st.warning(f"⚠️ Found {low_profit_days} days with significantly low profit (< 50% of average)")
            else:
                st.info("💡 Include 'Profit' column in your data for profitability analysis")
        
        with tab4:
            st.subheader("🎯 Strategic Recommendations")
            
            # Generate AI recommendations based on data analysis
            recommendations = []
            
            # Sales trend recommendation
            recent_trend = df['Sales'].tail(10).mean() - df['Sales'].tail(20).iloc[:10].mean()
            if recent_trend > 0:
                recommendations.append(("📈", "Positive Momentum", 
                    f"Sales are trending upward (₹{recent_trend:.2f} increase). Maintain current strategies and consider scaling successful campaigns."))
            else:
                recommendations.append(("📉", "Declining Trend", 
                    f"Sales declining by ₹{abs(recent_trend):.2f}. Review recent changes and consider promotional activities."))
            
            # Volatility recommendation
            volatility = df['Sales'].std() / df['Sales'].mean()
            if volatility > 0.3:
                recommendations.append(("⚠️", "High Volatility", 
                    "Sales show high variability. Focus on consistent marketing and inventory management to stabilize revenue."))
            else:
                recommendations.append(("✅", "Stable Performance", 
                    "Sales are relatively stable. Good foundation for strategic growth initiatives."))
            
            # Seasonality recommendation
            if len(df) > 60:
                df['Month'] = df['Date'].dt.month
                monthly_var = df.groupby('Month')['Sales'].mean().std()
                if monthly_var > df['Sales'].mean() * 0.2:
                    recommendations.append(("🗓️", "Seasonal Patterns Detected", 
                        "Significant monthly variations observed. Plan inventory and staffing according to peak seasons."))
            
            # Inflation impact (if available)
            if 'InflationRate' in df.columns:
                avg_inflation = df['InflationRate'].mean()
                if avg_inflation > 7:
                    recommendations.append(("🌡️", "High Inflation Environment", 
                        f"Average inflation at {avg_inflation:.1f}%. Consider price adjustments or value-added services to maintain margins."))
                elif avg_inflation < 4:
                    recommendations.append(("💰", "Favorable Economic Conditions", 
                        f"Low inflation ({avg_inflation:.1f}%) creates opportunity for competitive pricing and market expansion."))
            
            # Promotion effectiveness (if available)
            if 'Promotion' in df.columns:
                promo_sales = df[df['Promotion'] == 1]['Sales'].mean()
                non_promo_sales = df[df['Promotion'] == 0]['Sales'].mean()
                if promo_sales > non_promo_sales * 1.1:
                    lift = ((promo_sales / non_promo_sales) - 1) * 100
                    recommendations.append(("🎯", "Promotions Are Effective", 
                        f"Promotions boost sales by {lift:.1f}%. Increase promotional frequency during slow periods."))
                else:
                    recommendations.append(("🔍", "Review Promotion Strategy", 
                        "Current promotions show limited impact. Consider testing different offers or targeting strategies."))
            
            # Profit margin recommendation (if available)
            if 'Profit' in df.columns:
                avg_margin = (df['Profit'].mean() / df['Sales'].mean()) * 100
                if avg_margin < 20:
                    recommendations.append(("💰", "Improve Profit Margins", 
                        f"Current margin at {avg_margin:.1f}%. Focus on cost reduction or premium positioning to improve profitability."))
                elif avg_margin > 35:
                    recommendations.append(("💎", "Strong Margins", 
                        f"Excellent margins at {avg_margin:.1f}%. Consider strategic discounting to capture market share."))
            
            # Display recommendations
            for icon, title, description in recommendations:
                st.markdown(f"### {icon} {title}")
                st.markdown(f'<div class="insight-box">{description}</div>', unsafe_allow_html=True)
            
            # Action items
            st.markdown("---")
            st.markdown("### ✅ Immediate Action Items")
            
            action_items = [
                "Review and optimize underperforming periods",
                "Analyze top-performing days/periods and replicate success factors",
                "Adjust inventory levels based on forecast predictions",
                "Test new promotional strategies in controlled scenarios",
                "Monitor key metrics weekly and adjust strategies accordingly"
            ]
            
            for item in action_items:
                st.markdown(f"- {item}")
    
    except FileNotFoundError:
        st.error("❌ No data file found. Please upload data first.")
        if st.button("← Go to Upload", use_container_width=True):
            st.session_state.page = "📤 Upload Data"
            st.rerun()
    except Exception as e:
        st.error(f"⚠️ Error generating insights: {e}")

# USER GUIDE (NEW)
elif page == "📚 User Guide":
    st.header("📚 Complete User Guide")
    
    # Quick navigation
    st.markdown("### 📑 Table of Contents")
    guide_section = st.selectbox("Jump to section:", [
        "Getting Started",
        "Uploading Data",
        "Understanding Forecasts",
        "What-If Simulations",
        "Comparing Scenarios",
        "Trend Analysis",
        "Setting Goals",
        "Reading Insights",
        "Troubleshooting",
        "Best Practices"
    ])
    
    st.markdown("---")
    
    if guide_section == "Getting Started":
        st.subheader("🚀 Getting Started")
        st.markdown("""
        Welcome to the AI Business Forecasting Tool! This guide will help you get the most out of the platform.
        
        ### What You'll Need
        - A CSV file with your historical sales data
        - At minimum: **Date** and **Sales** columns
        - Ideally: 90+ days of historical data
        
        ### Quick Start Steps
        1. **Upload Data** → Navigate to the Upload Data page
        2. **Run Forecasts** → Click "Run Forecasts" to generate predictions
        3. **Explore Results** → View your forecast dashboard
        4. **Test Scenarios** → Try different business strategies
        5. **Take Action** → Implement data-driven decisions
        
        ### System Requirements
        - Modern web browser (Chrome, Firefox, Safari, Edge)
        - Stable internet connection
        - CSV file under 50MB
        """)
    
    elif guide_section == "Uploading Data":
        st.subheader("📤 Uploading Data")
        st.markdown("""
        ### File Format Requirements
        
        **Required Columns:**
        - `Date`: Any standard date format (YYYY-MM-DD, MM/DD/YYYY, etc.)
        - `Sales`: Numeric values representing your sales figures
        
        **Optional Columns (Improve Accuracy):**
        - `Profit`: Profit figures
        - `Category`: Product categories
        - `Region`: Geographic regions
        - `Promotion`: Binary (0/1) indicating promotional periods
        - `Holiday`: Binary (0/1) for holidays
        - `InflationRate`: Economic inflation percentage
        - `AdSpend`: Advertising expenditure
        
        ### Data Quality Tips
        - Remove duplicate entries
        - Ensure dates are in chronological order
        - Fill or remove missing values
        - Use consistent formatting throughout
        - Include at least 90 days for better predictions
        
        ### Common Issues
        - **"Date column not found"**: Check spelling and case sensitivity
        - **"Invalid date format"**: Convert dates to standard format
        - **"Missing values"**: Fill gaps or remove incomplete rows
        """)
        
        # Show sample
        sample = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Sales': [15000, 18000, 16500],
            'Profit': [4500, 5400, 4950]
        })
        st.markdown("### Sample Format")
        st.dataframe(sample)
    
    elif guide_section == "Understanding Forecasts":
        st.subheader("📊 Understanding Forecasts")
        st.markdown("""
        ### Three Forecasting Models
        
        #### 1️⃣ Prophet (Time Series)
        - **Best for**: Long-term trends and seasonality
        - **Strengths**: Captures holidays, weekly patterns, yearly cycles
        - **Use when**: Planning 30+ days ahead
        - **Output**: Daily predictions with confidence intervals
        
        #### 2️⃣ LSTM (Deep Learning)
        - **Best for**: Short-term predictions
        - **Strengths**: Learns complex patterns, adapts to recent changes
        - **Use when**: Forecasting next month's performance
        - **Output**: Single period prediction
        
        #### 3️⃣ AutoML (Ensemble)
        - **Best for**: Overall accuracy
        - **Strengths**: Combines multiple models automatically
        - **Use when**: You need the most reliable forecast
        - **Output**: Predictions with accuracy metrics (MAE)
        
        ### Reading the Charts
        - **Solid lines**: Actual predictions
        - **Dashed lines**: Moving averages or trends
        - **Shaded areas**: Confidence intervals (uncertainty range)
        - **Red markers**: Anomalies or unusual data points
        
        ### Understanding Metrics
        - **MAE (Mean Absolute Error)**: Average prediction error - lower is better
        - **Confidence Interval**: Range where actual value likely falls
        - **Growth %**: Percentage change from baseline
        - **Volatility**: How much values fluctuate (standard deviation)
        """)
    
    elif guide_section == "What-If Simulations":
        st.subheader("🔮 What-If Simulations")
        st.markdown("""
        ### How Simulations Work
        Simulations let you test different business scenarios before implementing them in real life.
        
        ### Available Parameters
        
        **Financial Factors:**
        - **Ad Budget**: Marketing spend impact on visibility
        - **Discount Rate**: Price reduction effects on volume
        - **Inflation**: Economic pressure on consumer spending
        
        **Operational Factors:**
        - **Staffing Level**: Service quality and capacity
        - **Digital Marketing**: Online reach and engagement
        - **Inventory Level**: Stock availability
        
        **External Factors:**
        - **Holiday Boost**: Seasonal demand spikes
        - **Competitor Activity**: Market competition impact
        - **Weather Impact**: Environmental effects on sales
        
        ### Best Practices
        1. Start with baseline (current) values
        2. Change one parameter at a time to see individual impact
        3. Save successful scenarios for comparison
        4. Test extreme scenarios to understand limits
        5. Use simulations to support budget requests
        
        ### Interpreting Results
        - **Green metrics**: Positive impact vs baseline
        - **Red metrics**: Negative impact vs baseline
        - **ROI Estimate**: Return on investment for ad spend
        - **Percentage change**: Relative impact magnitude
        """)
    
    elif guide_section == "Comparing Scenarios":
        st.subheader("⚖️ Comparing Scenarios")
        st.markdown("""
        ### How to Compare Scenarios
        
        1. Create multiple scenarios in What-If Simulation
        2. Save each scenario with a descriptive name
        3. Navigate to Scenario Comparison
        4. Review side-by-side performance
        
        ### What to Look For
        - **Highest projected sales**: Best case scenario
        - **Most realistic**: Balance of achievability and results
        - **Best ROI**: Highest return for investment
        - **Lowest risk**: Minimal investment, decent returns
        
        ### Use Cases
        - Budget allocation decisions
        - Marketing strategy selection
        - Seasonal planning
        - Risk assessment
        - Board presentations
        
        ### Tips
        - Compare maximum 5 scenarios at once for clarity
        - Use consistent naming conventions
        - Document assumptions for each scenario
        - Share comparison charts in reports
        """)
    
    elif guide_section == "Trend Analysis":
        st.subheader("📈 Trend Analysis")
        st.markdown("""
        ### Types of Analysis Available
        
        #### Overview
        - Sales trends over time
        - Moving averages
        - Key statistics (mean, median, std dev)
        - Period growth rates
        
        #### Patterns
        - **Day of Week**: Which days perform best
        - **Monthly**: Seasonal patterns throughout year
        - **Quarterly**: Business cycle performance
        
        #### Anomalies
        - Unusual sales days (Z-score > 2)
        - Potential data quality issues
        - Special events or outliers
        
        #### Seasonality
        - Recurring patterns
        - Peak and low periods
        - Predictable cycles
        
        ### Using Trend Insights
        - **Staffing**: Schedule more staff on busy days
        - **Inventory**: Stock up before peak periods
        - **Marketing**: Boost ads during slow periods
        - **Planning**: Anticipate seasonal changes
        """)
    
    elif guide_section == "Setting Goals":
        st.subheader("🎯 Setting Goals")
        st.markdown("""
        ### How to Set Effective Goals
        
        #### SMART Goals Framework
        - **Specific**: Clear target amount
        - **Measurable**: Tracked with metrics
        - **Achievable**: Within realistic reach
        - **Relevant**: Aligned with business objectives
        - **Time-bound**: Monthly/Quarterly/Yearly
        
        ### Recommended Targets
        - **Conservative**: 5-10% above current average
        - **Moderate**: 15-20% growth
        - **Aggressive**: 25%+ growth (requires significant changes)
        
        ### Tracking Progress
        - Green (80%+): On track
        - Yellow (50-80%): Needs attention
        - Red (<50%): Urgent action required
        
        ### Action Planning
        The tool calculates:
        - Daily increase needed
        - Weekly targets
        - Forecast vs goal comparison
        - Strategic recommendations
        """)
    
    elif guide_section == "Reading Insights":
        st.subheader("💡 Reading Insights")
        st.markdown("""
        ### Types of Insights Provided
        
        #### Performance Insights
        - Period-over-period comparison
        - Growth trends
        - Best/worst performing periods
        
        #### Correlation Insights
        - Which factors most influence sales
        - Positive vs negative relationships
        - Strength of relationships (-1 to +1)
        
        #### Profitability Insights
        - Margin analysis
        - Profit distribution
        - Low-profit day identification
        
        #### Strategic Recommendations
        - Data-driven action items
        - Opportunity identification
        - Risk warnings
        
        ### How to Act on Insights
        1. **Prioritize**: Focus on high-impact recommendations
        2. **Test**: Use simulations to validate strategies
        3. **Implement**: Roll out changes systematically
        4. **Monitor**: Track results weekly
        5. **Iterate**: Adjust based on performance
        """)
    
    elif guide_section == "Troubleshooting":
        st.subheader("🔧 Troubleshooting")
        st.markdown("""
        ### Common Issues & Solutions
        
        #### Upload Problems
        **Issue**: "Date column not found"
        - **Solution**: Check column name is exactly "Date" (case-sensitive)
        
        **Issue**: "Invalid date format"
        - **Solution**: Convert dates to YYYY-MM-DD format in Excel
        
        **Issue**: "File too large"
        - **Solution**: Compress data or split into smaller time periods
        
        #### Forecast Problems
        **Issue**: Forecasts seem inaccurate
        - **Solution**: Ensure at least 90 days of clean data
        - Check for and remove outliers or errors
        - Include optional columns for better predictions
        
        **Issue**: "Forecast failed" error
        - **Solution**: Check data has no missing values
        - Ensure Sales column has numeric values only
        - Try removing special characters
        
        #### Performance Issues
        **Issue**: App running slowly
        - **Solution**: Reduce data size or date range
        - Close unnecessary browser tabs
        - Clear browser cache
        
        #### Visualization Problems
        **Issue**: Charts not displaying
        - **Solution**: Refresh the page
        - Check internet connection
        - Try different browser
        
        ### Getting Help
        If issues persist:
        1. Check the data format requirements
        2. Review sample CSV format
        3. Ensure all required columns present
        4. Contact support with error message
        """)
    
    elif guide_section == "Best Practices":
        st.subheader("✨ Best Practices")
        st.markdown("""
        ### Data Management
        - ✅ Update data regularly (weekly/monthly)
        - ✅ Maintain consistent formatting
        - ✅ Document any unusual events in your data
        - ✅ Keep backup copies of original files
        - ✅ Validate data before uploading
        
        ### Forecasting
        - ✅ Use multiple models for comparison
        - ✅ Consider confidence intervals
        - ✅ Update forecasts monthly
        - ✅ Combine AI predictions with business knowledge
        - ✅ Document assumptions
        
        ### Scenario Planning
        - ✅ Test realistic ranges (not extremes)
        - ✅ Save scenarios with descriptive names
        - ✅ Review scenarios quarterly
        - ✅ Share scenarios with stakeholders
        - ✅ Track actual results vs scenarios
        
        ### Decision Making
        - ✅ Use data to support (not replace) judgment
        - ✅ Consider multiple scenarios
        - ✅ Start with small tests before full rollout
        - ✅ Monitor KPIs after implementing changes
        - ✅ Be prepared to adjust strategies
        
        ### Reporting
        - ✅ Download summary reports regularly
        - ✅ Share insights with team members
        - ✅ Present visual charts in meetings
        - ✅ Track progress against goals
        - ✅ Celebrate wins and learn from misses
        
        ### Continuous Improvement
        - ✅ Review forecast accuracy monthly
        - ✅ Identify which models work best for you
        - ✅ Refine scenarios based on outcomes
        - ✅ Add new data sources when available
        - ✅ Stay updated on new features
        """)
    
    st.markdown("---")
    
    # Quick links
    st.markdown("### 🔗 Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📤 Upload Data", use_container_width=True):
            st.session_state.page = "📤 Upload Data"
            st.rerun()
    
    with col2:
        if st.button("📊 View Dashboard", use_container_width=True):
            st.session_state.page = "📊 Forecast Dashboard"
            st.rerun()
    
    with col3:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = "🏠 Welcome"
            st.rerun()

# GLOBAL FEATURES - Compare Models (Available on all pages after forecast)
if results is not None and page not in ["🏠 Welcome", "📤 Upload Data", "📚 User Guide"]:
    with st.expander("🔍 Quick Model Comparison", expanded=False):
        st.markdown("### Side-by-Side Model Performance")
        
        try:
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            
            with comp_col1:
                st.markdown("**Prophet**")
                prophet_avg = results["Prophet"]["yhat"].mean()
                st.metric("Avg Forecast", f"₹{prophet_avg:.2f}")
                st.caption("Best for long-term trends")
            
            with comp_col2:
                st.markdown("**LSTM**")
                lstm_pred = float(results["LSTM"])
                st.metric("Next Month", f"₹{lstm_pred:.2f}")
                st.caption("Best for short-term")
            
            with comp_col3:
                st.markdown("**AutoML**")
                automl_mae = results["AutoML"]["mae"]
                st.metric("MAE", f"{automl_mae:.2f}")
                st.caption("Most accurate overall")
        except:
            st.warning("⚠️ Some model results unavailable")

# GLOBAL FEATURES - Download Summary Report
if results is not None and page not in ["🏠 Welcome", "📚 User Guide"]:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📥 Export")
    
    try:
        df = pd.read_csv("enriched_sales_data.csv", parse_dates=["Date"])
        
        summary = io.StringIO()
        summary.write("="*50 + "\n")
        summary.write("AI BUSINESS FORECASTING TOOL - SUMMARY REPORT\n")
        summary.write("="*50 + "\n\n")
        summary.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        summary.write("DATA OVERVIEW\n")
        summary.write("-"*50 + "\n")
        summary.write(f"Total Records: {len(df)}\n")
        summary.write(f"Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}\n")
        summary.write(f"Average Sales: ₹{df['Sales'].mean():.2f}\n")
        summary.write(f"Total Sales: ₹{df['Sales'].sum():.2f}\n\n")
        
        if "Profit" in df.columns:
            summary.write(f"Average Profit: ₹{df['Profit'].mean():.2f}\n")
            summary.write(f"Profit Margin: {(df['Profit'].mean()/df['Sales'].mean()*100):.1f}%\n\n")
        
        summary.write("FORECAST RESULTS\n")
        summary.write("-"*50 + "\n")
        
        try:
            summary.write(f"Prophet Average: ₹{results['Prophet']['yhat'].mean():.2f}\n")
            summary.write(f"LSTM Prediction: ₹{float(results['LSTM']):.2f}\n")
            summary.write(f"AutoML MAE: {results['AutoML']['mae']:.2f}\n\n")
        except:
            summary.write("Forecast details unavailable\n\n")
        
        if "InflationRate" in df.columns:
            summary.write("ECONOMIC FACTORS\n")
            summary.write("-"*50 + "\n")
            summary.write(f"Average Inflation: {df['InflationRate'].mean():.2f}%\n\n")
        
        summary.write("KEY INSIGHTS\n")
        summary.write("-"*50 + "\n")
        recent_avg = df['Sales'].tail(30).mean()
        overall_avg = df['Sales'].mean()
        if recent_avg > overall_avg:
            summary.write("✓ Recent sales trending above average\n")
        else:
            summary.write("⚠ Recent sales below average - review strategy\n")
        
        volatility = df['Sales'].std() / df['Sales'].mean()
        if volatility > 0.3:
            summary.write("⚠ High sales volatility detected\n")
        else:
            summary.write("✓ Sales relatively stable\n")
        
        summary.write("\n" + "="*50 + "\n")
        summary.write("End of Report\n")
        summary.write("="*50 + "\n")
        
        st.sidebar.download_button(
            label="📄 Download Report",
            data=summary.getvalue(),
            file_name=f"forecast_report_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    except:
        pass

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info("AI Business Forecasting Tool v2.0\n\nPowered by Prophet, LSTM & AutoML")

if st.sidebar.button("🔄 Reset Everything", use_container_width=True):
    if st.sidebar.checkbox("⚠️ Confirm reset"):
        st.session_state.clear()
        st.rerun()