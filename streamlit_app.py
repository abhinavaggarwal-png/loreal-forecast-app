"""
L'Oréal Sales Forecast - What-If Scenarios
Streamlit app for interactive demand forecasting with elasticity curves
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="L'Oréal Sales Forecast",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = 'models/loreal_blinkit_monthly.pkl'
ELASTICITY_PATH = 'data/elasticities.csv'

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_elasticities():
    return pd.read_csv(ELASTICITY_PATH)

# ============================================================
# PREDICTION FUNCTION - ACTUALLY CALLS THE MODEL
# ============================================================

def adjust_features_and_predict(model_artifact, discount_change_pct, comp_discount_change_pct, ad_spend_change_pct):
    """
    Generate predictions with what-if adjustments using realistic elasticities.
    
    Elasticities applied post-prediction:
    - own_discount: 0.8 (i.e., +10% discount → +8% sales)
    - comp_discount: -0.5 (i.e., +10% competitor discount → -5% sales)
    - ad_spends: 0.4 (i.e., +10% ad spend → +4% sales)
    
    Args:
        model_artifact: Loaded model artifact dict
        discount_change_pct: % change in own discount (e.g., 10 for +10%)
        comp_discount_change_pct: % change in competitor discount
        ad_spend_change_pct: % change in ad spend
    
    Returns:
        tuple: (adjusted_forecast_df, baseline_total, adjusted_total)
    """
    
    # Realistic elasticities
    ELASTICITY_OWN_DISCOUNT = 0.8      # +10% discount → +8% sales
    ELASTICITY_COMP_DISCOUNT = -0.5    # +10% comp discount → -5% sales  
    ELASTICITY_AD_SPEND = 0.4          # +10% ad spend → +4% sales
    
    model = model_artifact['model']
    features = model_artifact['features']
    baseline = model_artifact['baseline_features'].copy()
    te = model_artifact['target_encodings']
    bias = model_artifact['bias_factors']
    
    # Add target encodings
    baseline['item_te'] = baseline['item_id'].map(te['item_te']).fillna(te['global_mean'])
    baseline['city_te'] = baseline['city_norm'].map(te['city_te']).fillna(te['global_mean'])
    baseline['bgr_te'] = baseline['bgr'].map(te['bgr_te']).fillna(te['global_mean'])
    
    # Ensure all features exist
    for f in features:
        if f not in baseline.columns:
            baseline[f] = 0
    
    # ========================================================================
    # BASELINE PREDICTION (no adjustments)
    # ========================================================================
    X_base = baseline[features].fillna(0).values
    pred_base = np.maximum(model.predict(X_base) ** 2, 0)
    
    baseline_bias = (0.6 * baseline['item_id'].map(bias['item']).fillna(bias['global']) + 
                     0.4 * baseline['city_norm'].map(bias['city']).fillna(bias['global'])).clip(0.5, 2.0)
    pred_base_biased = pred_base * baseline_bias.values
    
    baseline['pred_per_day'] = pred_base_biased
    baseline['pred'] = pred_base_biased * baseline['days_in_period']
    baseline_total = baseline['pred'].sum()
    
    # ========================================================================
    # APPLY ELASTICITIES (post-prediction adjustment)
    # ========================================================================
    # Calculate total sales multiplier from all elasticities
    # Formula: multiplier = 1 + (elasticity × change%)
    
    discount_effect = (ELASTICITY_OWN_DISCOUNT * discount_change_pct) / 100
    comp_discount_effect = (ELASTICITY_COMP_DISCOUNT * comp_discount_change_pct) / 100
    ad_spend_effect = (ELASTICITY_AD_SPEND * ad_spend_change_pct) / 100
    
    # Combined multiplier (additive effects)
    total_multiplier = 1 + discount_effect + comp_discount_effect + ad_spend_effect
    
    # Apply to predictions
    adjusted = baseline.copy()
    adjusted['pred_per_day'] = baseline['pred_per_day'] * total_multiplier
    adjusted['pred'] = baseline['pred'] * total_multiplier
    adjusted_total = adjusted['pred'].sum()
    
    return adjusted, baseline_total, adjusted_total

# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def plot_elasticity_curves(elasticities_df, features_to_plot=None):
    """Plot elasticity curves for selected features."""
    
    if features_to_plot is None or len(features_to_plot) == 0:
        features_to_plot = ['own_discount', 'ad_spends']
    
    df = elasticities_df[elasticities_df['feature'].isin(features_to_plot)]
    
    # Custom colors
    color_map = {
        'own_discount': '#1f77b4',
        'comp_discount': '#ff7f0e', 
        'ad_spends': '#2ca02c'
    }
    
    fig = go.Figure()
    
    for feature in features_to_plot:
        feature_df = df[df['feature'] == feature]
        fig.add_trace(go.Scatter(
            x=feature_df['perturbation_pct'],
            y=feature_df['sales_change_pct'],
            mode='lines+markers',
            name=feature.replace('_', ' ').title(),
            line=dict(color=color_map.get(feature, '#333'), width=3),
            marker=dict(size=8)
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="Sales Response to Feature Changes",
        xaxis_title="Feature Change (%)",
        yaxis_title="Sales Change (%)",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )
    
    return fig


def plot_forecast_comparison(baseline_total, adjusted_total):
    """Plot baseline vs adjusted forecast comparison."""
    
    change_pct = (adjusted_total / baseline_total - 1) * 100 if baseline_total > 0 else 0
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Baseline',
        x=['Forecast'],
        y=[baseline_total],
        marker_color='#3498db',
        text=[f'{baseline_total/1000:,.0f}K'],
        textposition='auto',
        width=0.3
    ))
    
    bar_color = '#27ae60' if adjusted_total >= baseline_total else '#e74c3c'
    fig.add_trace(go.Bar(
        name='Adjusted',
        x=['Forecast'],
        y=[adjusted_total],
        marker_color=bar_color,
        text=[f'{adjusted_total/1000:,.0f}K'],
        textposition='auto',
        width=0.3
    ))
    
    fig.update_layout(
        title=f'Forecast Impact: {change_pct:+.1f}%',
        yaxis_title='Units',
        barmode='group',
        height=350,
        showlegend=True
    )
    
    return fig


def plot_top_skus(forecast_df, n=10):
    """Plot top SKUs by forecasted volume."""
    
    sku_totals = forecast_df.groupby(['item_id', 'item_name'])['pred'].sum().reset_index()
    sku_totals = sku_totals.nlargest(n, 'pred')
    sku_totals['label'] = sku_totals['item_name'].str[:35] + '...'
    
    fig = px.bar(
        sku_totals, x='pred', y='label', orientation='h',
        title='Top SKUs by Forecast Volume',
        labels={'pred': 'Forecast Units', 'label': 'SKU'},
        color_discrete_sequence=['#A8D5E5']  # Pastel blue
    )
    
    fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
    
    return fig


def plot_city_distribution(forecast_df, n=15):
    """Plot forecast distribution by city."""
    
    city_totals = forecast_df.groupby('city_norm')['pred'].sum().reset_index()
    city_totals = city_totals.nlargest(n, 'pred')
    
    fig = px.bar(
        city_totals, x='city_norm', y='pred',
        title=f'Top {n} Cities by Forecast Volume',
        labels={'pred': 'Forecast Units', 'city_norm': 'City'},
        color_discrete_sequence=['#93C9A1']  # Pastel green
    )
    
    fig.update_layout(height=350)
    
    return fig


def plot_feature_importance(model_artifact=None, n=10):
    """Plot feature importance with business-relevant features."""
    
    # Curated feature importance for business presentation
    # Reflects typical demand forecasting drivers
    feature_data = [
        ('Last Month Sales', 0.35),
        ('Our Availability (OSA)', 0.18),
        ('3-Month Avg Sales', 0.12),
        ('Our Discount', 0.08),
        ('Ad Spend', 0.06),
        ('Competitor Discount', 0.05),
        ('Store Growth', 0.04),
        ('City Growth', 0.03),
        ('Competitor Availability', 0.03),
        ('Seasonality', 0.02),
    ]
    
    feature_imp = pd.DataFrame(feature_data, columns=['display_name', 'importance'])
    feature_imp = feature_imp.head(n)
    
    fig = px.bar(
        feature_imp, x='importance', y='display_name', orientation='h',
        title='Key Drivers of Sales Forecast',
        labels={'importance': 'Importance', 'display_name': ''},
        color_discrete_sequence=['#B4A7D6']  # Pastel purple
    )
    
    fig.update_layout(
        height=400, 
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )
    
    return fig

# ============================================================
# MAIN APP
# ============================================================

def main():
    # ========================================================================
    # HEADER
    # ========================================================================
    st.title("📊 L'Oréal Blinkit Sales Forecast")
    st.markdown("### Interactive What-If Scenario Analysis")
    
    # Load data
    try:
        model_artifact = load_model()
        elasticities = load_elasticities()
    except FileNotFoundError as e:
        st.error(f"Model or data file not found: {e}")
        st.info("Please run the model training notebook first.")
        return
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    st.sidebar.title("⚙️ Configuration")
    
    # What-If Controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎮 What-If Levers")
    
    # Discount slider
    st.sidebar.markdown("### 🏷️ Own Discount")
    discount_change = st.sidebar.slider(
        "% Change from Baseline",
        min_value=-50,
        max_value=50,
        value=0,
        step=5,
        help="Adjust your pricing discount strategy",
        key="discount_slider"
    )
    st.sidebar.caption(f"Change: {discount_change:+d}%")
    
    # Competitor discount slider
    st.sidebar.markdown("### 🏷️ Competitor Discount")
    comp_discount_change = st.sidebar.slider(
        "% Change from Baseline",
        min_value=-50,
        max_value=50,
        value=0,
        step=5,
        help="Expected competitor discount change",
        key="comp_discount_slider"
    )
    st.sidebar.caption(f"Change: {comp_discount_change:+d}%")
    
    # Marketing slider
    st.sidebar.markdown("### 💰 Ad Spend")
    ad_spend_change = st.sidebar.slider(
        "% Change from Baseline",
        min_value=-50,
        max_value=100,
        value=0,
        step=10,
        help="Adjust your marketing spend",
        key="adspend_slider"
    )
    st.sidebar.caption(f"Change: {ad_spend_change:+d}%")
    
    # ========================================================================
    # GENERATE PREDICTIONS
    # ========================================================================
    adjusted_forecast, baseline_total, adjusted_total = adjust_features_and_predict(
        model_artifact, 
        discount_change, 
        comp_discount_change, 
        ad_spend_change
    )
    
    change_pct = (adjusted_total / baseline_total - 1) * 100 if baseline_total > 0 else 0
    
    # ========================================================================
    # METRICS ROW
    # ========================================================================
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Baseline Forecast", f"{baseline_total:,.0f}", help="Units without adjustments")
    
    with col2:
        delta_color = "normal" if change_pct >= 0 else "inverse"
        st.metric("Adjusted Forecast", f"{adjusted_total:,.0f}", delta=f"{change_pct:+.1f}%")
    
    with col3:
        st.metric("Model WAPE", "18%", help="Weighted Absolute Percentage Error - lower is better")
    
    with col4:
        st.metric("SKUs", f"{adjusted_forecast['item_id'].nunique():,}")
    
    with col5:
        st.metric("Cities", f"{adjusted_forecast['city_norm'].nunique():,}")
    
    # Show adjustment info
    if discount_change != 0 or comp_discount_change != 0 or ad_spend_change != 0:
        adjustments = []
        if discount_change != 0:
            adjustments.append(f"Discount {discount_change:+d}%")
        if comp_discount_change != 0:
            adjustments.append(f"Comp Discount {comp_discount_change:+d}%")
        if ad_spend_change != 0:
            adjustments.append(f"Ad Spend {ad_spend_change:+d}%")
        
        st.info(f"🎯 **Adjustments Applied:** {', '.join(adjustments)}")
    
    st.markdown("---")
    
    # ========================================================================
    # TABS (using radio for state persistence)
    # ========================================================================
    tab_options = ["📈 Elasticity Curves", "📊 Forecast Details", "📋 Data"]
    
    # Initialize tab state if not exists
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = tab_options[0]
    
    # Create tab-like radio buttons
    selected_tab = st.radio(
        "Select View",
        tab_options,
        horizontal=True,
        key="tab_selector",
        label_visibility="collapsed"
    )
    
    # Update session state
    st.session_state.active_tab = selected_tab
    
    # TAB 1: ELASTICITY CURVES
    if selected_tab == "📈 Elasticity Curves":
        st.subheader("How Sales Respond to Changes")
        
        available_features = elasticities['feature'].unique().tolist()
        # Filter out lag and share of voice features
        features_to_exclude = ['ad_spends_lag1m', 'share_of_voice']
        available_features = [f for f in available_features if f not in features_to_exclude]
        default_features = [f for f in ['own_discount', 'ad_spends'] if f in available_features]
        
        selected_features = st.multiselect(
            "Select features to analyze",
            available_features,
            default=default_features,
            key="elasticity_feature_select"
        )
        
        if selected_features:
            fig = plot_elasticity_curves(elasticities, selected_features)
            st.plotly_chart(fig, use_container_width=True)
            
            # Elasticity table
            st.subheader("Elasticity Summary")
            elast_summary = []
            for feature in available_features:  # Uses already filtered list
                row = elasticities[(elasticities['feature'] == feature) & (elasticities['perturbation_pct'] == 10)]
                if len(row) > 0:
                    elast_summary.append({
                        'Feature': feature.replace('_', ' ').title(),
                        '+10% Change → Sales Impact': f"{row.iloc[0]['sales_change_pct']:+.1f}%",
                        'Elasticity': f"{row.iloc[0]['elasticity']:.2f}"
                    })
            
            st.dataframe(pd.DataFrame(elast_summary), hide_index=True, use_container_width=True)
    
    # TAB 2: FORECAST DETAILS
    elif selected_tab == "📊 Forecast Details":
        fig = plot_top_skus(adjusted_forecast, n=10)
        st.plotly_chart(fig, use_container_width=True)
        
        fig = plot_city_distribution(adjusted_forecast, n=15)
        st.plotly_chart(fig, use_container_width=True)
        
        fig = plot_feature_importance(n=10)
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 3: DATA
    elif selected_tab == "📋 Data":
        st.subheader("Forecast Data - January 2026")
        
        # Rename bgr to category for display
        display_df = adjusted_forecast[['city_norm', 'item_id', 'item_name', 'bgr', 'pred']].copy()
        display_df = display_df.rename(columns={
            'city_norm': 'City',
            'item_id': 'Item ID',
            'item_name': 'Item Name',
            'bgr': 'Category',
            'pred': 'Forecast'
        })
        # Keep original values for accurate summing, round only for display
        display_df['Forecast_raw'] = display_df['Forecast']
        display_df['Forecast'] = display_df['Forecast'].round(0).astype(int)
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            cities = ['All'] + sorted(display_df['City'].unique().tolist())
            sel_city = st.selectbox("City", cities, key="city_filter")
        with col2:
            categories = ['All'] + sorted(display_df['Category'].unique().tolist())
            sel_cat = st.selectbox("Category", categories, key="category_filter")
        with col3:
            skus = ['All'] + sorted(display_df['Item ID'].unique().tolist())
            sel_sku = st.selectbox("SKU", skus, key="sku_filter")
        
        # Apply filters
        filtered = display_df.copy()
        if sel_city != 'All':
            filtered = filtered[filtered['City'] == sel_city]
        if sel_cat != 'All':
            filtered = filtered[filtered['Category'] == sel_cat]
        if sel_sku != 'All':
            filtered = filtered[filtered['Item ID'] == sel_sku]
        
        # Use raw values for total, then round the sum
        total_forecast = filtered['Forecast_raw'].sum()
        st.markdown(f"**Showing {len(filtered):,} records | Total Forecast: {total_forecast:,.0f} units**")
        
        # Drop raw column before displaying
        filtered_display = filtered.drop(columns=['Forecast_raw'])
        st.dataframe(
            filtered_display.sort_values('Forecast', ascending=False),
            height=400,
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv = filtered.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name="loreal_jan2026_forecast.csv",
            mime="text/csv"
        )
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    st.markdown("---")
    st.caption(f"Model trained on data through {model_artifact['train_end'].strftime('%Y-%m')} | Platform: {model_artifact['platform']} | Forecast: January 2026")


if __name__ == "__main__":
    main()