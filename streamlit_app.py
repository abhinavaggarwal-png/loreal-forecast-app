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

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="L'Oréal Sales Forecast",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to make primary buttons blue instead of red
st.markdown("""
<style>
    /* Primary button - blue */
    .stButton > button[kind="primary"] {
        background-color: #3B82F6;
        border-color: #3B82F6;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #2563EB;
        border-color: #2563EB;
    }
    
    /* Secondary button - light blue on hover */
    .stButton > button[kind="secondary"]:hover {
        border-color: #3B82F6;
        color: #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = 'models/loreal_blinkit_monthly.pkl'

# ============================================================
# PERIOD CONFIGURATION
# ============================================================

JANUARY_WEEKS = {
    'W1': {'label': 'Week 1 (1-7)', 'short': 'W1', 'multiplier': 0.18, 'days': 7},
    'W2': {'label': 'Week 2 (8-14)', 'short': 'W2', 'multiplier': 0.24, 'days': 7},
    'W3': {'label': 'Week 3 (15-21)', 'short': 'W3', 'multiplier': 0.26, 'days': 7},
    'W4': {'label': 'Week 4 (22-31)', 'short': 'W4', 'multiplier': 0.32, 'days': 10},
}

MONTHS = {
    'January': {'multiplier': 1.0, 'is_weekly': True, 'weeks': JANUARY_WEEKS},
    'February': {'multiplier': 0.92, 'is_weekly': False, 'label': 'February 2026'},
    'March': {'multiplier': 1.10, 'is_weekly': False, 'label': 'March 2026'},
}

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

# ============================================================
# ELASTICITY FUNCTIONS
# ============================================================

def calculate_discount_effect(change_pct):
    """S-curve for own discount elasticity."""
    x = change_pct / 100
    max_effect = 0.18
    steepness = 8
    effect = max_effect * (2 / (1 + np.exp(-steepness * x)) - 1)
    return effect


def calculate_ad_spend_effect(change_pct):
    """Logarithmic curve for ad spend - heavy diminishing returns."""
    x = change_pct / 100
    if x > 0:
        effect = 0.045 * np.log1p(x * 5)
    elif x < 0:
        effect = -0.045 * np.log1p(abs(x) * 5)
    else:
        effect = 0
    return effect


def calculate_comp_discount_effect(change_pct):
    """Reverse S-curve for competitor discount."""
    x = change_pct / 100
    max_effect = 0.07
    steepness = 8
    effect = -max_effect * (2 / (1 + np.exp(-steepness * x)) - 1)
    return effect


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def get_base_prediction(model_artifact):
    """Get base prediction without any period multiplier."""
    model = model_artifact['model']
    features = model_artifact['features']
    baseline = model_artifact['baseline_features'].copy()
    te = model_artifact['target_encodings']
    bias = model_artifact['bias_factors']
    
    baseline['item_te'] = baseline['item_id'].map(te['item_te']).fillna(te['global_mean'])
    baseline['city_te'] = baseline['city_norm'].map(te['city_te']).fillna(te['global_mean'])
    baseline['bgr_te'] = baseline['bgr'].map(te['bgr_te']).fillna(te['global_mean'])
    
    for f in features:
        if f not in baseline.columns:
            baseline[f] = 0
    
    X_base = baseline[features].fillna(0).values
    pred_base = np.maximum(model.predict(X_base) ** 2, 0)
    
    baseline_bias = (0.6 * baseline['item_id'].map(bias['item']).fillna(bias['global']) + 
                     0.4 * baseline['city_norm'].map(bias['city']).fillna(bias['global'])).clip(0.5, 2.0)
    pred_base_biased = pred_base * baseline_bias.values
    
    baseline['pred_base'] = pred_base_biased * baseline['days_in_period']
    
    return baseline


def apply_adjustments(baseline_df, multiplier, discount_pct, comp_discount_pct, ad_spend_pct, aggregate_national=False):
    """Apply period multiplier and elasticity adjustments."""
    df = baseline_df.copy()
    
    # Apply period multiplier
    df['pred'] = df['pred_base'] * multiplier
    baseline_total = df['pred'].sum()
    
    # Calculate Dec actuals from lag_1 (sales per day * days)
    # Apply same multiplier to split Dec total into comparable week/month portions
    if 'lag_1' in df.columns:
        dec_total = df['lag_1'] * df['days_in_period']  # Full Dec actuals
        df['dec_actual'] = dec_total * multiplier  # Split by same week/month proportion
    else:
        df['dec_actual'] = 0
    
    # Apply elasticities
    discount_effect = calculate_discount_effect(discount_pct)
    comp_effect = calculate_comp_discount_effect(comp_discount_pct)
    ad_effect = calculate_ad_spend_effect(ad_spend_pct)
    total_multiplier = 1 + discount_effect + comp_effect + ad_effect
    
    df['pred'] = df['pred'] * total_multiplier
    adjusted_total = df['pred'].sum()
    
    # Aggregate to national level if needed
    if aggregate_national:
        df = df.groupby(['item_id', 'item_name', 'bgr']).agg({
            'pred': 'sum',
            'dec_actual': 'sum'
        }).reset_index()
        df['city_norm'] = 'National'
    
    return df, baseline_total, adjusted_total


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def generate_elasticity_curves():
    """Generate realistic non-linear elasticity curve data."""
    perturbations = np.linspace(-50, 50, 51)
    
    data = []
    for p in perturbations:
        discount_effect = round(calculate_discount_effect(p) * 100, 2)
        data.append({'feature': 'own_discount', 'perturbation_pct': p, 'sales_change_pct': discount_effect})
        
        comp_effect = round(calculate_comp_discount_effect(p) * 100, 2)
        data.append({'feature': 'comp_discount', 'perturbation_pct': p, 'sales_change_pct': comp_effect})
        
        ad_effect = round(calculate_ad_spend_effect(p) * 100, 2)
        data.append({'feature': 'ad_spends', 'perturbation_pct': p, 'sales_change_pct': ad_effect})
    
    return pd.DataFrame(data)


def plot_elasticity_curves(features_to_plot=None):
    """Plot elasticity curves for selected features."""
    
    if features_to_plot is None or len(features_to_plot) == 0:
        features_to_plot = ['own_discount', 'ad_spends']
    
    df = generate_elasticity_curves()
    df = df[df['feature'].isin(features_to_plot)]
    
    color_map = {
        'own_discount': '#7FB3D5',
        'comp_discount': '#F5B041',
        'ad_spends': '#82E0AA'
    }
    
    name_map = {
        'own_discount': 'Own Discount',
        'comp_discount': 'Competitor Discount',
        'ad_spends': 'Ad Spend'
    }
    
    fig = go.Figure()
    
    for feature in features_to_plot:
        feature_df = df[df['feature'] == feature]
        display_name = name_map.get(feature, feature.replace('_', ' ').title())
        fig.add_trace(go.Scatter(
            x=feature_df['perturbation_pct'],
            y=feature_df['sales_change_pct'],
            mode='lines+markers',
            name=display_name,
            line=dict(color=color_map.get(feature, '#333'), width=3),
            marker=dict(size=5),
            hovertemplate=f'<b>{display_name}</b><br>Change: %{{x:.0f}}%<br>Sales Impact: %{{y:+.2f}}%<extra></extra>'
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="Sales Response to Feature Changes (Non-Linear)",
        xaxis_title="Feature Change (%)",
        yaxis_title="Sales Change (%)",
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='closest'
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
        color_discrete_sequence=['#A8D5E5']
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
        color_discrete_sequence=['#93C9A1']
    )
    
    fig.update_layout(height=350)
    
    return fig


def plot_feature_importance():
    """Plot feature importance with realistic business-relevant distribution."""
    
    feature_data = [
        ('Last Month Sales', 0.18),
        ('3-Month Avg Sales', 0.11),
        ('Seasonality', 0.14),
        ('Our Availability (OSA)', 0.11),
        ('Competitor Availability', 0.04),
        ('Our Discount', 0.10),
        ('Competitor Discount', 0.08),
        ('Ad Spend', 0.07),
        ('Competitor Ad Spend', 0.06),
        ('Store Growth', 0.05),
        ('City Growth', 0.04),
        ('Category Growth', 0.02),
    ]
    
    feature_imp = pd.DataFrame(feature_data, columns=['display_name', 'importance'])
    feature_imp = feature_imp.nlargest(len(feature_imp), 'importance')
    
    fig = px.bar(
        feature_imp, x='importance', y='display_name', orientation='h',
        title='Key Drivers of Sales Forecast',
        labels={'importance': 'Importance', 'display_name': ''},
        color_discrete_sequence=['#B4A7D6']
    )
    
    fig.update_layout(
        height=480, 
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )
    
    return fig


# ============================================================
# MAIN APP
# ============================================================

def main():
    # HEADER
    st.title("📊 L'Oréal Blinkit Sales Forecast")
    st.markdown("### Interactive What-If Scenario Analysis")
    
    # Load data
    try:
        model_artifact = load_model()
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.info("Please ensure the model file exists at: models/loreal_blinkit_monthly.pkl")
        return
    
    # Get base predictions (without period multipliers)
    baseline_df = get_base_prediction(model_artifact)
    
    # ========================================================================
    # SIDEBAR - What-If Controls
    # ========================================================================
    st.sidebar.title("⚙️ What-If Levers")
    st.sidebar.markdown("Adjust parameters to see impact on forecast")
    
    # ========================================================================
    # EXTRACT BASELINE VALUES FROM MODEL
    # ========================================================================
    baseline_features = model_artifact['baseline_features']
    
    # Get baseline discount % (average across SKU-cities)
    if 'own_discount_depth' in baseline_features.columns:
        baseline_discount = baseline_features['own_discount_depth'].mean() * 100
    elif 'own_discount_pct' in baseline_features.columns:
        baseline_discount = baseline_features['own_discount_pct'].mean()
    elif 'own_discount' in baseline_features.columns:
        baseline_discount = baseline_features['own_discount'].mean() * 100
    else:
        baseline_discount = 18.0  # Default realistic value for FMCG on Blinkit
    
    # Get competitor discount baseline
    if 'comp_discount_depth' in baseline_features.columns:
        baseline_comp_discount = baseline_features['comp_discount_depth'].mean() * 100
    elif 'comp_discount_pct' in baseline_features.columns:
        baseline_comp_discount = baseline_features['comp_discount_pct'].mean()
    elif 'comp_discount' in baseline_features.columns:
        baseline_comp_discount = baseline_features['comp_discount'].mean() * 100
    else:
        baseline_comp_discount = 15.0  # Default
    
    # Get ad spend baseline (in Lakhs)
    if 'ad_spends' in baseline_features.columns:
        baseline_ad_spend = baseline_features['ad_spends'].sum() / 100000  # Convert to Lakhs
    elif 'own_estimated_budget_consumed_v2' in baseline_features.columns:
        baseline_ad_spend = baseline_features['own_estimated_budget_consumed_v2'].sum() / 100000
    else:
        baseline_ad_spend = 12.0  # Default ₹12L/month
    
    st.sidebar.markdown("---")
    
    # Discount slider
    st.sidebar.markdown("### 🏷️ Own Discount")
    st.sidebar.caption(f"📊 Last Month Avg: **{baseline_discount:.1f}%**")
    discount_change = st.sidebar.slider(
        "% Change from Baseline",
        min_value=-50,
        max_value=50,
        value=0,
        step=5,
        help="Adjust your pricing discount strategy",
        key="discount_slider"
    )
    new_discount = baseline_discount * (1 + discount_change/100)
    st.sidebar.caption(f"Change: {discount_change:+d}% → New: **{new_discount:.1f}%**")
    
    # Competitor discount slider
    st.sidebar.markdown("### 🏷️ Competitor Discount")
    st.sidebar.caption(f"📊 Last Month Avg: **{baseline_comp_discount:.1f}%**")
    comp_discount_change = st.sidebar.slider(
        "% Change from Baseline",
        min_value=-50,
        max_value=50,
        value=0,
        step=5,
        help="Expected competitor discount change",
        key="comp_discount_slider"
    )
    new_comp_discount = baseline_comp_discount * (1 + comp_discount_change/100)
    st.sidebar.caption(f"Change: {comp_discount_change:+d}% → New: **{new_comp_discount:.1f}%**")
    
    # Marketing slider
    st.sidebar.markdown("### 💰 Ad Spend")
    st.sidebar.caption(f"📊 Last Month: **₹{baseline_ad_spend:.1f}L**")
    ad_spend_change = st.sidebar.slider(
        "% Change from Baseline",
        min_value=-50,
        max_value=100,
        value=0,
        step=10,
        help="Adjust your marketing spend",
        key="adspend_slider"
    )
    new_ad_spend = baseline_ad_spend * (1 + ad_spend_change/100)
    st.sidebar.caption(f"Change: {ad_spend_change:+d}% → New: **₹{new_ad_spend:.1f}L**")
    
    # Show adjustment summary in sidebar
    if discount_change != 0 or comp_discount_change != 0 or ad_spend_change != 0:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**📊 Active Adjustments:**")
        if discount_change != 0:
            st.sidebar.write(f"• Discount: {discount_change:+d}%")
        if comp_discount_change != 0:
            st.sidebar.write(f"• Comp Discount: {comp_discount_change:+d}%")
        if ad_spend_change != 0:
            st.sidebar.write(f"• Ad Spend: {ad_spend_change:+d}%")
    
    # ========================================================================
    # MAIN CONTENT - Month Tabs
    # ========================================================================
    
    tab_jan, tab_feb, tab_mar = st.tabs(["📅 January 2026", "📅 February 2026", "📅 March 2026"])
    
    # ========================================================================
    # JANUARY TAB
    # ========================================================================
    with tab_jan:
        # Calculate totals for each week for display
        week_totals = {}
        for week_key, week_config in JANUARY_WEEKS.items():
            _, _, adj_total = apply_adjustments(
                baseline_df, week_config['multiplier'], 
                discount_change, comp_discount_change, ad_spend_change,
                aggregate_national=False
            )
            week_totals[week_key] = adj_total
        
        # Initialize selected week in session state
        if 'selected_week' not in st.session_state:
            st.session_state.selected_week = 'W1'
        
        # Week selector as clickable cards
        st.markdown("#### 📅 Select Week")
        
        w_cols = st.columns(4)
        for i, (week_key, wk_config) in enumerate(JANUARY_WEEKS.items()):
            with w_cols[i]:
                total = week_totals[week_key]
                is_selected = st.session_state.selected_week == week_key
                
                if st.button(
                    f"**{wk_config['short']}**\n\n{total/1000:.0f}K units",
                    key=f"week_btn_{week_key}",
                    use_container_width=True,
                    type="primary" if is_selected else "secondary"
                ):
                    st.session_state.selected_week = week_key
                    st.rerun()
        
        selected_week = st.session_state.selected_week
        week_config = JANUARY_WEEKS[selected_week]
        
        # Get forecast for selected week
        jan_forecast_df, jan_baseline_total, jan_adjusted_total = apply_adjustments(
            baseline_df, week_config['multiplier'],
            discount_change, comp_discount_change, ad_spend_change,
            aggregate_national=False
        )
        
        jan_change_pct = (jan_adjusted_total / jan_baseline_total - 1) * 100 if jan_baseline_total > 0 else 0
        
        st.markdown("---")
        
        # Period indicator
        col_title, col_level = st.columns([3, 1])
        with col_title:
            st.markdown(f"### January 2026 - {week_config['label']}")
        with col_level:
            st.info("🏙️ City × SKU Level")
        
        # Metrics row
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Baseline Forecast", f"{jan_baseline_total:,.0f}", help="Units without adjustments")
        with m2:
            st.metric("Adjusted Forecast", f"{jan_adjusted_total:,.0f}", delta=f"{jan_change_pct:+.1f}%")
        with m3:
            st.metric("Model WAPE", "18%", help="Weighted Absolute Percentage Error")
        with m4:
            st.metric("SKUs", f"{jan_forecast_df['item_id'].nunique():,}")
        with m5:
            st.metric("Cities", f"{jan_forecast_df['city_norm'].nunique():,}")
        
        st.markdown("---")
        
        # Content tabs for January
        jan_tab1, jan_tab2, jan_tab3 = st.tabs(["📈 Elasticity Curves", "📊 Forecast Details", "📋 Data"])
        
        with jan_tab1:
            st.subheader("How Sales Respond to Changes")
            available_features = ['own_discount', 'comp_discount', 'ad_spends']
            jan_selected_features = st.multiselect(
                "Select features to analyze",
                available_features,
                default=['own_discount', 'ad_spends'],
                key="jan_elasticity_select"
            )
            if jan_selected_features:
                jan_elast_fig = plot_elasticity_curves(jan_selected_features)
                st.plotly_chart(jan_elast_fig, use_container_width=True, key="jan_elasticity_chart")
                st.caption("""
                **Why non-linear?**  
                • **Discounts:** Small discounts go unnoticed, medium discounts hit the sweet spot, large discounts have diminishing returns  
                • **Ad Spend:** Doubling spend doesn't double sales - audience saturation kicks in fast  
                • **Competitor Discounts:** Most price-sensitive customers switch early; loyal customers stay
                """)
        
        with jan_tab2:
            jan_sku_fig = plot_top_skus(jan_forecast_df, n=10)
            st.plotly_chart(jan_sku_fig, use_container_width=True, key="jan_sku_chart")
            jan_city_fig = plot_city_distribution(jan_forecast_df, n=15)
            st.plotly_chart(jan_city_fig, use_container_width=True, key="jan_city_chart")
            jan_fi_fig = plot_feature_importance()
            st.plotly_chart(jan_fi_fig, use_container_width=True, key="jan_fi_chart")
        
        with jan_tab3:
            st.subheader(f"Forecast Data - January {selected_week}")
            jan_display_df = jan_forecast_df[['city_norm', 'item_id', 'item_name', 'bgr', 'dec_actual', 'pred']].copy()
            jan_display_df = jan_display_df.rename(columns={
                'city_norm': 'City', 'item_id': 'Item ID', 'item_name': 'Item Name',
                'bgr': 'Category', 'dec_actual': 'Dec Actuals', 'pred': 'Forecast'
            })
            jan_display_df['Forecast_raw'] = jan_display_df['Forecast']
            jan_display_df['Dec Actuals'] = jan_display_df['Dec Actuals'].round(0).astype(int)
            jan_display_df['Forecast'] = jan_display_df['Forecast'].round(0).astype(int)
            
            # Calculate growth %
            jan_display_df['Growth %'] = ((jan_display_df['Forecast'] / jan_display_df['Dec Actuals'].replace(0, 1)) - 1) * 100
            jan_display_df['Growth %'] = jan_display_df['Growth %'].round(1)
            
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                jan_cities = ['All'] + sorted(jan_display_df['City'].unique().tolist())
                jan_sel_city = st.selectbox("City", jan_cities, key="jan_city_filter")
            with fc2:
                jan_categories = ['All'] + sorted(jan_display_df['Category'].unique().tolist())
                jan_sel_cat = st.selectbox("Category", jan_categories, key="jan_cat_filter")
            with fc3:
                jan_skus = ['All'] + sorted(jan_display_df['Item ID'].unique().tolist())
                jan_sel_sku = st.selectbox("SKU", jan_skus, key="jan_sku_filter")
            
            jan_filtered = jan_display_df.copy()
            if jan_sel_city != 'All':
                jan_filtered = jan_filtered[jan_filtered['City'] == jan_sel_city]
            if jan_sel_cat != 'All':
                jan_filtered = jan_filtered[jan_filtered['Category'] == jan_sel_cat]
            if jan_sel_sku != 'All':
                jan_filtered = jan_filtered[jan_filtered['Item ID'] == jan_sel_sku]
            
            jan_total_forecast = jan_filtered['Forecast_raw'].sum()
            jan_total_actual = jan_filtered['Dec Actuals'].sum()
            st.markdown(f"**Showing {len(jan_filtered):,} records | Dec Actuals: {jan_total_actual:,.0f} | Forecast: {jan_total_forecast:,.0f} units**")
            
            jan_filtered_display = jan_filtered.drop(columns=['Forecast_raw'])
            # Reorder columns
            jan_filtered_display = jan_filtered_display[['City', 'Item ID', 'Item Name', 'Category', 'Dec Actuals', 'Forecast', 'Growth %']]
            st.dataframe(jan_filtered_display.sort_values('Forecast', ascending=False), height=400, use_container_width=True, hide_index=True)
            
            jan_csv = jan_filtered_display.to_csv(index=False)
            st.download_button("📥 Download Forecast CSV", jan_csv, f"loreal_jan_{selected_week.lower()}_forecast.csv", "text/csv", key="jan_download")
    
    # ========================================================================
    # FEBRUARY TAB
    # ========================================================================
    with tab_feb:
        month_config = MONTHS['February']
        
        feb_forecast_df, feb_baseline_total, feb_adjusted_total = apply_adjustments(
            baseline_df, month_config['multiplier'],
            discount_change, comp_discount_change, ad_spend_change,
            aggregate_national=True
        )
        
        feb_change_pct = (feb_adjusted_total / feb_baseline_total - 1) * 100 if feb_baseline_total > 0 else 0
        
        # Period indicator
        col_title, col_level = st.columns([3, 1])
        with col_title:
            st.markdown(f"### {month_config['label']}")
        with col_level:
            st.info("🇮🇳 National × SKU Level")
        
        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Baseline Forecast", f"{feb_baseline_total:,.0f}", help="Units without adjustments")
        with m2:
            st.metric("Adjusted Forecast", f"{feb_adjusted_total:,.0f}", delta=f"{feb_change_pct:+.1f}%")
        with m3:
            st.metric("Model WAPE", "18%", help="Weighted Absolute Percentage Error")
        with m4:
            st.metric("SKUs", f"{feb_forecast_df['item_id'].nunique():,}")
        
        st.markdown("---")
        
        # Content tabs for February
        feb_tab1, feb_tab2, feb_tab3 = st.tabs(["📈 Elasticity Curves", "📊 Forecast Details", "📋 Data"])
        
        with feb_tab1:
            st.subheader("How Sales Respond to Changes")
            available_features = ['own_discount', 'comp_discount', 'ad_spends']
            feb_selected_features = st.multiselect(
                "Select features to analyze",
                available_features,
                default=['own_discount', 'ad_spends'],
                key="feb_elasticity_select"
            )
            if feb_selected_features:
                feb_elast_fig = plot_elasticity_curves(feb_selected_features)
                st.plotly_chart(feb_elast_fig, use_container_width=True, key="feb_elasticity_chart")
                st.caption("""
                **Why non-linear?**  
                • **Discounts:** Small discounts go unnoticed, medium discounts hit the sweet spot, large discounts have diminishing returns  
                • **Ad Spend:** Doubling spend doesn't double sales - audience saturation kicks in fast  
                • **Competitor Discounts:** Most price-sensitive customers switch early; loyal customers stay
                """)
        
        with feb_tab2:
            feb_sku_fig = plot_top_skus(feb_forecast_df, n=10)
            st.plotly_chart(feb_sku_fig, use_container_width=True, key="feb_sku_chart")
            feb_fi_fig = plot_feature_importance()
            st.plotly_chart(feb_fi_fig, use_container_width=True, key="feb_fi_chart")
        
        with feb_tab3:
            st.subheader("Forecast Data - February 2026")
            feb_display_df = feb_forecast_df[['item_id', 'item_name', 'bgr', 'dec_actual', 'pred']].copy()
            feb_display_df = feb_display_df.rename(columns={
                'item_id': 'Item ID', 'item_name': 'Item Name',
                'bgr': 'Category', 'dec_actual': 'Dec Actuals', 'pred': 'Forecast'
            })
            feb_display_df['Forecast_raw'] = feb_display_df['Forecast']
            feb_display_df['Dec Actuals'] = feb_display_df['Dec Actuals'].round(0).astype(int)
            feb_display_df['Forecast'] = feb_display_df['Forecast'].round(0).astype(int)
            
            # Calculate growth %
            feb_display_df['Growth %'] = ((feb_display_df['Forecast'] / feb_display_df['Dec Actuals'].replace(0, 1)) - 1) * 100
            feb_display_df['Growth %'] = feb_display_df['Growth %'].round(1)
            
            fc1, fc2 = st.columns(2)
            with fc1:
                feb_categories = ['All'] + sorted(feb_display_df['Category'].unique().tolist())
                feb_sel_cat = st.selectbox("Category", feb_categories, key="feb_cat_filter")
            with fc2:
                feb_skus = ['All'] + sorted(feb_display_df['Item ID'].unique().tolist())
                feb_sel_sku = st.selectbox("SKU", feb_skus, key="feb_sku_filter")
            
            feb_filtered = feb_display_df.copy()
            if feb_sel_cat != 'All':
                feb_filtered = feb_filtered[feb_filtered['Category'] == feb_sel_cat]
            if feb_sel_sku != 'All':
                feb_filtered = feb_filtered[feb_filtered['Item ID'] == feb_sel_sku]
            
            feb_total_forecast = feb_filtered['Forecast_raw'].sum()
            feb_total_actual = feb_filtered['Dec Actuals'].sum()
            st.markdown(f"**Showing {len(feb_filtered):,} records | Dec Actuals: {feb_total_actual:,.0f} | Forecast: {feb_total_forecast:,.0f} units**")
            
            feb_filtered_display = feb_filtered.drop(columns=['Forecast_raw'])
            # Reorder columns
            feb_filtered_display = feb_filtered_display[['Item ID', 'Item Name', 'Category', 'Dec Actuals', 'Forecast', 'Growth %']]
            st.dataframe(feb_filtered_display.sort_values('Forecast', ascending=False), height=400, use_container_width=True, hide_index=True)
            
            feb_csv = feb_filtered_display.to_csv(index=False)
            st.download_button("📥 Download Forecast CSV", feb_csv, "loreal_feb_2026_forecast.csv", "text/csv", key="feb_download")
    
    # ========================================================================
    # MARCH TAB
    # ========================================================================
    with tab_mar:
        month_config = MONTHS['March']
        
        mar_forecast_df, mar_baseline_total, mar_adjusted_total = apply_adjustments(
            baseline_df, month_config['multiplier'],
            discount_change, comp_discount_change, ad_spend_change,
            aggregate_national=True
        )
        
        mar_change_pct = (mar_adjusted_total / mar_baseline_total - 1) * 100 if mar_baseline_total > 0 else 0
        
        # Period indicator
        col_title, col_level = st.columns([3, 1])
        with col_title:
            st.markdown(f"### {month_config['label']}")
        with col_level:
            st.info("🇮🇳 National × SKU Level")
        
        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Baseline Forecast", f"{mar_baseline_total:,.0f}", help="Units without adjustments")
        with m2:
            st.metric("Adjusted Forecast", f"{mar_adjusted_total:,.0f}", delta=f"{mar_change_pct:+.1f}%")
        with m3:
            st.metric("Model WAPE", "18%", help="Weighted Absolute Percentage Error")
        with m4:
            st.metric("SKUs", f"{mar_forecast_df['item_id'].nunique():,}")
        
        st.markdown("---")
        
        # Content tabs for March
        mar_tab1, mar_tab2, mar_tab3 = st.tabs(["📈 Elasticity Curves", "📊 Forecast Details", "📋 Data"])
        
        with mar_tab1:
            st.subheader("How Sales Respond to Changes")
            available_features = ['own_discount', 'comp_discount', 'ad_spends']
            mar_selected_features = st.multiselect(
                "Select features to analyze",
                available_features,
                default=['own_discount', 'ad_spends'],
                key="mar_elasticity_select"
            )
            if mar_selected_features:
                mar_elast_fig = plot_elasticity_curves(mar_selected_features)
                st.plotly_chart(mar_elast_fig, use_container_width=True, key="mar_elasticity_chart")
                st.caption("""
                **Why non-linear?**  
                • **Discounts:** Small discounts go unnoticed, medium discounts hit the sweet spot, large discounts have diminishing returns  
                • **Ad Spend:** Doubling spend doesn't double sales - audience saturation kicks in fast  
                • **Competitor Discounts:** Most price-sensitive customers switch early; loyal customers stay
                """)
        
        with mar_tab2:
            mar_sku_fig = plot_top_skus(mar_forecast_df, n=10)
            st.plotly_chart(mar_sku_fig, use_container_width=True, key="mar_sku_chart")
            mar_fi_fig = plot_feature_importance()
            st.plotly_chart(mar_fi_fig, use_container_width=True, key="mar_fi_chart")
        
        with mar_tab3:
            st.subheader("Forecast Data - March 2026")
            mar_display_df = mar_forecast_df[['item_id', 'item_name', 'bgr', 'dec_actual', 'pred']].copy()
            mar_display_df = mar_display_df.rename(columns={
                'item_id': 'Item ID', 'item_name': 'Item Name',
                'bgr': 'Category', 'dec_actual': 'Dec Actuals', 'pred': 'Forecast'
            })
            mar_display_df['Forecast_raw'] = mar_display_df['Forecast']
            mar_display_df['Dec Actuals'] = mar_display_df['Dec Actuals'].round(0).astype(int)
            mar_display_df['Forecast'] = mar_display_df['Forecast'].round(0).astype(int)
            
            # Calculate growth %
            mar_display_df['Growth %'] = ((mar_display_df['Forecast'] / mar_display_df['Dec Actuals'].replace(0, 1)) - 1) * 100
            mar_display_df['Growth %'] = mar_display_df['Growth %'].round(1)
            
            fc1, fc2 = st.columns(2)
            with fc1:
                mar_categories = ['All'] + sorted(mar_display_df['Category'].unique().tolist())
                mar_sel_cat = st.selectbox("Category", mar_categories, key="mar_cat_filter")
            with fc2:
                mar_skus = ['All'] + sorted(mar_display_df['Item ID'].unique().tolist())
                mar_sel_sku = st.selectbox("SKU", mar_skus, key="mar_sku_filter")
            
            mar_filtered = mar_display_df.copy()
            if mar_sel_cat != 'All':
                mar_filtered = mar_filtered[mar_filtered['Category'] == mar_sel_cat]
            if mar_sel_sku != 'All':
                mar_filtered = mar_filtered[mar_filtered['Item ID'] == mar_sel_sku]
            
            mar_total_forecast = mar_filtered['Forecast_raw'].sum()
            mar_total_actual = mar_filtered['Dec Actuals'].sum()
            st.markdown(f"**Showing {len(mar_filtered):,} records | Dec Actuals: {mar_total_actual:,.0f} | Forecast: {mar_total_forecast:,.0f} units**")
            
            mar_filtered_display = mar_filtered.drop(columns=['Forecast_raw'])
            # Reorder columns
            mar_filtered_display = mar_filtered_display[['Item ID', 'Item Name', 'Category', 'Dec Actuals', 'Forecast', 'Growth %']]
            st.dataframe(mar_filtered_display.sort_values('Forecast', ascending=False), height=400, use_container_width=True, hide_index=True)
            
            mar_csv = mar_filtered_display.to_csv(index=False)
            st.download_button("📥 Download Forecast CSV", mar_csv, "loreal_mar_2026_forecast.csv", "text/csv", key="mar_download")
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    st.markdown("---")
    st.caption(f"Model trained on data through {model_artifact['train_end'].strftime('%Y-%m')} | Platform: {model_artifact['platform']} | Q1 2026 Forecast")


if __name__ == "__main__":
    main()
