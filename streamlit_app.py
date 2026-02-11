"""
Demand Forecasting Dashboard v4
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(page_title="Demand Forecasting", page_icon="📊", layout="wide")
st.markdown("""<style>
.stButton>button[kind="primary"]{background:#3B82F6;border-color:#3B82F6}
</style>""", unsafe_allow_html=True)

DATA_DIR = 'data'

@st.cache_data
def load_data():
    d = {}
    with open(f'{DATA_DIR}/config.pkl','rb') as f: d['config'] = pickle.load(f)
    try:
        d['forecasts_df'] = pd.read_parquet(f'{DATA_DIR}/forecasts_df.parquet')
    except FileNotFoundError:
        d['forecasts_df'] = pd.DataFrame()
    try:
        d['city_forecast_df'] = pd.read_parquet(f'{DATA_DIR}/city_forecast_df.parquet')
    except FileNotFoundError:
        d['city_forecast_df'] = pd.DataFrame()
    try:
        d['delta_table'] = pd.read_parquet(f'{DATA_DIR}/delta_table.parquet')
    except FileNotFoundError:
        d['delta_table'] = pd.DataFrame()
    for n in ['curves','sales_validation','live_validation','po_validation','po_base',
              'po_splits','po_features','historical_sales','forecast_weeks',
              'item_names','historical_po','rca_data','po_rca_data','baseline_levers','feature_importances']:
        try:
            with open(f'{DATA_DIR}/{n}.pkl','rb') as f: d[n] = pickle.load(f)
        except FileNotFoundError:
            d[n] = [] if n == 'live_validation' else {}
    d['po_forecasts'], d['po_forecasts_adj'] = {}, {}
    for p in d['config']['platforms']:
        try:
            d['po_forecasts'][p] = pd.read_parquet(f'{DATA_DIR}/po_forecasts_{p}.parquet')
        except FileNotFoundError:
            d['po_forecasts'][p] = pd.DataFrame()
        try:
            d['po_forecasts_adj'][p] = pd.read_parquet(f'{DATA_DIR}/po_forecasts_adj_{p}.parquet')
        except FileNotFoundError:
            d['po_forecasts_adj'][p] = pd.DataFrame()
    return d

D = load_data(); C = D['config']
FM = sorted(set(fm['month'] for fm in C['forecast_months']))

def iname(iid, plat): return D.get('item_names',{}).get(plat,{}).get(iid, str(iid))

def wklabels(mo):
    ms=pd.Timestamp(mo+'-01'); me=ms+pd.offsets.MonthEnd(0); L={}; cs=ms; w=1
    while cs<=me:
        ce=min(cs+pd.Timedelta(days=6-cs.weekday()),me)
        L[w]=f"W{w} ({cs.strftime('%d %b')}-{ce.strftime('%d %b')})"; cs=ce+pd.Timedelta(days=1); w+=1
    return L

def sfc(plat, mo, gk='wh_sku_weekly'):
    fc=D['forecasts_df']; fc=fc[(fc['platform']==plat)&(fc['forecast_month']==mo)&(fc['gran_key']==gk)]
    if 'horizon' in fc.columns and len(fc)>0: fc=fc[fc['horizon']==sorted(fc['horizon'].unique())[0]]
    return fc

def pfc(plat, mo, moq=True, gk=None):
    s=D['po_forecasts_adj'] if moq else D['po_forecasts']; po=s[plat]; po=po[po['forecast_month']==mo]
    if 'horizon' in po.columns and len(po)>0: po=po[po['horizon']==sorted(po['horizon'].unique())[0]]
    if 'gran_key' in po.columns and len(po)>0:
        if gk and gk in po['gran_key'].values:
            po=po[po['gran_key']==gk]
        elif 'wh_sku_weekly' in po['gran_key'].values:
            po=po[po['gran_key']=='wh_sku_weekly']
        else:
            po=po[po['gran_key']==po['gran_key'].iloc[0]]
    return po

def hscaled(plat):
    h=D['historical_sales'].get(plat,pd.DataFrame()).copy(); p=C.get('partial_month',{})
    if p and len(h)>0:
        pm=pd.Timestamp(p['month']+'-01'); mask=h['month']==pm
        if mask.any() and p['actual_days']<p['expected_days']:
            h.loc[mask,'qty_sold']*=p['expected_days']/p['actual_days']
    return h

def compute_wi(fc, disc, ad, comp, osa_pct=0):
    curves=D.get('curves',{})
    if len(fc)==0 or not curves: return None
    base=fc['pred'].sum()
    results={'base':base}
    for cp in [0,50,100]:
        effective={'rpi':disc*(1-cp/100),'sov':ad*(1-cp/100),'osa':osa_pct*(1-cp/100)}
        total_effect=sum(
            curves[l]['fitted']['params']['L']*np.tanh(curves[l]['fitted']['params']['k']*eff)
            for l,eff in effective.items() if l in curves and eff!=0
        )
        results[cp]=base*(1+total_effect/100)
    return results

def get_gran_keys_for_month(mo):
    """Get all gran_keys configured for a forecast month."""
    keys = set()
    for fm in C['forecast_months']:
        if fm['month'] == mo:
            gmap = {'WH x SKU':'wh_sku','National x SKU':'national_sku','City x SKU':'city_sku'}
            gran = gmap.get(fm['granularity'], 'wh_sku')
            tg = fm.get('time_gran', 'monthly')
            keys.add(f"{gran}_{tg}")
    return sorted(keys)

# ── SIDEBAR ───────────────────────────────────────────────
st.sidebar.title("📊 Demand Forecasting")
st.sidebar.markdown(f"**Brand:** {C['brand_db']}")
st.sidebar.markdown(f"**Data observed till:** {C['data_end_date']}")
platform = st.sidebar.selectbox("Platform", C['platforms'])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Validation Accuracy")
sv=D.get('sales_validation',{})
if platform in sv and len(sv[platform])>0:
    wapes=[r['wape']['monthly_indirect'] for r in sv[platform]]
    st.sidebar.metric("Sales WAPE",f"{np.mean(wapes)*100:.1f}%",help=f"City×SKU, avg over {len(wapes)} validation months")
pv=D.get('po_validation',{})
if platform in pv and len(pv[platform])>0:
    po_wh_rows = [v for v in pv[platform] if v.get('granularity','') != 'National × SKU']
    if po_wh_rows:
        po_wapes=[v['wape_sku'] for v in po_wh_rows]
        st.sidebar.metric("PO WAPE",f"{np.mean(po_wapes)*100:.1f}%",help=f"WH×SKU level, avg over {len(po_wapes)} validation months")

st.sidebar.markdown("---")
st.sidebar.caption("Built by Gobble Cube")

# ── TITLE ─────────────────────────────────────────────────
st.title(f"📊 {C['brand_db']} — {platform} Demand Forecast")

# Build tab list: one tab per (month, granularity) combination
tab_specs = []
for fm in C['forecast_months']:
    mo = fm['month']
    gran = fm['granularity']
    hz = fm['horizon']
    tg = fm.get('time_gran', 'monthly')
    gmap = {'WH x SKU':'wh_sku','National x SKU':'national_sku','City x SKU':'city_sku'}
    gran_key = f"{gmap.get(gran,'wh_sku')}_{tg}"
    label = f"📅 {pd.Timestamp(mo+'-01').strftime('%b %Y')} ({gran})"
    tab_specs.append({'month': mo, 'gran': gran, 'horizon': hz, 'time_gran': tg, 'gran_key': gran_key, 'label': label})

# Deduplicate (same month+gran)
seen = set()
unique_tabs = []
for ts in tab_specs:
    key = (ts['month'], ts['gran'])
    if key not in seen:
        seen.add(key)
        unique_tabs.append(ts)

mtabs = st.tabs([ts['label'] for ts in unique_tabs])

for ti, (mtab, tspec) in enumerate(zip(mtabs, unique_tabs)):
    mo = tspec['month']
    gran = tspec['gran']
    hz = tspec['horizon']
    gran_key = tspec['gran_key']
    is_national = 'national' in gran.lower()
    
    with mtab:
        if is_national:
            fc_wh = sfc(platform, mo, gran_key)
        else:
            fc_wh = sfc(platform, mo)
        pr = pfc(platform, mo, False, gran_key); pa = pfc(platform, mo, True, gran_key)
        h = hscaled(platform); wl = wklabels(mo)

        st_tot=round(fc_wh['pred'].sum()) if len(fc_wh)>0 else 0
        pt_tot=round(pr['pred_po'].sum()) if len(pr)>0 else 0
        pa_tot=round(pa['pred_po'].sum()) if len(pa)>0 else 0
        last_a=round(h.iloc[-1]['qty_sold']) if len(h)>0 else 0
        last_mn=h.iloc[-1]['month'].strftime('%b %Y') if len(h)>0 else ''

        st.markdown(f"### {pd.Timestamp(mo+'-01').strftime('%B %Y')} — {gran} | {hz}")

        m1,m2,m3,m4,m5=st.columns(5)
        m1.metric("Sales Forecast",f"{st_tot:,}",f"{(st_tot/last_a-1)*100:+.1f}% vs {last_mn}" if last_a>0 else "")
        m2.metric("PO Forecast",f"{pt_tot:,}")
        m3.metric("PO + MOQ",f"{pa_tot:,}",f"{(pa_tot/pt_tot-1)*100:+.1f}% uplift" if pt_tot>0 else "")
        m4.metric("SKUs",f"{fc_wh['item_id'].nunique() if len(fc_wh)>0 else 0}")
        if is_national:
            m5.metric("Level","National")
        else:
            m5.metric("Warehouses",f"{fc_wh['brand_wh'].nunique() if 'brand_wh' in fc_wh.columns and len(fc_wh)>0 else 0}")
        
        st.markdown("---")
        tab_options = ["📈 Overview","🛒 Sales Detail","📦 PO Detail","🎛️ What-If Scenarios","🔍 Validation & RCA"]
        active_tab = st.radio("", tab_options, horizontal=True, key=f"inner_tab_{ti}", label_visibility="collapsed")

        # ═══════════════════════════════════════════════════
        # OVERVIEW
        # ═══════════════════════════════════════════════════
        if active_tab == tab_options[0]:
            st.subheader("Sales Trend + Forecast")
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=h['month'],y=h['qty_sold'],mode='lines+markers',name='Actual',
                line=dict(color='#5DADE2',width=2.5),marker=dict(size=6),
                hovertemplate='<b>%{x|%b %Y}</b><br>%{y:,.0f} units<extra></extra>'))
            fpts=[]
            for fm in FM:
                f=sfc(platform,fm)
                if len(f)>0: fpts.append({'month':pd.Timestamp(fm+'-01'),'v':f['pred'].sum()})
            if fpts:
                fd=pd.DataFrame(fpts)
                fig.add_trace(go.Scatter(x=[h.iloc[-1]['month'],fd.iloc[0]['month']],y=[h.iloc[-1]['qty_sold'],fd.iloc[0]['v']],
                    mode='lines',showlegend=False,hoverinfo='skip',line=dict(color='#E74C3C',width=2.5,dash='dash')))
                fig.add_trace(go.Scatter(x=fd['month'],y=fd['v'],mode='lines+markers',name='Forecast',
                    line=dict(color='#E74C3C',width=2.5,dash='dash'),marker=dict(size=8,symbol='diamond'),
                    hovertemplate='<b>%{x|%b %Y}</b><br>%{y:,.0f} units<extra></extra>'))
            fig.update_layout(height=370,template='plotly_white',margin=dict(b=5,t=30),
                xaxis_title='',yaxis_title='Units',hovermode='closest',
                legend=dict(orientation='h',yanchor='bottom',y=1.02,x=0.5,xanchor='center'))
            st.plotly_chart(fig,use_container_width=True,key=f"st_{ti}")
            st.caption(f"*Historical data observed till {C['data_end_date']}.")
            p=C.get('partial_month',{})
            if p.get('actual_days',31)<p.get('expected_days',31):
                st.caption(f"*{p['month']} sales scaled from {p['actual_days']} to {p['expected_days']} days")

            # PO trend
            ph=D.get('historical_po',{}).get(platform)
            if ph is not None and len(ph)>0:
                st.subheader("PO Trend + Forecast")
                phv=ph[ph['po_qty']>0]
                fp=go.Figure()
                fp.add_trace(go.Scatter(x=phv['month'],y=phv['po_qty'],mode='lines+markers',name='Actual PO',
                    line=dict(color='#5DADE2',width=2.5),marker=dict(size=6),
                    hovertemplate='<b>%{x|%b %Y}</b><br>%{y:,.0f} units<extra></extra>'))
                ppf=[]
                for fm in FM:
                    pf=pfc(platform,fm,True)
                    if len(pf)>0: ppf.append({'month':pd.Timestamp(fm+'-01'),'v':pf['pred_po'].sum()})
                if ppf:
                    pdf=pd.DataFrame(ppf)
                    fp.add_trace(go.Scatter(x=[phv.iloc[-1]['month'],pdf.iloc[0]['month']],y=[phv.iloc[-1]['po_qty'],pdf.iloc[0]['v']],
                        mode='lines',showlegend=False,hoverinfo='skip',line=dict(color='#E74C3C',width=2.5,dash='dash')))
                    fp.add_trace(go.Scatter(x=pdf['month'],y=pdf['v'],mode='lines+markers',name='PO Forecast (MOQ adj)',
                        line=dict(color='#E74C3C',width=2.5,dash='dash'),marker=dict(size=8,symbol='diamond'),
                        hovertemplate='<b>%{x|%b %Y}</b><br>%{y:,.0f} units<extra></extra>'))
                fp.update_layout(height=340,template='plotly_white',margin=dict(b=5,t=30),
                    xaxis_title='',yaxis_title='Units',hovermode='closest',
                    legend=dict(orientation='h',yanchor='bottom',y=1.02,x=0.5,xanchor='center'))
                st.plotly_chart(fp,use_container_width=True,key=f"pt_{ti}")
                st.caption(f"*Historical data observed till {C['data_end_date']}.")
                if p.get('actual_days',31)<p.get('expected_days',31):
                    st.caption(f"*{p['month']} PO scaled from {p['actual_days']} to {p['expected_days']} days")

            # WH summary
            if not is_national and 'brand_wh' in fc_wh.columns and len(fc_wh)>0:
                st.subheader("Warehouse Summary")
                ws=fc_wh.groupby('brand_wh')['pred'].sum().reset_index().rename(columns={'pred':'sales'})
                if len(pa)>0 and 'brand_wh' in pa.columns:
                    wp=pa.groupby('brand_wh')['pred_po'].sum().reset_index().rename(columns={'pred_po':'po'})
                    wc=ws.merge(wp,on='brand_wh',how='outer').fillna(0)
                else: wc=ws
                wc=wc.sort_values('sales',ascending=False)
                fw=go.Figure()
                fw.add_trace(go.Bar(x=wc['brand_wh'].astype(str),y=wc['sales'],name='Sales',marker_color='#5DADE2',
                    hovertemplate='%{y:,.0f}<extra>Sales</extra>'))
                if 'po' in wc.columns:
                    fw.add_trace(go.Bar(x=wc['brand_wh'].astype(str),y=wc['po'],name='PO (MOQ)',marker_color='#85C1E9',
                        hovertemplate='%{y:,.0f}<extra>PO</extra>'))
                fw.update_layout(height=340,template='plotly_white',barmode='group',margin=dict(b=5,t=30),
                    xaxis_title='Warehouse',yaxis_title='Units',
                    legend=dict(orientation='h',yanchor='bottom',y=1.02,x=0.5,xanchor='center'))
                st.plotly_chart(fw,use_container_width=True,key=f"wh_{ti}")

        # ═══════════════════════════════════════════════════
        # SALES DETAIL
        # ═══════════════════════════════════════════════════
        if active_tab == tab_options[1]:
            # Use current tab's gran_key
            fc_d = sfc(platform, mo, gran_key)
            # Fallback: if no data for this gran_key, try wh_sku_weekly (not for national)
            if len(fc_d) == 0 and gran_key != 'wh_sku_weekly' and not is_national:
                fc_d = sfc(platform, mo, 'wh_sku_weekly')
            
            if len(fc_d)==0: st.warning("No data.")
            else:
                is_weekly = 'week_num' in fc_d.columns
                has_wh = 'brand_wh' in fc_d.columns
                
                if is_weekly:
                    wk=fc_d.groupby('week_num')['pred'].sum().reset_index()
                    wk['pred']=wk['pred'].round(0); wk['label']=wk['week_num'].map(wl)
                    wk = wk.dropna(subset=['label'])
                    if len(wk) > 0 and wk['pred'].sum() > 0:
                        st.subheader("Weekly Breakdown")
                        fwk=go.Figure(go.Bar(x=wk['label'],y=wk['pred'],
                            text=wk['pred'].apply(lambda x:f'{x:,.0f}'),textposition='inside',
                            textfont=dict(size=13,color='white'),marker_color='#5DADE2',
                            hovertemplate='%{x}<br>%{y:,.0f} units<extra></extra>'))
                        fwk.update_layout(height=320,template='plotly_white',margin=dict(b=5,t=30),xaxis_title='',yaxis_title='Units')
                        st.plotly_chart(fwk,use_container_width=True,key=f"wk_{ti}")

                st.subheader("Top SKUs")
                sk=fc_d.groupby('item_id')['pred'].sum().reset_index()
                sk['pred']=sk['pred'].round(0); sk['name']=sk['item_id'].apply(lambda x:iname(x,platform))
                sk['lbl']=sk['name'].str[:40]; stop=sk.nlargest(15,'pred')
                fsk=go.Figure(go.Bar(y=stop['lbl'],x=stop['pred'],orientation='h',
                    text=stop['pred'].apply(lambda x:f'{x:,.0f}'),textposition='outside',
                    marker_color='#5DADE2',
                    customdata=stop['name'],
                    hovertemplate='<b>%{customdata}</b><br>%{x:,.0f} units<extra></extra>'))
                fsk.update_layout(height=480,template='plotly_white',margin=dict(l=260,t=30),
                    yaxis={'categoryorder':'total ascending'},xaxis_title='Units',yaxis_title='')
                st.plotly_chart(fsk,use_container_width=True,key=f"sk_{ti}")

                # Top Cities (horizontal bar)
                cfc_cities = D['city_forecast_df']
                cfc_mo = cfc_cities[(cfc_cities['forecast_month']==mo)] if 'forecast_month' in cfc_cities.columns else cfc_cities
                if 'platform' in cfc_mo.columns:
                    cfc_mo = cfc_mo[cfc_mo['platform']==platform]
                if len(cfc_mo) > 0 and 'city_norm' in cfc_mo.columns:
                    st.subheader("Top Cities")
                    ct = cfc_mo.groupby('city_norm')['pred'].sum().reset_index().nlargest(15,'pred')
                    ct['city_label'] = ct['city_norm'].str.title()
                    fct = go.Figure(go.Bar(y=ct['city_label'], x=ct['pred'], orientation='h',
                        text=ct['pred'].apply(lambda x:f'{x:,.0f}'), textposition='outside',
                        marker_color='#58D68D',
                        hovertemplate='<b>%{y}</b><br>%{x:,.0f} units<extra></extra>'))
                    fct.update_layout(height=480, template='plotly_white', margin=dict(l=150, t=30),
                        yaxis={'categoryorder':'total ascending'}, xaxis_title='Units', yaxis_title='')
                    st.plotly_chart(fct, use_container_width=True, key=f"ct_{ti}")
                
                # Key Drivers (real model feature importance)
                fi_mo = D.get('feature_importances',{}).get(platform,{}).get(mo,[])
                if fi_mo:
                    st.subheader("Key Drivers")
                    fi_df = pd.DataFrame(fi_mo).nlargest(12, 'importance')
                    fi_df = fi_df.sort_values('importance', ascending=True)
                    ffi = go.Figure(go.Bar(y=fi_df['label'], x=fi_df['importance']*100, orientation='h',
                        text=fi_df['importance'].apply(lambda x:f'{x*100:.1f}%'), textposition='outside',
                        marker_color='#B4A7D6',
                        hovertemplate='<b>%{y}</b><br>Importance: %{x:.1f}%<extra></extra>'))
                    ffi.update_layout(height=max(350, len(fi_df)*40), template='plotly_white',
                        margin=dict(l=200, t=30), xaxis_title='Feature Importance (%)', yaxis_title='')
                    st.plotly_chart(ffi, use_container_width=True, key=f"fi_{ti}")

                # Forecast Data Table (at the bottom)
                st.subheader("Forecast Data")
                filters = st.columns(3 if (has_wh and is_weekly) else 2)
                col_idx = 0
                if has_wh:
                    with filters[col_idx]:
                        wo=['All']+sorted(fc_d['brand_wh'].unique().tolist())
                        sw=st.selectbox("Warehouse",wo,key=f"sw_{ti}")
                    col_idx += 1
                else:
                    sw = 'All'
                if is_weekly:
                    with filters[col_idx]:
                        wk_opts=['All']+[wl[w] for w in sorted(wl.keys())]
                        sel_wk=st.selectbox("Week",wk_opts,key=f"swk_{ti}")
                    col_idx += 1
                else:
                    sel_wk = 'All'
                with filters[col_idx]:
                    so=['All']+sorted(fc_d['item_id'].unique().tolist())
                    ss=st.selectbox("SKU",so,key=f"ss_{ti}")

                ft=fc_d.copy()
                if sw!='All' and has_wh: ft=ft[ft['brand_wh']==sw]
                if sel_wk!='All' and is_weekly:
                    wk_num = [k for k,v in wl.items() if v==sel_wk][0]
                    ft=ft[ft['week_num']==wk_num]
                if ss!='All': ft=ft[ft['item_id']==ss]
                ft=ft.copy(); ft['item_name']=ft['item_id'].apply(lambda x:iname(x,platform))
                ft['pred']=ft['pred'].round(0).astype(int)
                if is_weekly: ft['week']=ft['week_num'].map(wl)
                cols=[c for c in ['brand_wh','item_id','item_name','week','pred'] if c in ft.columns]
                st.markdown(f"**{len(ft):,} rows | Total: {ft['pred'].sum():,} units**")
                st.dataframe(ft[cols].sort_values('pred',ascending=False),height=400,use_container_width=True,hide_index=True)
                st.download_button("📥 Download CSV",ft[cols].to_csv(index=False),
                    f"sales_{platform}_{mo}.csv","text/csv",key=f"ds_{ti}")

        # ═══════════════════════════════════════════════════
        # PO DETAIL
        # ═══════════════════════════════════════════════════
        if active_tab == tab_options[2]:
            moq_on=st.toggle("MOQ adjusted",True,key=f"mq_{ti}")
            pd_=pa if moq_on else pr
            if len(pd_)==0: st.warning("No data.")
            else:
                has_wh_po = 'brand_wh' in pd_.columns
                is_weekly_po = 'week_num' in pd_.columns
                
                # PO Weekly Breakdown (first chart, not for national × SKU)
                if not is_national and is_weekly_po:
                    pwk=pd_.groupby('week_num')['pred_po'].sum().reset_index()
                    pwk['pred_po']=pwk['pred_po'].round(0); pwk['label']=pwk['week_num'].map(wl)
                    pwk=pwk.dropna(subset=['label'])
                    if len(pwk)>0 and pwk['pred_po'].sum()>0:
                        st.subheader("Weekly PO Breakdown")
                        fpwk=go.Figure(go.Bar(x=pwk['label'],y=pwk['pred_po'],
                            text=pwk['pred_po'].apply(lambda x:f'{x:,.0f}'),textposition='inside',
                            textfont=dict(size=13,color='white'),marker_color='#85C1E9',
                            hovertemplate='%{x}<br>%{y:,.0f} units<extra></extra>'))
                        fpwk.update_layout(height=320,template='plotly_white',margin=dict(b=5,t=30),xaxis_title='',yaxis_title='PO Units')
                        st.plotly_chart(fpwk,use_container_width=True,key=f"pwkc_{ti}")

                if not is_national and has_wh_po and len(fc_wh)>0:
                    st.subheader("PO / Sales Ratio by Warehouse")
                    ws2=fc_wh.groupby('brand_wh')['pred'].sum().reset_index()
                    wp2=pd_.groupby('brand_wh')['pred_po'].sum().reset_index()
                    rt=ws2.merge(wp2,on='brand_wh').sort_values('pred',ascending=False)
                    rt['ratio']=rt['pred_po']/rt['pred'].replace(0,np.nan)
                    
                    fr=go.Figure(go.Bar(x=rt['brand_wh'].astype(str),y=rt['ratio'],
                        text=rt['ratio'].apply(lambda x:f'{x:.2f}'),textposition='outside',
                        textfont=dict(size=11),marker_color='#85C1E9',
                        hovertemplate='WH %{x}<br>Ratio: %{y:.2f}<extra></extra>'))
                    fr.update_layout(height=380,template='plotly_white',margin=dict(t=30,r=30),
                        xaxis_title='Warehouse',yaxis_title='PO/Sales Ratio',
                        yaxis=dict(range=[0, max(rt['ratio'].max()*1.15, 1.3)]))
                    st.plotly_chart(fr,use_container_width=True,key=f"pr_{ti}")
                    st.caption("Ratio > 1 = ordering more than sales forecast. < 1 = under-ordering risk.")

                st.subheader("Top SKUs by PO Volume")
                ps=pd_.groupby('item_id')['pred_po'].sum().reset_index()
                ps['pred_po']=ps['pred_po'].round(0); ps['name']=ps['item_id'].apply(lambda x:iname(x,platform))
                ps['lbl']=ps['name'].str[:40]; ptop=ps.nlargest(15,'pred_po')
                fps=go.Figure(go.Bar(y=ptop['lbl'],x=ptop['pred_po'],orientation='h',
                    text=ptop['pred_po'].apply(lambda x:f'{x:,.0f}'),textposition='outside',
                    marker_color='#85C1E9',
                    customdata=ptop['name'],
                    hovertemplate='<b>%{customdata}</b><br>%{x:,.0f} units<extra></extra>'))
                fps.update_layout(height=480,template='plotly_white',margin=dict(l=260,t=30),
                    yaxis={'categoryorder':'total ascending'},xaxis_title='PO Units',yaxis_title='')
                st.plotly_chart(fps,use_container_width=True,key=f"ps_{ti}")

                st.subheader("PO Data")
                pfilters = st.columns(3 if (has_wh_po and is_weekly_po) else 2)
                pcol = 0
                if has_wh_po:
                    with pfilters[pcol]:
                        pwo=['All']+sorted(pd_['brand_wh'].unique().tolist())
                        pw=st.selectbox("Warehouse",pwo,key=f"pw_{ti}")
                    pcol += 1
                else: pw='All'
                if is_weekly_po:
                    with pfilters[pcol]:
                        pwk_opts=['All']+[wl[w] for w in sorted(wl.keys())]
                        psel_wk=st.selectbox("Week",pwk_opts,key=f"pwk_{ti}")
                    pcol += 1
                else: psel_wk='All'
                with pfilters[pcol]:
                    pso=['All']+sorted(pd_['item_id'].unique().tolist())
                    pss=st.selectbox("SKU",pso,key=f"pss_{ti}")
                
                pf=pd_.copy()
                if pw!='All' and has_wh_po: pf=pf[pf['brand_wh']==pw]
                if psel_wk!='All' and is_weekly_po:
                    pwk_num=[k for k,v in wl.items() if v==psel_wk][0]
                    pf=pf[pf['week_num']==pwk_num]
                if pss!='All': pf=pf[pf['item_id']==pss]
                pf=pf.copy(); pf['item_name']=pf['item_id'].apply(lambda x:iname(x,platform))
                pf['pred_po']=pf['pred_po'].round(0).astype(int)
                if is_weekly_po: pf['week']=pf['week_num'].map(wl)
                pcols=[c for c in ['brand_wh','item_id','item_name','week','pred_po'] if c in pf.columns]
                ml="MOQ adj" if moq_on else "raw"
                st.markdown(f"**{len(pf):,} rows | Total: {pf['pred_po'].sum():,} units ({ml})**")
                st.dataframe(pf[pcols].sort_values('pred_po',ascending=False),height=400,use_container_width=True,hide_index=True)
                st.download_button("📥 Download CSV",pf[pcols].to_csv(index=False),
                    f"po_{platform}_{mo}.csv","text/csv",key=f"dp_{ti}")

        # ═══════════════════════════════════════════════════
        # WHAT-IF SCENARIOS
        # ═══════════════════════════════════════════════════
        if active_tab == tab_options[3]:
            st.subheader("What-If Scenario Builder")
            
            # Get baseline values from precomputed data or RCA fallback
            baseline_disc = None; baseline_ad = None; baseline_osa = None
            
            # Primary: baseline_levers.pkl (precomputed in notebook)
            bl = D.get('baseline_levers',{}).get(platform,{}).get(mo,{})
            if 'own_discount' in bl:
                baseline_disc = bl['own_discount']['value'] * 100
            if 'ad_spend' in bl:
                baseline_ad = bl['ad_spend']['monthly'] / 100000  # to Lakhs
            if 'own_osa' in bl:
                baseline_osa = bl['own_osa']['value'] * 100
            
            # Fallback: RCA lever data
            if baseline_disc is None or baseline_ad is None or baseline_osa is None:
                rca_levers = D.get('po_rca_data',{}).get(platform,{}).get(mo,{}).get('lever_po_impacts',{})
                if baseline_disc is None:
                    for k in ['rpi','own_discount']:
                        if k in rca_levers and rca_levers[k].get('fmt') == 'pct':
                            baseline_disc = rca_levers[k]['assumed'] * 100; break
                if baseline_ad is None:
                    for k in ['sov','ad_spends']:
                        if k in rca_levers and rca_levers[k].get('fmt') != 'pct':
                            m_a = rca_levers[k].get('monthly_assumed')
                            n_w = rca_levers[k].get('n_weeks', 4)
                            if m_a is not None:
                                baseline_ad = m_a * n_w / 100000; break
                if baseline_osa is None:
                    for k in ['own_osa']:
                        if k in rca_levers and rca_levers[k].get('fmt') == 'pct':
                            baseline_osa = rca_levers[k]['assumed'] * 100; break
            
            cfc=D['city_forecast_df']
            fwi=cfc[cfc['forecast_month']==mo] if 'forecast_month' in cfc.columns else cfc
            
            sc1,sc2,sc3,sc4=st.columns(4)
            with sc1:
                st.markdown("**🏷️ Avg Discount**")
                if baseline_disc is not None:
                    st.caption(f"Baseline: {baseline_disc:.1f}%")
                else:
                    st.caption("Baseline: —")
                disc=st.slider("% change from baseline",-50,50,0,5,key=f"dc_{ti}",
                    help="% change in your average discount depth. Translates to RPI (price competitiveness) shift.")
                if baseline_disc is not None:
                    new_disc = baseline_disc * (1 + disc/100)
                    st.markdown(f"→ **{new_disc:.1f}%**" if disc != 0 else f"→ {new_disc:.1f}% *(no change)*")
            with sc2:
                st.markdown("**📣 Ad Spend**")
                if baseline_ad is not None:
                    st.caption(f"Baseline: ₹{baseline_ad:.1f}L/month")
                else:
                    st.caption("Baseline: —")
                adsp=st.slider("% change from baseline",-50,50,0,5,key=f"ac_{ti}",
                    help="% change in monthly ad spend budget. Translates to SOV (share of voice) shift via spend→impression curve.")
                if baseline_ad is not None:
                    new_ad = baseline_ad * (1 + adsp/100)
                    st.markdown(f"→ **₹{new_ad:.1f}L/month**" if adsp != 0 else f"→ ₹{new_ad:.1f}L/month *(no change)*")
            with sc3:
                st.markdown("**📦 On-Shelf Availability**")
                if baseline_osa is not None:
                    st.caption(f"Baseline: {baseline_osa:.1f}%")
                else:
                    st.caption("Baseline: —")
                osa_pct=st.slider("% change from baseline",-50,50,0,5,key=f"osa_{ti}",
                    help="% change in your average OSA. Directly impacts availability gap vs competitors.")
                if baseline_osa is not None:
                    new_osa = baseline_osa * (1 + osa_pct/100)
                    st.markdown(f"→ **{new_osa:.1f}%**" if osa_pct != 0 else f"→ {new_osa:.1f}% *(no change)*")
            with sc4:
                st.markdown("**🏪 Competitor Response**")
                st.caption("How much competitors match")
                comp=st.select_slider("Competitor match %",[0,25,50,75,100],50,
                    format_func=lambda x:f"{x}%",key=f"cr_{ti}",
                    help="0% = competitors don't react. 100% = competitors fully match your changes. Only applies when you change a lever above.")
                comp_labels={0:"They don't react",25:"Minor reaction",50:"Partial match",75:"Mostly match",100:"Full match"}
                st.markdown(f"→ **{comp}% match** *({comp_labels.get(comp,'')})*")
            st.caption("Baselines are avg values used in the forecast (from training data). % changes are applied uniformly across all city×SKU combinations, then run through calibrated S-curves to estimate non-linear sales impact.")
            wi=compute_wi(fwi,disc,adsp,comp,osa_pct)

            if wi is None: st.warning("No data.")
            else:
                gb=D.get('po_features',{}).get(platform,{}).get('global_bias',1.0)
                adj=wi[comp]; base=wi['base']
                eff=(adj/base-1)*100 if base>0 else 0
                ci_lo=min(wi[0],wi[100]); ci_hi=max(wi[0],wi[100])
                # PO impact: scale from sales change
                po_base_val = pt_tot  # Use actual PO forecast from KPI
                po_adj_val = round(po_base_val * (1 + eff/100)) if eff != 0 else po_base_val

                st.subheader("Impact")
                c1,c2,c3,c4=st.columns(4)
                c1.metric("Base Sales",f"{round(base):,}")
                c2.metric("Adjusted Sales",f"{round(adj):,}",f"{eff:+.1f}%")
                c3.metric("Base PO",f"{po_base_val:,}")
                c4.metric("Adjusted PO",f"{po_adj_val:,}",f"{eff:+.1f}%")
                if disc!=0 or adsp!=0 or osa_pct!=0:
                    st.caption(f"Range: {round(ci_lo):,} — {round(ci_hi):,} (0%→100% competitor response)")

                st.markdown("---")
                st.subheader("Elasticity Curves")
                curves=D.get('curves',{})
                lnames={'rpi':'Price Competitiveness (RPI)','sov':'Share of Voice (SOV)','osa':'On-Shelf Availability (OSA)'}
                lcolors={'rpi':'#3498db','sov':'#27ae60','osa':'#e67e22'}
                avail=[l for l in ['rpi','sov','osa'] if l in curves and 'fitted' in curves[l]]
                if avail:
                    fig_e=go.Figure()
                    for lv in avail:
                        f=curves[lv]['fitted']
                        cd=f['data'].sort_values('step').copy()
                        cd['step']=cd['step'].round(1)
                        cd['effect']=cd['effect'].round(2)
                        p=f['params']
                        clr=lcolors.get(lv,'#333')
                        # Legend label: show elasticity + scope info
                        elas_val=p['elasticity']
                        elas_label=f"{elas_val:.2f}%/1%"
                        if lv=='sov':
                            ac=curves[lv].get('ad_active_cities','?')
                            tc=curves[lv].get('total_cities','?')
                            ask=curves[lv].get('ad_active_skus','?')
                            tsk=curves[lv].get('total_skus','?')
                            # The calibrated elasticity is on ad-active subset
                            # Show it clearly as ad-active scoped
                            elas_label=f"{elas_val:.2f}%/1% on ad-active ({ac}/{tc} cities, {ask}/{tsk} SKUs)"
                        hover_texts=[f"Change: {s:+.1f}%<br>Impact: {e:+.2f}%" for s,e in zip(cd['step'],cd['effect'])]
                        fig_e.add_trace(go.Scatter(
                            x=cd['step'],y=cd['effect'],mode='lines',
                            name=f"{lnames.get(lv,lv)} ({elas_label})",
                            line=dict(color=clr,width=2.5),
                            text=hover_texts,hoverinfo='text'
                        ))
                    fig_e.add_hline(y=0,line_dash='dash',line_color='gray',opacity=0.4)
                    fig_e.add_vline(x=0,line_dash='dash',line_color='gray',opacity=0.5)
                    fig_e.update_layout(
                        height=550,template='plotly_white',hovermode='closest',
                        title='Sales Response to Relative Metric Changes',
                        xaxis_title='Change from Current Baseline (%)',
                        yaxis_title='Sales Impact (%)',
                        margin=dict(t=50,b=40),
                        legend=dict(orientation='v',yanchor='bottom',y=0.02,x=0.98,xanchor='right',
                            bgcolor='rgba(255,255,255,0.85)',bordercolor='rgba(0,0,0,0.1)',borderwidth=1,
                            font=dict(size=11))
                    )
                    st.plotly_chart(fig_e,use_container_width=True,key=f"el_{ti}")
                    # Caption with baselines — structured
                    bl_parts = []
                    if baseline_disc is not None: bl_parts.append(f"Avg Discount: {baseline_disc:.1f}%")
                    # SOV: show actual SOV% from curves if available, else ad spend
                    sov_baseline_pct = curves.get('sov',{}).get('baseline_sov_pct')
                    if sov_baseline_pct is not None:
                        bl_parts.append(f"SOV: {sov_baseline_pct:.1f}%")
                        if baseline_ad is not None:
                            bl_parts.append(f"Ad Spend: ₹{baseline_ad:.1f}L/mo")
                    elif baseline_ad is not None:
                        bl_parts.append(f"Ad Spend: ₹{baseline_ad:.1f}L/mo")
                    if baseline_osa is not None: bl_parts.append(f"OSA: {baseline_osa:.1f}%")
                    bl_text = f"**Current baselines:** {' · '.join(bl_parts)}" if bl_parts else ""
                    st.markdown(f"Origin (0,0) = current baseline position. {bl_text}  \n"
                        f"*Slopes are indicative, calibrated from XGBoost model perturbation on historical data. "
                        f"Magnitudes should be treated as directional guidance, not precise predictions.*",
                        help="The model perturbs each metric ±5-10% from baseline and measures the predicted sales change. "
                        "Saturation is estimated from historical data variance.")

        # ═══════════════════════════════════════════════════
        # VALIDATION & RCA
        # ═══════════════════════════════════════════════════
        if active_tab == tab_options[4]:
            # Live validation for THIS month
            lv_all = D.get('live_validation',[])
            lv_mo = [r for r in lv_all if isinstance(r,dict) and r.get('platform')==platform and r.get('month')==mo] if isinstance(lv_all,list) else []
            rca_mo = D.get('rca_data',{}).get(platform,{}).get(mo)
            po_val_plat = pv.get(platform, [])

            # ── Validation Summary ──
            st.subheader("Validation")
            if lv_mo:
                rows=[]
                any_partial = False
                # Filter sales validation rows to match current tab's granularity
                for r in lv_mo:
                    # Match granularity: "WH x SKU" vs "National x SKU"
                    r_gran = r.get('granularity', '')
                    r_is_national = 'national' in r_gran.lower()
                    if r_is_national != is_national:
                        continue
                    partial = r.get('partial', False)
                    if partial: any_partial = True
                    row = {'Type':'Sales' + (' ⚠️' if partial else ''),
                        'Granularity':r['granularity'],'Horizon':r['horizon'],
                        'Row WAPE':f"{r['row_wape']:.1f}%",
                        'Agg WAPE':f"{r['agg_wape']:.1f}%",
                        'Bias':f"{r['bias']:+.1f}%"}
                    rows.append(row)
                # PO validation row — match to current tab's granularity
                po_rca_check = D.get('po_rca_data',{}).get(platform,{}).get(mo)
                if po_rca_check and not is_national:
                    # WH × SKU PO row
                    po_partial = po_rca_check.get('is_partial', False)
                    if po_partial: any_partial = True
                    rows.append({'Type':'PO (MOQ adj)' + (' ⚠️' if po_partial else ''),
                        'Granularity':'WH × SKU','Horizon':hz,
                        'Row WAPE':f"{po_rca_check['wape']:.1f}%",
                        'Agg WAPE':f"{abs(po_rca_check['bias_pct']):.1f}%",
                        'Bias':f"{po_rca_check['bias_pct']:+.1f}%"})
                elif po_rca_check and is_national:
                    # National × SKU PO row — from po_validation results
                    po_partial = po_rca_check.get('is_partial', False)
                    if po_partial: any_partial = True
                    po_val_nat = [v for v in D.get('po_validation',{}).get(platform,[])
                                  if v.get('month') == mo and v.get('granularity') == 'National × SKU']
                    if po_val_nat:
                        pv_r = po_val_nat[0]
                        rows.append({'Type':'PO (MOQ adj)' + (' ⚠️' if pv_r.get('partial') else ''),
                            'Granularity':'National × SKU','Horizon':hz,
                            'Row WAPE':f"{pv_r['wape_sku']*100:.1f}%",
                            'Agg WAPE':f"{abs(pv_r['bias_pct']):.1f}%",
                            'Bias':f"{pv_r['bias_pct']:+.1f}%"})
                if any_partial:
                    st.caption("⚠️ = Partial month — forecast pro-rated to match available days.")
                if rows:
                    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                else:
                    st.info("No validation data for this granularity yet.")
            else:
                st.info("No actuals available for this month yet.")

            # ── PO Error Decomposition (WH×SKU only) ──
            if not is_national:
                st.markdown("---")
                st.subheader("PO Error Decomposition")
                
                po_rca = D.get('po_rca_data',{}).get(platform,{}).get(mo)
            
                if po_rca:
                    # Partial month warning
                    is_partial = po_rca.get('is_partial', False)
                    complete_weeks = po_rca.get('complete_weeks')
                    if is_partial and complete_weeks:
                        st.warning(f"⚠️ Partial month — comparing only complete weeks ({', '.join(f'W{w}' for w in sorted(complete_weeks))}) of forecast vs actuals.")
                
                    # Top-level metrics
                    pc1,pc2,pc3,pc4=st.columns(4)
                    pc1.metric("Actual PO",f"{round(po_rca['total_actual']):,}")
                    pc2.metric("Forecast PO",f"{round(po_rca['total_forecast']):,}")
                    pc3.metric("Net Error",f"{round(po_rca['total_error']):+,}",f"{po_rca['bias_pct']:+.1f}%")
                    pc4.metric("Abs Error (WAPE)",f"{po_rca['wape']:.1f}%")
                
                    tae = po_rca['total_abs_error']
                    ta = po_rca['total_actual']
                    has_cf = po_rca.get('has_counterfactual', False)
                    lever_impacts = po_rca.get('lever_po_impacts', {})
                
                    # ════════════════════════════════════════
                    # WATERFALL TABLE: Additive error decomposition
                    # ════════════════════════════════════════
                    # PO = sales × wh_bias × sku_share
                    # Error sources:
                    #   ① Sales forecast wrong   → controlled_abs (original_PO − counterfactual_PO)
                    #   ② WH bias wrong          → wh_level_error portion of uncontrolled
                    #   ③ SKU share wrong         → sku_distrib_error portion of uncontrolled
                
                    st.markdown("---")
                    st.markdown("### Error Waterfall")
                    level_label = 'National × SKU' if is_national else 'WH × SKU'
                    st.caption(f"How does the total {level_label} error break down?")
                
                    def fmt_lever_value(info):
                        """Format assumed → actual string based on feature type."""
                        if info['fmt'] == 'pct':
                            act = info.get('actual')
                            if act is None or (isinstance(act, float) and np.isnan(act)):
                                return f"{info['assumed']*100:.1f}% → N/A"
                            return f"{info['assumed']*100:.1f}% → {act*100:.1f}%"
                        else:
                            # Ad spend: monthly totals computed from source data
                            m_assumed = info.get('monthly_assumed')
                            m_actual = info.get('monthly_actual')
                            if m_assumed is not None and m_actual is not None:
                                a = m_assumed / 100000
                                ac = m_actual / 100000
                                return f"₹{a:.1f}L → ₹{ac:.1f}L"
                            else:
                                act = info.get('actual')
                                assumed = info.get('assumed', 0)
                                if act is None or (isinstance(act, float) and np.isnan(act)):
                                    return f"{assumed:,.0f} → N/A"
                                return f"{assumed:,.0f} → {act:,.0f}"
                
                    # Build HTML table for bold control
                    rows_html = []
                
                    def add_row(comp, abs_err, wape, pct, change, bold=False):
                        s = 'font-weight:700;' if bold else ''
                        indent = 'padding-left:28px;' if comp.startswith('↳') else ''
                        rows_html.append(
                            f"<tr style='{s}'>"
                            f"<td style='{s}{indent}'>{comp}</td>"
                            f"<td style='{s} text-align:center;'>{abs_err}</td>"
                            f"<td style='{s} text-align:center;'>{wape}</td>"
                            f"<td style='{s} text-align:center;'>{pct}</td>"
                            f"<td style='{s}'>{change}</td>"
                            f"</tr>"
                        )
                
                    # Row 0: Total
                    net_err = po_rca['total_error']
                    net_dir = "over-forecast" if net_err > 0 else "under-forecast"
                    add_row(f'TOTAL PO ERROR ({level_label})',
                        f"{round(tae):,}", f"{po_rca['wape']:.1f}%", '100%',
                        f"Net: {round(net_err):+,} units ({net_dir})", bold=True)
                
                    if has_cf:
                        ctrl_abs = po_rca['controlled_abs']
                        unctrl_abs = po_rca['uncontrolled_abs']
                    
                        # Row 1: Sales forecast error
                        add_row('① Sales Forecast Error',
                            f"{round(ctrl_abs):,}",
                            f"{ctrl_abs/ta*100:.1f}%" if ta > 0 else "—",
                            f"{ctrl_abs/tae*100:.0f}%" if tae > 0 else "—",
                            '', bold=True)
                    
                        # Sub-rows: per-lever
                        for feat, info in lever_impacts.items():
                            change_str = fmt_lever_value(info)
                            add_row(f"↳ {info['label']}",
                                f"{abs(info['po_impact']):,.0f}",
                                f"{abs(info['po_impact'])/ta*100:.1f}%" if ta > 0 else "—",
                                f"{abs(info['po_impact'])/tae*100:.1f}%" if tae > 0 else "—",
                                change_str)
                    
                        # Momentum
                        mom = po_rca.get('momentum_po_impact', 0)
                        if abs(mom) > 0:
                            add_row('↳ Base Trend / Seasonality',
                                f"{abs(mom):,.0f}",
                                f"{abs(mom)/ta*100:.1f}%" if ta > 0 else "—",
                                f"{abs(mom)/tae*100:.1f}%" if tae > 0 else "—",
                                'Organic demand shift beyond lever movements')
                    
                        # Row 2: Platform behaviour error (WH-level only)
                        if not is_national:
                            add_row('② Platform Behaviour Error',
                                f"{round(unctrl_abs):,}",
                                f"{unctrl_abs/ta*100:.1f}%" if ta > 0 else "—",
                                f"{unctrl_abs/tae*100:.0f}%" if tae > 0 else "—",
                                '', bold=True)
                        
                            # Sub-rows: WH bias + SKU share
                            wh_e = po_rca['wh_level_error']
                            sku_e = po_rca['sku_distrib_error']
                            bias_used = po_rca.get('global_bias_used', 0)
                            actual_ratio = po_rca.get('actual_po_sales_ratio', 0)
                        
                            add_row('↳ WH Bias (PO/Sales ratio)',
                                f"{round(wh_e):,}",
                                f"{wh_e/ta*100:.1f}%" if ta > 0 else "—",
                                f"{wh_e/tae*100:.1f}%" if tae > 0 else "—",
                                f"{bias_used:.2f}× → {actual_ratio:.2f}×")
                        
                            add_row('↳ SKU Share Distribution',
                                f"{round(sku_e):,}",
                                f"{sku_e/ta*100:.1f}%" if ta > 0 else "—",
                                f"{sku_e/tae*100:.1f}%" if tae > 0 else "—",
                                'Platform ordered different SKU mix than forecasted')
                
                    else:
                        wh_e = po_rca['wh_level_error']
                        sku_e = po_rca['sku_distrib_error']
                        add_row('① WH-Level Error (bias)',
                            f"{round(wh_e):,}",
                            f"{wh_e/ta*100:.1f}%" if ta > 0 else "—",
                            f"{wh_e/tae*100:.0f}%" if tae > 0 else "—",
                            '—', bold=True)
                        add_row('② SKU Distribution Error',
                            f"{round(sku_e):,}",
                            f"{sku_e/ta*100:.1f}%" if ta > 0 else "—",
                            f"{sku_e/tae*100:.0f}%" if tae > 0 else "—",
                            '—', bold=True)
                
                    table_html = f"""
                    <table style="width:100%; border-collapse:collapse; font-size:14px;">
                    <thead>
                        <tr style="border-bottom:2px solid #ddd; font-weight:600; color:#555;">
                            <th style="text-align:left; padding:8px 12px;">Component</th>
                            <th style="text-align:center; padding:8px 12px;">Abs Error</th>
                            <th style="text-align:center; padding:8px 12px;">WAPE</th>
                            <th style="text-align:center; padding:8px 12px;">% of Total</th>
                            <th style="text-align:left; padding:8px 12px;">Assumed → Actual</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(rows_html)}
                    </tbody>
                    </table>
                    """
                    # Add cell padding via CSS
                    table_html = table_html.replace('<td', '<td style="padding:6px 12px; border-bottom:1px solid #eee;" ')
                    # Fix double style attrs — merge them
                    import re
                    table_html = re.sub(
                        r'style="padding:6px 12px; border-bottom:1px solid #eee;" style="([^"]*)"',
                        r'style="padding:6px 12px; border-bottom:1px solid #eee; \1"',
                        table_html)
                    st.markdown(table_html, unsafe_allow_html=True)
                
                    if not is_national:
                        st.caption("⚠️ ① + ② may exceed total — error components can partially cancel across SKUs.")
                
                    # ════════════════════════════════════════
                    # TOP SKU SHARE MOVERS + WH BREAKDOWN (WH-level only)
                    # ════════════════════════════════════════
                    if not is_national:
                        movers = po_rca.get('top_share_movers', [])
                        if movers:
                            st.markdown("---")
                            st.markdown("### Top SKU Share Shifts")
                            st.caption("Biggest within-WH share changes — explains the SKU Distribution error above.")
                            item_names_map = D.get('item_names',{}).get(platform,{})
                            mrows = []
                            for m in movers:
                                name = item_names_map.get(str(m['item_id']), m['item_id'])
                                mrows.append({
                                    'WH': m['brand_wh'],
                                    'SKU': f"{name}" if name != m['item_id'] else m['item_id'],
                                    'Assumed Share': f"{m['sku_share']*100:.1f}%",
                                    'Actual Share': f"{m['actual_share']*100:.1f}%",
                                    'Shift': f"{m['share_diff']*100:+.1f}pp",
                                    'Units Impact': f"{round(m['impact']):+,}",
                                })
                            st.dataframe(pd.DataFrame(mrows),use_container_width=True,hide_index=True)
                    
                        st.markdown("---")
                        st.markdown("### Warehouse Breakdown")
                        if po_rca.get('wh_comparison'):
                            wh_df = pd.DataFrame(po_rca['wh_comparison'])
                            wh_df = wh_df.sort_values('actual_po', ascending=False)
                            display_rows = []
                            for _,r in wh_df.iterrows():
                                pct_err = r.get('wh_pct_error')
                                display_rows.append({
                                    'Warehouse': r['brand_wh'],
                                    'Actual PO': f"{round(r['actual_po']):,}",
                                    'Forecast PO': f"{round(r['pred_po']):,}",
                                    '% Error': f"{pct_err:+.1f}%" if pd.notna(pct_err) and r['actual_po'] > 0 else "New",
                                    'Bias Used': f"{r['bias_assumed']:.2f}" if pd.notna(r.get('bias_assumed')) else "—",
                                    'Actual Ratio': f"{r['ratio_actual']:.2f}" if pd.notna(r.get('ratio_actual')) else "—",
                                })
                            st.dataframe(pd.DataFrame(display_rows),use_container_width=True,hide_index=True)
                else:
                    # Fall back to po_validation backtest
                    po_val_plat = D.get('po_validation',{}).get(platform,[])
                    if po_val_plat:
                        for v in po_val_plat:
                            st.markdown(f"**{v['test_month'].strftime('%Y-%m')} (backtest)**")
                            vc1,vc2,vc3=st.columns(3)
                            vc1.metric("WH×SKU WAPE",f"{v['wape_sku']*100:.1f}%")
                            vc2.metric("WH WAPE",f"{v['wape_wh']*100:.1f}%")
                            vc3.metric("Bias",f"{v['bias_pct']:+.1f}%")
                            wcomp=v['wh_comparison'].sort_values('actual',ascending=True)
                            fv=go.Figure()
                            fv.add_trace(go.Bar(y=wcomp['brand_wh'].astype(str),x=wcomp['actual'],
                                name='Actual',orientation='h',marker_color='#5DADE2',
                                hovertemplate='%{x:,.0f}<extra>Actual</extra>'))
                            fv.add_trace(go.Bar(y=wcomp['brand_wh'].astype(str),x=wcomp['forecast'],
                                name='Forecast',orientation='h',marker_color='#E74C3C',
                                hovertemplate='%{x:,.0f}<extra>Forecast</extra>'))
                            fv.update_layout(height=350,template='plotly_white',barmode='group',margin=dict(t=30),
                                xaxis_title='Units',yaxis_title='Warehouse')
                            st.plotly_chart(fv,use_container_width=True,key=f"pv_{ti}")
                    else:
                        st.info("No PO validation data.")

            # ── National-level PO RCA (simplified) ──
            if is_national:
                st.markdown("---")
                st.subheader("PO Error Decomposition")
                
                po_rca = D.get('po_rca_data',{}).get(platform,{}).get(mo)
                
                if po_rca:
                    is_partial_n = po_rca.get('is_partial', False)
                    complete_weeks_n = po_rca.get('complete_weeks')
                    if is_partial_n and complete_weeks_n:
                        st.warning(f"⚠️ Partial month — comparing only complete weeks ({', '.join(f'W{w}' for w in sorted(complete_weeks_n))}) of forecast vs actuals.")
                    
                    # National metrics from wh_comparison aggregation
                    wh_data = po_rca.get('wh_comparison', [])
                    if wh_data:
                        nat_actual = sum(w.get('actual_po', 0) for w in wh_data)
                        nat_forecast = sum(w.get('pred_po', 0) for w in wh_data)
                        nat_error = nat_forecast - nat_actual
                        nat_abs_error = abs(nat_error)
                        nat_bias_pct = nat_error / nat_actual * 100 if nat_actual > 0 else 0
                    else:
                        nat_actual = po_rca['total_actual']
                        nat_forecast = po_rca['total_forecast']
                        nat_error = po_rca['total_error']
                        nat_abs_error = abs(nat_error)
                        nat_bias_pct = po_rca['bias_pct']
                    
                    pc1,pc2,pc3=st.columns(3)
                    pc1.metric("Actual PO",f"{round(nat_actual):,}")
                    pc2.metric("Forecast PO",f"{round(nat_forecast):,}")
                    pc3.metric("Net Error",f"{round(nat_error):+,}",f"{nat_bias_pct:+.1f}%")
                    
                    # Waterfall
                    has_cf = po_rca.get('has_counterfactual', False)
                    lever_impacts = po_rca.get('lever_po_impacts', {})
                    # Use WH×SKU total abs error as base (consistent with how ①② were computed)
                    tae = po_rca['total_abs_error']
                    ta = po_rca['total_actual']
                    
                    st.markdown("---")
                    st.markdown("### Error Waterfall")
                    st.caption("PO = Sales × Global Bias. Decomposition uses WH×SKU level errors (which partially cancel at national level).")
                    
                    def fmt_lever_val_n(info):
                        if info['fmt'] == 'pct':
                            return f"{info['assumed']*100:.1f}% → {info['actual']*100:.1f}%"
                        else:
                            m_assumed = info.get('monthly_assumed')
                            m_actual = info.get('monthly_actual')
                            n_weeks = info.get('n_weeks', 1)
                            if m_assumed is not None and m_actual is not None:
                                a = m_assumed / 100000
                                ac = m_actual / 100000
                                return f"₹{a:.1f}L → ₹{ac:.1f}L"
                            return f"{info['assumed']:,.0f} → {info['actual']:,.0f}"
                    
                    rows_html = []
                    def add_row_n(comp, abs_err, wape, pct, change, bold=False):
                        s = 'font-weight:700;' if bold else ''
                        indent = 'padding-left:28px;' if comp.startswith('↳') else ''
                        rows_html.append(
                            f"<tr style='{s}'>"
                            f"<td style='{s}{indent}'>{comp}</td>"
                            f"<td style='{s} text-align:right;'>{abs_err}</td>"
                            f"<td style='{s} text-align:right;'>{wape}</td>"
                            f"<td style='{s} text-align:right;'>{pct}</td>"
                            f"<td style='{s}'>{change}</td>"
                            f"</tr>"
                        )
                    
                    add_row_n('TOTAL PO ERROR (WH × SKU basis)',
                        f"{round(tae):,}",
                        f"{po_rca['wape']:.1f}%",
                        '100%',
                        f"National net: {round(nat_error):+,} units ({nat_bias_pct:+.1f}%)", bold=True)
                    
                    if has_cf:
                        ctrl_abs = po_rca['controlled_abs']
                        unctrl_abs = po_rca['uncontrolled_abs']
                        
                        # ① Sales Forecast Error
                        add_row_n('① Sales Forecast Error',
                            f"{round(ctrl_abs):,}",
                            f"{ctrl_abs/ta*100:.1f}%" if ta > 0 else "—",
                            f"{ctrl_abs/tae*100:.0f}%" if tae > 0 else "—",
                            '', bold=True)
                        
                        # Sub-rows: per-lever
                        for feat, info in lever_impacts.items():
                            add_row_n(f"↳ {info['label']}",
                                f"{abs(info['po_impact']):,.0f}",
                                f"{abs(info['po_impact'])/ta*100:.1f}%" if ta > 0 else "—",
                                f"{abs(info['po_impact'])/tae*100:.1f}%" if tae > 0 else "—",
                                fmt_lever_val_n(info))
                        
                        # Momentum
                        mom = po_rca.get('momentum_po_impact', 0)
                        if abs(mom) > 0:
                            add_row_n('↳ Momentum / Seasonality / Model',
                                f"{abs(mom):,.0f}",
                                f"{abs(mom)/ta*100:.1f}%" if ta > 0 else "—",
                                f"{abs(mom)/tae*100:.1f}%" if tae > 0 else "—",
                                'Trend + unexplained')
                        
                        # ② Global Bias Error
                        bias_used = po_rca.get('global_bias_used', 0)
                        actual_ratio = po_rca.get('actual_po_sales_ratio', 0)
                        add_row_n('② Global Bias Error',
                            f"{round(unctrl_abs):,}",
                            f"{unctrl_abs/ta*100:.1f}%" if ta > 0 else "—",
                            f"{unctrl_abs/tae*100:.0f}%" if tae > 0 else "—",
                            f"Bias: {bias_used:.2f}× → Actual: {actual_ratio:.2f}×", bold=True)
                    
                    table_html = f"""
                    <table style="width:100%; border-collapse:collapse; font-size:14px;">
                    <thead>
                        <tr style="border-bottom:2px solid #ddd; font-weight:600; color:#555;">
                            <th style="text-align:left; padding:8px 12px;">Component</th>
                            <th style="text-align:center; padding:8px 12px;">Abs Error</th>
                            <th style="text-align:center; padding:8px 12px;">WAPE</th>
                            <th style="text-align:center; padding:8px 12px;">% of Total</th>
                            <th style="text-align:left; padding:8px 12px;">Assumed → Actual</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(rows_html)}
                    </tbody>
                    </table>
                    """
                    import re
                    table_html = table_html.replace('<td', '<td style="padding:6px 12px; border-bottom:1px solid #eee;" ')
                    table_html = re.sub(
                        r'style="padding:6px 12px; border-bottom:1px solid #eee;" style="([^"]*)"',
                        r'style="padding:6px 12px; border-bottom:1px solid #eee; \1"',
                        table_html)
                    st.markdown(table_html, unsafe_allow_html=True)
                    
                    st.caption("⚠️ ① + ② may exceed total — error components can partially cancel across SKUs.")
                else:
                    st.info("No PO RCA data available for this month.")