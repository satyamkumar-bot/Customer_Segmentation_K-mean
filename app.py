import streamlit as st
import pandas as pd
from utils import data_loader, preprocessing, feature_engineering, clustering, evaluation, visualization

st.set_page_config(page_title="Customer Segmentation", layout="wide", page_icon="📊")


st.markdown("<h1 style='text-align: center;'>Customer Segmentation Dashboard</h1>", unsafe_allow_html=True)

st.title("👥 Customer Segmentation Dashboard")

if 'data' not in st.session_state: st.session_state.data = None
if 'results' not in st.session_state: st.session_state.results = None

tab1, tab2, tab3, tab4 = st.tabs(["📂 Upload", "⚙️ Analysis", "📊 Results", "📥 Download"])


with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])
       
        
    with col2:
        if uploaded_file:
            df = data_loader.load_data(uploaded_file)
            valid, msg = data_loader.validate_data(df)
            
            if valid:
                st.session_state.data = df
                st.success("✅ Data Loaded Successfully")
                st.dataframe(df.head(), use_container_width=True)
                summ = data_loader.get_data_summary(df)
                c1, c2 = st.columns(2)
                c1.metric("Total Rows", summ['rows'])
                c2.metric("Total Columns", summ['cols'])
            else:
                st.error(f"❌ {msg}")

with tab2:
    if st.session_state.data is not None:
        st.header("1️⃣ Data Processing")
        
        st.info("🤖 **Engine Status:** Isolation Forest is active. Optimized for 95% data retention and high-value 'Whale' detection.")
        
        df_clean = preprocessing.handle_missing_values(st.session_state.data)
        
        df_final, df_outliers = preprocessing.remove_outliers(df_clean, method="isolation_forest")
        
        c1, c2 = st.columns(2)
        c1.metric("Active Data Rows", len(df_final))
        c2.metric("Anomalies Removed", len(df_outliers))
        
        if not df_outliers.empty:
            with st.expander("🔍 View Removed Data (Anomalies)"):
                st.dataframe(df_outliers.head(), width='stretch')

        st.divider()
        
        st.header("2️⃣ Feature Selection")
        
        features_df = feature_engineering.auto_feature_selection(df_final)
        default_cols = feature_engineering.get_priority_features(features_df.columns)
        
        selected_cols = st.multiselect(
            "Select features to cluster:", 
            options=features_df.columns.tolist(), 
            default=default_cols
        )
        
        st.divider()
        
        if st.button("🚀 Run Segmentation Analysis", type="primary"):
            if len(selected_cols) < 2:
                st.error("Select at least 2 features to build a multi-dimensional cluster.")
            else:
                with st.spinner("AI Engine Clustering..."):
                    final_feats = features_df[selected_cols]
                    scaled, scaler = preprocessing.scale_data(final_feats)
                    
        
                    best_k, wcss, sil_scores, k_range = clustering.find_optimal_k(scaled)
                    labels, model = clustering.run_clustering(scaled, best_k)
                    
                    st.session_state.results = {
                        'labels': labels, 'k': best_k, 'wcss': wcss, 'sil_scores': sil_scores, 
                        'k_range': k_range, 'scaled': scaled, 'raw': final_feats, 'clean': df_final
                    }
                    st.success(f"Segmentation Complete! Optimized for {best_k} distinct business personas.")
                    
                    st.plotly_chart(visualization.plot_elbow(wcss, k_range), width='stretch')




with tab3:
    if st.session_state.results:
        res = st.session_state.results
        metrics = evaluation.get_metrics(res['scaled'], res['labels'])
        
        st.subheader("📋 Data Preview")
        preview = res['clean'].copy()
        preview['Cluster'] = res['labels']
        st.dataframe(preview.head(), use_container_width=True)
        st.divider()

        m1, m2, m3 = st.columns(3)
        m1.metric("Segments", res['k'])
        m2.metric("Silhouette Score", f"{metrics['silhouette']:.3f}")
        m3.metric("Quality", metrics['rating'])
        
        c1, c2 = st.columns(2)
        c1.plotly_chart(visualization.plot_clusters_2d(res['scaled'], res['labels']), use_container_width=True)
        c2.plotly_chart(visualization.plot_cluster_dist(res['labels']), use_container_width=True)

        st.subheader("💡 Business Recommendations")
        profile = evaluation.get_cluster_profiles(res['raw'], res['labels'])
        recs = evaluation.get_recommendations(profile)
        names = evaluation.generate_segment_names(profile)
        
        for cid, row in profile.iterrows():
            with st.expander(f"🔵 {names.get(cid, str(cid))}", expanded=True):
                st.dataframe(row.to_frame().T, hide_index=True)
                st.info(recs[cid])


with tab4:
    if st.session_state.results:
        st.header("📥 Export")
        final_df = st.session_state.results['clean'].copy()
        final_df['Cluster'] = st.session_state.results['labels']
        profile = evaluation.get_cluster_profiles(st.session_state.results['raw'], st.session_state.results['labels'])
        names = evaluation.generate_segment_names(profile)
        final_df['Segment'] = final_df['Cluster'].map(names)
        
        csv = final_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "segments.csv", "text/csv", type="primary")


