from sklearn.metrics import silhouette_score, davies_bouldin_score
import pandas as pd

def get_metrics(data, labels):

    if len(set(labels)) < 2: 
        return {'rating': 'Error', 'silhouette': 0, 'davies_bouldin': 0}

    sil = silhouette_score(data, labels)
    db = davies_bouldin_score(data, labels)
    

    if sil > 0.5: rating = "Excellent"
    elif sil > 0.25: rating = "Good"
    elif sil > 0: rating = "Fair"
    else: rating = "Poor"
    
    return {'silhouette': sil, 'davies_bouldin': db, 'rating': rating}

def get_cluster_profiles(df, labels):

    df_temp = df.copy()
    df_temp['Cluster'] = labels
    
 
    numeric_cols = df_temp.select_dtypes(include=['number']).columns
    profile = df_temp.groupby('Cluster')[numeric_cols].mean()
    
    profile['Count'] = df_temp['Cluster'].value_counts().sort_index()
    return profile

def get_recommendations(profile):

    recs = {}
    global_means = profile.mean()
    
    for cluster_id, row in profile.iterrows():
        strategies = []
        

        impacts = []
        
        for col in row.index:
            if col == 'Count': continue
            
            cluster_val = row[col]
            global_val = global_means[col]
            if global_val == 0: global_val = 1e-9
            

            diff = (cluster_val - global_val) / abs(global_val)
            impacts.append((col, diff))
            

        impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        top_features = impacts[:3] 
        
        for feature, diff in top_features:
            direction = "High" if diff > 0 else "Low"
            clean_name = feature.replace("_", " ").title()
            feat_lower = feature.lower()
            

            if any(x in feat_lower for x in ['spend', 'amount', 'bill', 'charge', 'profit', 'sales', 'income', 'revenue']):
                if direction == "High":
                    strategies.append(f"💎 **High {clean_name}:** Premium value group. Upsell higher-tier services or offer VIP perks.")
                else:
                    strategies.append(f"💸 **Low {clean_name}:** Price sensitive. Use discounts, bundles, or value-promotions.")
            

            elif any(x in feat_lower for x in ['freq', 'visit', 'count', 'order', 'session', 'click', 'view']):
                if direction == "High":
                    strategies.append(f"🔥 **High {clean_name}:** Very active users. Enroll in loyalty programs and ask for reviews.")
                else:
                    strategies.append(f"💤 **Low {clean_name}:** Risk of churn. Send re-engagement emails ('We miss you').")
            
 
            elif any(x in feat_lower for x in ['tenure', 'age', 'year', 'month', 'day', 'duration', 'time']):
                if direction == "High":
                    strategies.append(f"📅 **High {clean_name}:** Established/Mature. Focus on retention and community building.")
                else:
                    strategies.append(f"🆕 **Low {clean_name}:** New or fresh. Needs onboarding, tutorials, and welcome offers.")
            

            else:
                if direction == "High":
                    strategies.append(f"📈 **High {clean_name}:** This group is defined by high {clean_name}. Leverage this specific strength.")
                else:
                    strategies.append(f"📉 **Low {clean_name}:** Below average {clean_name}. Consider strategies to improve this metric.")

        if not strategies:
            recs[cluster_id] = "Standard Segment: No extreme behaviors detected."
        else:
            recs[cluster_id] = " | ".join(strategies[:2])
            
    return recs

def generate_segment_names(profile):
    """Auto-generates short names like 'High-Spender' or 'New-User'."""
    names = {}
    global_means = profile.mean()
    
    for cid, row in profile.iterrows():
 
        max_diff = 0
        best_tag = "Standard"
        
        for col in row.index:
            if col == 'Count': continue
            
            global_val = global_means[col]
            if global_val == 0: global_val = 1e-9
            
            diff = (row[col] - global_val) / abs(global_val)
            

            if abs(diff) > 0.25 and abs(diff) > max_diff:
                max_diff = abs(diff)
                direction = "High" if diff > 0 else "Low"
                

                simple_name = col.replace("_", "").replace("Score", "").replace("Total", "").replace("Annual", "")
                if 'spend' in col.lower(): simple_name = "Spender"
                if 'income' in col.lower(): simple_name = "Income"
                if 'tenure' in col.lower(): simple_name = "Tenure"
                
                best_tag = f"{direction}-{simple_name}"
        
        names[cid] = f"{best_tag} Segment"
            
    return names