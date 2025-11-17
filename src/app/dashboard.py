"""Dashboard for historical Phase II to Phase III trial data."""
import streamlit as st
import pandas as pd
import plotly.express as px
from src.app.data_loader import load_historical_trials


def render_dashboard():
    """Render the dashboard."""
    st.title("📊 Phase II → Phase III Historical Data")
    st.markdown("Explore 6,604 historical Phase II trials and their Phase III outcomes.")
    
    df = load_data()
    
    # Filters
    st.sidebar.header("🔍 Filters")
    st.sidebar.info("💡 Use filters below to analyze the 6,604 historical Phase II→III trials displayed on the right")
    st.sidebar.markdown("---")
    
    # Outcome status filter
    outcome_status = st.sidebar.radio(
        "Outcome Status",
        ["All", "Success", "Failure"],
        help="Filter trials based on Phase III outcome (success or failure)"
    )
    
    # Text search
    search = st.sidebar.text_input(
        "🔎 Search Keywords",
        placeholder="e.g., pembrolizumab, lung cancer...",
        help="Search across interventions, conditions, and trial titles"
    )
    
    # Sponsor filter
    org = st.sidebar.selectbox(
        "🏢 Sponsor Type",
        ["All"] + sorted(df["org_class"].unique().tolist()),
        help="Filter by organization type (INDUSTRY, NIH, etc.)"
    )
    
    # Primary purpose filter
    purpose = st.sidebar.selectbox(
        "🎯 Primary Purpose",
        ["All"] + sorted([x for x in df["primary_purpose"].unique() if x != "Unknown"]),
        help="Filter by the primary purpose of the trial"
    )
    
    # Outcome filter
    outcome = st.sidebar.radio(
        "✅ Actual Phase III Outcome",
        ["All", "Success", "Failure"],
        help="Filter by actual Phase III success/failure (only for trials with known outcomes)"
    )
    
    # Date range filter
    st.sidebar.markdown("📅 **Trial Start Date Range**")
    enable_date_filter = st.sidebar.checkbox("Enable date filtering", value=False)
    
    if enable_date_filter:
        df_with_dates = df[df["start_date"] != "Unknown"].copy()
        df_with_dates["parsed_date"] = pd.to_datetime(df_with_dates["start_date"], errors="coerce")
        valid_dates = df_with_dates["parsed_date"].dropna()
        
        if len(valid_dates) > 0:
            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                start_date = st.date_input("From", min_date, min_value=min_date, max_value=max_date)
            with col2:
                end_date = st.date_input("To", max_date, min_value=min_date, max_value=max_date)
            
            date_range = (start_date, end_date)
        else:
            st.sidebar.warning("No valid dates found")
            date_range = None
    else:
        date_range = None
    
    # Cancer type multi-select
    st.sidebar.markdown("🎯 **Cancer Type Filter**")
    top_cancers = df["conditions"].value_counts().head(20).index.tolist()
    selected_cancers = st.sidebar.multiselect(
        "Select cancer types (optional)",
        top_cancers,
        help="Filter by specific cancer types. Leave empty to show all."
    )
    
    # Apply filters
    filtered = df.copy()
    
    # Outcome status
    if outcome_status == "Success":
        filtered = filtered[filtered["actual_success"] == 1]
    elif outcome_status == "Failure":
        filtered = filtered[filtered["actual_success"] == 0]
    
    # Text search
    if search:
        filtered = filtered[
            filtered["interventions"].str.contains(search, case=False, na=False) | 
            filtered["conditions"].str.contains(search, case=False, na=False) |
            filtered["brief_title"].str.contains(search, case=False, na=False)
        ]
    
    # Sponsor type
    if org != "All":
        filtered = filtered[filtered["org_class"] == org]
    
    # Primary purpose
    if purpose != "All":
        filtered = filtered[filtered["primary_purpose"] == purpose]
    
    # Actual outcome
    if outcome == "Success":
        filtered = filtered[filtered["actual_success"] == 1]
    elif outcome == "Failure":
        filtered = filtered[filtered["actual_success"] == 0]
    
    # Date range
    if date_range is not None:
        filtered_dates = filtered[filtered["start_date"] != "Unknown"].copy()
        filtered_dates["parsed_date"] = pd.to_datetime(filtered_dates["start_date"], errors="coerce")
        filtered_dates = filtered_dates.dropna(subset=["parsed_date"])
        filtered_dates = filtered_dates[
            (filtered_dates["parsed_date"].dt.date >= date_range[0]) &
            (filtered_dates["parsed_date"].dt.date <= date_range[1])
        ]
        filtered = filtered_dates
    
    # Cancer types
    if selected_cancers:
        filtered = filtered[filtered["conditions"].isin(selected_cancers)]
    
    # Metrics with annotations
    st.markdown("### 📊 Summary Metrics")
    st.caption("📌 Key statistics for the filtered dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Total Trials", 
            f"{len(filtered):,}",
            help="Total number of trials matching current filters"
        )
    
    known = filtered[filtered["outcome_known"] == True]
    success_count = known["actual_success"].sum() if len(known) > 0 else 0
    
    with col2:
        st.metric(
            "Successful Trials", 
            f"{int(success_count):,}",
            help="Number of trials that successfully progressed to Phase III"
        )
    
    with col3:
        if len(known) > 0:
            st.metric(
                "Success Rate", 
                f"{known['actual_success'].mean():.1%}",
                help="Percentage of trials with known outcomes that succeeded"
            )
        else:
            st.metric("Success Rate", "N/A")
    
    with col4:
        st.metric(
            "Top Sponsor Type", 
            filtered["org_class"].mode()[0] if len(filtered) > 0 else "N/A",
            help="Most common sponsor type in filtered results"
        )
    
    # Visualizations
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "📊 Overview", 
        "🏢 Sponsor Analysis", 
        "💊 Top Interventions", 
        "📈 Advanced Analytics",
        "🎯 Cancer Type Deep Dive",
        "🔬 Intervention Patterns",
        "📅 Temporal Trends",
        "📋 Trial Status Analysis",
        "🌐 Organization Insights",
        "🔗 Correlation Matrix"
    ])
    
    with tab1:
        st.caption("📌 High-level overview of trial outcomes and success patterns across different purposes")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Outcome Distribution")
            if len(filtered) > 0:
                outcome_counts = filtered["outcome_label"].value_counts()
                colors = {"✅ Success": "#2ca02c", "⚠️ Failure": "#d62728", "Unknown": "#7f7f7f"}
                fig = px.pie(values=outcome_counts.values, names=outcome_counts.index, 
                           hole=0.4, color=outcome_counts.index, color_discrete_map=colors)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, width="stretch")
        with col2:
            st.markdown("#### Success Rate by Primary Purpose")
            if len(known) > 0:
                purpose_stats = known.groupby("primary_purpose")["actual_success"].agg(["mean", "count"]).reset_index()
                purpose_stats = purpose_stats[purpose_stats["count"] >= 5].sort_values("mean", ascending=False).head(10)
                fig = px.bar(purpose_stats, x="primary_purpose", y="mean", 
                           text=purpose_stats["mean"].apply(lambda x: f"{x:.0%}"),
                           color="mean", color_continuous_scale="RdYlGn")
                fig.update_yaxes(tickformat=".0%", title="Success Rate")
                fig.update_xaxes(title="Primary Purpose")
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, width="stretch")
    
    with tab2:
        st.caption("📌 Analysis of sponsor types: success rates, trial volumes, and performance comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Success Rate by Sponsor Type")
            if len(known) > 0:
                sponsor_stats = known.groupby("org_class")["actual_success"].agg(["mean", "count"]).reset_index()
                if len(sponsor_stats) > 0:
                    sponsor_stats = sponsor_stats.sort_values("mean", ascending=False)
                    fig = px.bar(sponsor_stats, x="org_class", y="mean", 
                               text=sponsor_stats["mean"].apply(lambda x: f"{x:.0%}"),
                               color="mean", color_continuous_scale="Viridis",
                               labels={"org_class": "Sponsor Type", "mean": "Success Rate"})
                    fig.update_yaxes(tickformat=".0%")
                    fig.update_traces(textposition="outside")
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("No data available for this filter combination")
        with col2:
            st.markdown("#### Trial Volume by Sponsor")
            if len(filtered) > 0:
                sponsor_counts = filtered["org_class"].value_counts().head(10)
                if len(sponsor_counts) > 0:
                    fig = px.bar(x=sponsor_counts.index, y=sponsor_counts.values,
                               labels={"x": "Sponsor Type", "y": "Number of Trials"},
                               color=sponsor_counts.values, color_continuous_scale="Blues")
                    fig.update_traces(text=sponsor_counts.values, textposition="outside")
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("No data available")
            else:
                st.info("No data available")
    
    with tab3:
        st.markdown("#### Top 20 Interventions by Frequency")
        if len(filtered) > 0:
            intervention_counts = filtered["interventions"].value_counts().head(20)
            if len(intervention_counts) > 0:
                fig = px.bar(y=intervention_counts.index, x=intervention_counts.values, orientation="h",
                           labels={"x": "Number of Trials", "y": "Intervention"},
                           color=intervention_counts.values, color_continuous_scale="Oranges")
                fig.update_traces(text=intervention_counts.values, textposition="outside")
                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No intervention data available")
        else:
            st.info("No data available")
        
        st.markdown("#### Top 20 Cancer Types by Frequency")
        if len(filtered) > 0:
            condition_counts = filtered["conditions"].value_counts().head(20)
            if len(condition_counts) > 0:
                fig = px.bar(y=condition_counts.index, x=condition_counts.values, orientation="h",
                           labels={"x": "Number of Trials", "y": "Cancer Type"},
                           color=condition_counts.values, color_continuous_scale="Purples")
                fig.update_traces(text=condition_counts.values, textposition="outside")
                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No cancer type data available")
        else:
            st.info("No data available")
    
    with tab4:
        st.markdown("#### Statistical Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Success Rate Distribution by Sponsor")
            if len(known) > 0:
                sponsor_success = known.groupby("org_class")["actual_success"].agg(["mean", "count", "std"]).reset_index()
                if len(sponsor_success) > 0:
                    sponsor_success["std"] = sponsor_success["std"].fillna(0)
                    fig = px.scatter(sponsor_success, x="count", y="mean", size="count", 
                                   hover_data=["org_class", "std"],
                                   labels={"count": "Number of Trials", "mean": "Success Rate"},
                                   color="mean", color_continuous_scale="RdYlGn")
                    fig.update_yaxes(tickformat=".0%")
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("Insufficient data for analysis")
            else:
                st.info("No trials with known outcomes")
        
        with col2:
            st.markdown("##### Trials Over Time")
            df_dates = filtered[filtered["start_date"] != "Unknown"].copy()
            if len(df_dates) > 0:
                df_dates["year"] = pd.to_datetime(df_dates["start_date"], errors="coerce").dt.year
                df_dates = df_dates.dropna(subset=["year"])
                if len(df_dates) > 0:
                    yearly = df_dates.groupby("year").size().reset_index(name="count")
                    fig = px.line(yearly, x="year", y="count", markers=True,
                                labels={"year": "Year", "count": "Number of Trials"})
                    fig.update_traces(line=dict(width=3))
                    st.plotly_chart(fig, width="stretch")
        
        st.markdown("##### Success Rate Comparison: Top Interventions")
        if len(known) > 0:
            # Get top 15 interventions by frequency
            top_interventions = known["interventions"].value_counts().head(15).index
            if len(top_interventions) > 0:
                intervention_success = known[known["interventions"].isin(top_interventions)].groupby("interventions")["actual_success"].agg(["mean", "count"]).reset_index()
                if len(intervention_success) > 0:
                    intervention_success = intervention_success.sort_values("mean", ascending=True)
                    
                    fig = px.bar(intervention_success, y="interventions", x="mean", orientation="h",
                               text=intervention_success["mean"].apply(lambda x: f"{x:.0%}"),
                               color="mean", color_continuous_scale="RdYlGn",
                               labels={"mean": "Success Rate", "interventions": "Intervention"})
                    fig.update_xaxes(tickformat=".0%")
                    fig.update_traces(textposition="outside")
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("Insufficient intervention data")
            else:
                st.info("No interventions found")
        else:
            st.info("No trials with known outcomes")
    
    with tab5:
        st.caption("📌 Comprehensive cancer type analysis: success rates, volumes, and distribution patterns")
        st.markdown("#### 🎯 Cancer Type Deep Dive")
        
        # Full width horizontal bar chart for better readability
        st.markdown("##### Top 15 Cancer Types by Success Rate")
        if len(known) > 0:
            # Filter to cancer types with at least 5 trials
            cancer_stats = known.groupby("conditions")["actual_success"].agg(["mean", "count"]).reset_index()
            cancer_stats = cancer_stats[cancer_stats["count"] >= 5].sort_values("mean", ascending=True).head(15)
            
            if len(cancer_stats) > 0:
                fig = px.bar(cancer_stats, y="conditions", x="mean", orientation="h",
                           text=cancer_stats["mean"].apply(lambda x: f"{x:.0%}"),
                           color="mean", color_continuous_scale="Viridis",
                           labels={"conditions": "Cancer Type", "mean": "Success Rate"})
                fig.update_xaxes(tickformat=".0%", title="Success Rate")
                fig.update_yaxes(title="")
                fig.update_traces(textposition="outside")
                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("Insufficient cancer type data (need at least 5 trials per type)")
        else:
            st.info("No trials with known outcomes")
        
        # Two columns for scatter and treemap
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Trial Volume vs Success Rate")
            if len(known) > 0:
                cancer_bubble = known.groupby("conditions")["actual_success"].agg(["mean", "count"]).reset_index()
                cancer_bubble = cancer_bubble[cancer_bubble["count"] >= 3]
                
                if len(cancer_bubble) > 0:
                    fig = px.scatter(cancer_bubble, x="count", y="mean", size="count",
                                   hover_data=["conditions"],
                                   labels={"count": "Number of Trials", "mean": "Success Rate"},
                                   color="mean", color_continuous_scale="Turbo",
                                   size_max=50)
                    fig.update_yaxes(tickformat=".0%")
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("Insufficient data for scatter plot")
            else:
                st.info("No trials with known outcomes")
        
        with col2:
            st.markdown("##### Cancer Type Distribution (Treemap)")
            if len(filtered) > 0:
                cancer_treemap = filtered["conditions"].value_counts().head(30).reset_index()
                if len(cancer_treemap) > 0:
                    cancer_treemap.columns = ["condition", "count"]
                    fig = px.treemap(cancer_treemap, path=["condition"], values="count",
                                   color="count", color_continuous_scale="Greens",
                                   labels={"count": "Number of Trials"})
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("No cancer type data")
            else:
                st.info("No data available")
    
    with tab6:
        st.markdown("#### 🔬 Intervention Patterns")
        
        st.markdown("##### Combination vs Single Agent Therapies")
        if len(known) > 0:
            # Identify combination therapies
            known_copy = known.copy()
            known_copy["is_combination"] = known_copy["interventions"].str.contains(",|\\+|and", case=False, na=False)
            combo_stats = known_copy.groupby("is_combination")["actual_success"].agg(["mean", "count"]).reset_index()
            combo_stats["therapy_type"] = combo_stats["is_combination"].map({True: "Combination", False: "Single Agent"})
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(combo_stats, x="therapy_type", y="mean",
                           text=combo_stats["mean"].apply(lambda x: f"{x:.0%}"),
                           color="mean", color_continuous_scale="RdYlGn",
                           labels={"therapy_type": "Therapy Type", "mean": "Success Rate"})
                fig.update_yaxes(tickformat=".0%")
                fig.update_traces(textposition="outside")
                fig.update_layout(height=400)
                st.plotly_chart(fig, width="stretch")
            
            with col2:
                fig = px.pie(combo_stats, values="count", names="therapy_type",
                           hole=0.4, color_discrete_sequence=["#ff7f0e", "#1f77b4"])
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=400)
                st.plotly_chart(fig, width="stretch")
        
        st.markdown("##### Most Common Intervention Keywords")
        if len(filtered) > 0:
            # Extract keywords from interventions
            all_interventions = " ".join(filtered["interventions"].str.lower()).split()
            # Filter out common words and count
            from collections import Counter
            keyword_counts = Counter([word.strip(",.+-") for word in all_interventions if len(word) > 3])
            top_keywords = pd.DataFrame(keyword_counts.most_common(30), columns=["keyword", "count"])
            
            fig = px.bar(top_keywords, y="keyword", x="count", orientation="h",
                       color="count", color_continuous_scale="Plasma",
                       labels={"keyword": "Keyword", "count": "Frequency"})
            fig.update_layout(height=600, showlegend=False)
            fig.update_traces(textposition="outside", text=top_keywords["count"])
            st.plotly_chart(fig, width="stretch")
    
    with tab7:
        st.caption("📌 Historical trends: trial volumes over time, success rate evolution, and sponsor activity patterns")
        st.markdown("#### 📅 Temporal Trends Analysis")
        
        df_dates = filtered[filtered["start_date"] != "Unknown"].copy()
        if len(df_dates) > 0:
            df_dates["year"] = pd.to_datetime(df_dates["start_date"], errors="coerce").dt.year
            df_dates = df_dates.dropna(subset=["year"])
            
            if len(df_dates) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Trial Starts by Year")
                    yearly_counts = df_dates.groupby("year").size().reset_index(name="count")
                    fig = px.area(yearly_counts, x="year", y="count",
                                labels={"year": "Year", "count": "Number of Trials"},
                                color_discrete_sequence=["#636EFA"])
                    fig.update_traces(line=dict(width=2))
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, width="stretch")
                
                with col2:
                    st.markdown("##### Success Rate Over Time")
                    df_dates_known = df_dates[df_dates["outcome_known"] == True]
                    if len(df_dates_known) > 0:
                        yearly_success = df_dates_known.groupby("year")["actual_success"].agg(["mean", "count"]).reset_index()
                        yearly_success = yearly_success[yearly_success["count"] >= 3]
                        
                        fig = px.line(yearly_success, x="year", y="mean", markers=True,
                                    labels={"year": "Year", "mean": "Success Rate"})
                        fig.update_yaxes(tickformat=".0%")
                        fig.update_traces(line=dict(width=3, color="#00CC96"))
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, width="stretch")
                
                st.markdown("##### Sponsor Activity Over Time")
                sponsor_yearly = df_dates.groupby(["year", "org_class"]).size().reset_index(name="count")
                top_sponsors = df_dates["org_class"].value_counts().head(5).index
                sponsor_yearly_top = sponsor_yearly[sponsor_yearly["org_class"].isin(top_sponsors)]
                
                fig = px.line(sponsor_yearly_top, x="year", y="count", color="org_class",
                            labels={"year": "Year", "count": "Number of Trials", "org_class": "Sponsor Type"},
                            markers=True)
                fig.update_traces(line=dict(width=2))
                fig.update_layout(height=450)
                st.plotly_chart(fig, width="stretch")
    
    with tab8:
        st.markdown("#### 📋 Trial Status Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Phase II Status Distribution")
            phase2_counts = filtered["phase2_status"].value_counts().head(10)
            fig = px.bar(x=phase2_counts.values, y=phase2_counts.index, orientation="h",
                       color=phase2_counts.values, color_continuous_scale="Blues",
                       labels={"x": "Number of Trials", "y": "Phase II Status"})
            fig.update_traces(text=phase2_counts.values, textposition="outside")
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            st.markdown("##### Phase III Status Distribution")
            phase3_counts = filtered["phase3_status"].value_counts().head(10)
            fig = px.bar(x=phase3_counts.values, y=phase3_counts.index, orientation="h",
                       color=phase3_counts.values, color_continuous_scale="Reds",
                       labels={"x": "Number of Trials", "y": "Phase III Status"})
            fig.update_traces(text=phase3_counts.values, textposition="outside")
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, width="stretch")
        
        st.markdown("##### Success Rate by Phase II Status")
        if len(known) > 0:
            phase2_success = known.groupby("phase2_status")["actual_success"].agg(["mean", "count"]).reset_index()
            phase2_success = phase2_success[phase2_success["count"] >= 5].sort_values("mean", ascending=False)
            
            fig = px.bar(phase2_success, x="phase2_status", y="mean",
                       text=phase2_success["mean"].apply(lambda x: f"{x:.0%}"),
                       color="mean", color_continuous_scale="RdYlGn",
                       labels={"phase2_status": "Phase II Status", "mean": "Success Rate"})
            fig.update_yaxes(tickformat=".0%")
            fig.update_traces(textposition="outside")
            fig.update_layout(height=400)
            st.plotly_chart(fig, width="stretch")
    
    with tab9:
        st.markdown("#### 🌐 Organization Insights")
        
        st.markdown("##### Top 20 Organizations by Trial Volume")
        org_counts = filtered["organization_name"].value_counts().head(20)
        fig = px.bar(y=org_counts.index, x=org_counts.values, orientation="h",
                   color=org_counts.values, color_continuous_scale="Teal",
                   labels={"x": "Number of Trials", "y": "Organization"})
        fig.update_traces(text=org_counts.values, textposition="outside")
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, width="stretch")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Success Rate by Top Organizations")
            if len(known) > 0:
                top_orgs = known["organization_name"].value_counts().head(15).index
                org_success = known[known["organization_name"].isin(top_orgs)].groupby("organization_name")["actual_success"].agg(["mean", "count"]).reset_index()
                org_success = org_success[org_success["count"] >= 3].sort_values("mean", ascending=False).head(10)
                
                fig = px.bar(org_success, x="organization_name", y="mean",
                           text=org_success["mean"].apply(lambda x: f"{x:.0%}"),
                           color="mean", color_continuous_scale="Viridis",
                           labels={"organization_name": "Organization", "mean": "Success Rate"})
                fig.update_yaxes(tickformat=".0%")
                fig.update_xaxes(tickangle=-45)
                fig.update_traces(textposition="outside")
                fig.update_layout(height=450)
                st.plotly_chart(fig, width="stretch")
        
        with col2:
            st.markdown("##### Organization Type Performance")
            if len(known) > 0:
                org_type_perf = known.groupby("org_class").agg({
                    "actual_success": ["mean", "count", "std"]
                }).reset_index()
                org_type_perf.columns = ["org_class", "mean", "count", "std"]
                org_type_perf["std"] = org_type_perf["std"].fillna(0)
                
                fig = px.scatter(org_type_perf, x="mean", y="count", size="count",
                               text="org_class", color="mean",
                               color_continuous_scale="RdYlGn",
                               labels={"mean": "Success Rate", "count": "Number of Trials"})
                fig.update_xaxes(tickformat=".0%")
                fig.update_traces(textposition="top center")
                fig.update_layout(height=450)
                st.plotly_chart(fig, width="stretch")
    
    with tab10:
        st.caption("📌 Advanced pattern recognition: cross-variable correlations and hidden relationships in the data")
        st.markdown("#### 🔗 Correlation & Pattern Analysis")
        
        if len(known) > 0:
            st.markdown("##### Success Rate Heatmap: Purpose × Sponsor")
            
            # Create pivot table
            heatmap_data = known.groupby(["primary_purpose", "org_class"])["actual_success"].agg(["mean", "count"]).reset_index()
            heatmap_data = heatmap_data[heatmap_data["count"] >= 3]
            
            # Pivot for heatmap
            heatmap_pivot = heatmap_data.pivot(index="primary_purpose", columns="org_class", values="mean")
            
            fig = px.imshow(heatmap_pivot, 
                          labels=dict(x="Sponsor Type", y="Primary Purpose", color="Success Rate"),
                          color_continuous_scale="RdYlGn",
                          aspect="auto")
            fig.update_layout(height=500)
            st.plotly_chart(fig, width="stretch")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Intervention Length Impact")
                known_copy = known.copy()
                known_copy["intervention_length"] = known_copy["interventions"].str.len()
                known_copy["length_category"] = pd.cut(known_copy["intervention_length"], 
                                                       bins=[0, 20, 40, 60, 1000],
                                                       labels=["Short", "Medium", "Long", "Very Long"])
                
                length_stats = known_copy.groupby("length_category", observed=False)["actual_success"].agg(["mean", "count"]).reset_index()
                
                fig = px.bar(length_stats, x="length_category", y="mean",
                           text=length_stats["mean"].apply(lambda x: f"{x:.0%}"),
                           color="mean", color_continuous_scale="Cividis",
                           labels={"length_category": "Intervention Name Length", "mean": "Success Rate"})
                fig.update_yaxes(tickformat=".0%")
                fig.update_traces(textposition="outside")
                fig.update_layout(height=400)
                st.plotly_chart(fig, width="stretch")
            
            with col2:
                st.markdown("##### Purpose Diversity by Sponsor")
                purpose_diversity = filtered.groupby("org_class")["primary_purpose"].nunique().reset_index()
                purpose_diversity.columns = ["org_class", "num_purposes"]
                purpose_diversity = purpose_diversity.sort_values("num_purposes", ascending=False)
                
                fig = px.bar(purpose_diversity, x="org_class", y="num_purposes",
                           text="num_purposes",
                           color="num_purposes", color_continuous_scale="Sunset",
                           labels={"org_class": "Sponsor Type", "num_purposes": "Number of Different Purposes"})
                fig.update_traces(textposition="outside")
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, width="stretch")
    
    # Table
    st.markdown("---")
    st.markdown(f"### 📋 Detailed Trial Information")
    
    # Limit display to prevent freezing
    max_display = 500
    if len(filtered) > max_display:
        st.warning(f"⚠️ Showing first {max_display} of {len(filtered):,} trials. Use filters to narrow results or download CSV for full data.")
        display_df = filtered.head(max_display)
    else:
        st.caption(f"📌 Displaying all {len(filtered):,} trials matching your filters. Click column headers to sort.")
        display_df = filtered
    
    display = display_df[[
        "trial_index", 
        "interventions", 
        "conditions", 
        "brief_title", 
        "org_class", 
        "primary_purpose", 
        "start_date",
        "outcome_label"
    ]].copy()
    
    display.columns = [
        "Trial ID", 
        "Intervention", 
        "Cancer Type", 
        "Brief Title", 
        "Sponsor Type", 
        "Purpose",
        "Start Date",
        "Phase III Outcome"
    ]
    
    st.dataframe(
        display, 
        height=600, 
        hide_index=True,
        column_config={
            "Trial ID": st.column_config.NumberColumn("Trial ID", help="Internal trial identifier"),
            "Intervention": st.column_config.TextColumn("Intervention", help="Drug or treatment intervention"),
            "Cancer Type": st.column_config.TextColumn("Cancer Type", help="Target cancer condition"),
            "Brief Title": st.column_config.TextColumn("Brief Title", help="Short trial description"),
            "Sponsor Type": st.column_config.TextColumn("Sponsor Type", help="Organization type sponsoring the trial"),
            "Purpose": st.column_config.TextColumn("Purpose", help="Primary purpose of the trial"),
            "Start Date": st.column_config.TextColumn("Start Date", help="Trial start date (YYYY-MM format)"),
            "Phase III Outcome": st.column_config.TextColumn("Phase III Outcome", help="Actual Phase III success/failure status")
        }
    )
    
    # Export
    st.markdown("---")
    st.markdown("### 💾 Export Filtered Data")
    st.caption("📌 Download the current filtered dataset for external analysis")
    
    if len(filtered) > 0:
        csv = filtered.to_csv(index=False)
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            st.download_button(
                "📥 Download as CSV",
                csv,
                "filtered_trials.csv",
                "text/csv",
                use_container_width=True
            )
    else:
        st.warning("⚠️ No trials to export. Adjust your filters to see results.")


@st.cache_data
def load_data():
    return load_historical_trials()
