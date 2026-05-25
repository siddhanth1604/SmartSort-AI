import streamlit as st
import joblib
import pandas as pd
import time
import plotly.express as px
from feature_extractor import FeatureExtractor


st.set_page_config(page_title="SmartSort.ai", page_icon="🧬", layout="wide")

ALGO_DETAILS = {
    "TimSort": {"desc": "Python's default sorter. A genius hybrid of Merge Sort and Insertion Sort. Unbeatable for real-world, partially sorted data.", "speed": "O(N log N)"},
    "QuickSort": {"desc": "The chaotic good of sorting. Uses a pivot to divide and conquer. Extremely fast for highly randomized data (high entropy).", "speed": "O(N log N)"},
    "MergeSort": {"desc": "The reliable workhorse. Splits data down to single elements and merges them perfectly. Great for massive, unpredictable datasets.", "speed": "O(N log N)"},
    "InsertionSort": {"desc": "The card player's sort. Builds the array one item at a time. The absolute mathematical champion for tiny arrays (under 50 items).", "speed": "O(N²)"},
    "CountingSort": {"desc": "The rule-breaker. Doesn't compare elements at all! Uses math to map numbers directly to their positions. Insanely fast for data with low variance.", "speed": "O(N + K)"}
}

@st.cache_resource
def load_brain():
    model = joblib.load("rf_sorting_model.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

model, encoder = load_brain()
extractor = FeatureExtractor()

st.title("🧬 SmartSort.ai")
st.markdown("*Advanced Neural Routing for Optimal Data Sorting*")
st.divider()

col1, col2 = st.columns([1, 2.5])

with col1:
    st.subheader("📥 Data Input")
    st.info("Paste your dataset below. Try different patterns (sorted, random, tiny, duplicates) to test the AI's logic!")
    
    user_input = st.text_area("Enter numbers separated by commas:", 
                              "4, 4, 1, 4, 2, 4, 4, 3, 4, 1", 
                              height=120)
    
    analyze_btn = st.button("⚡ Execute AI Analysis", use_container_width=True, type="primary")

with col2:
    if analyze_btn:
        try:
            arr = [int(x.strip()) for x in user_input.split(",")]
            
            # Subtle loading bar
            progress_text = "Scanning Mathematical DNA..."
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.005)
                my_bar.progress(percent_complete + 1, text=progress_text)
            my_bar.empty()
            
            features_dict = extractor.extract(arr) 
            feature_values = [list(features_dict.values())] 
            
            unique_ratio = len(set(arr)) / len(arr) 
            out_of_order = sum(1 for i in range(len(arr)-1) if arr[i] > arr[i+1])
            chaos_ratio = out_of_order / len(arr) if len(arr) > 1 else 0
            
            st.warning(f"🔍 System Check: Length: {len(arr)} | Unique: {unique_ratio:.2f} | Chaos Ratio: {chaos_ratio:.2f}")
            
            all_algos = ["TimSort", "QuickSort", "MergeSort", "InsertionSort", "CountingSort"]
            
            if len(arr) < 32:
                best_algo = "InsertionSort"
                # Force 100% confidence for InsertionSort
                probs = [1.0 if a == "InsertionSort" else 0.0 for a in all_algos]
                df_probs = pd.DataFrame({'Algorithm': all_algos, 'Confidence': probs})
                st.toast("⚡ FSM Override: Tiny array detected!", icon="⚡")
                
            elif unique_ratio < 0.35:
                best_algo = "CountingSort"
                # Force 100% confidence for CountingSort
                probs = [1.0 if a == "CountingSort" else 0.0 for a in all_algos]
                df_probs = pd.DataFrame({'Algorithm': all_algos, 'Confidence': probs})
                st.toast("🧬 FSM Override: Massive duplicates detected!", icon="🧬")
                
            elif 0.40 <= chaos_ratio <= 0.60:
                best_algo = "QuickSort"
                # Force 100% confidence for QuickSort
                probs = [1.0 if a == "QuickSort" else 0.0 for a in all_algos]
                df_probs = pd.DataFrame({'Algorithm': all_algos, 'Confidence': probs})
                st.toast("🌪️ FSM Override: High Chaos detected!", icon="🌪️")
                
            else:
             
                prediction = model.predict(feature_values)
                best_algo = encoder.inverse_transform(prediction)[0]
                model_probs = model.predict_proba(feature_values)[0]
                model_classes = encoder.classes_
                df_probs = pd.DataFrame({'Algorithm': model_classes, 'Confidence': model_probs})
            
           
            st.subheader("📊 Analysis Report")
            
            st.markdown(f"### 🏆 Optimal Strategy: **{best_algo}**")
            desc = ALGO_DETAILS.get(best_algo, {"desc": "A highly efficient algorithm.", "speed": "Unknown"})
            st.write(f"**Reasoning:** {desc['desc']}")
            st.write(f"**Time Complexity:** `{desc['speed']}`")
            
            with st.expander("View Sorted Array Output"):
                st.write(sorted(arr))
                
            st.divider()
            
            st.markdown("## 📈 Visual Analytics")
            
            chart_col1, chart_col2 = st.columns(2, gap="large")

            with chart_col1:
                fig_pie = px.pie(
                    df_probs, 
                    names='Algorithm', 
                    values='Confidence', 
                    title="Neural Confidence Breakdown",
                    hole=0.4
                )
                fig_pie.update_layout(
                    height=450, # Matched height for side-by-side layout
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    margin=dict(t=50, b=50, l=0, r=0)
                )
                fig_pie.update_traces(
                    textposition='inside', 
                    textinfo='percent+label', 
                    marker=dict(line=dict(color='#000000', width=1))
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with chart_col2:
                features_df = pd.DataFrame({
                    'Metric': [k.replace("_", " ").title() for k in features_dict.keys()],
                    'Value': list(features_dict.values())
                })
                fig_bar = px.bar(
                    features_df, 
                    x='Value', y='Metric', orientation='h',
                    title="Array DNA Signature",
                    color='Value', color_continuous_scale='Viridis',
                    text_auto='.2f' 
                )
                fig_bar.update_layout(
                    height=450, 
                    margin=dict(t=50, b=50, l=0, r=0),
                    yaxis={'categoryorder':'total ascending'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            st.divider()
            st.markdown("#### 🧮 Raw DNA Metrics")
            met_cols = st.columns(3)
            feature_items = list(features_dict.items())
            
            for i, (key, value) in enumerate(feature_items):
                display_val = round(value, 4) if isinstance(value, float) else value
                met_cols[i % 3].metric(label=key.replace("_", " ").title(), value=display_val)

        except ValueError:
            st.error("Oops! Please ensure you only enter numbers separated by commas.")
        except AttributeError:
             st.error("Wait! Make sure your extractor function name matches (e.g., `extract` vs `extract_features`). Check `feature_extractor.py`!")

             import streamlit as st
import pandas as pd
import numpy as np

st.title("My Sorting Algorithm Selector")
st.write("If you can read this, Streamlit is working perfectly!")

# Just a quick test button
if st.button("Click Me"):
    st.success("Libraries are installed and the app is alive!")