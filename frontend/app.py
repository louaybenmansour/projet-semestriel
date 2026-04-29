import streamlit as st
import requests
import plotly.graph_objects as go
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
DEFAULT_API_BASE = "http://127.0.0.1:8000"
API_BASE = os.getenv("API_URL", DEFAULT_API_BASE).rstrip("/")
API_URL = f"{API_BASE}/predict"
HEALTH_URL = f"{API_BASE}/health"

st.set_page_config(
    page_title="Cognitive Core",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- WORLD-CLASS CSS INJECTION ---
def inject_custom_css():
    st.markdown("""
        <style>
        /* Import premium font */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
        
        /* Global Reset & Background */
        .stApp {
            font-family: 'Outfit', sans-serif;
            background-color: #09090b;
            background-image: 
                radial-gradient(circle at 10% 10%, rgba(56, 189, 248, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 90% 90%, rgba(139, 92, 246, 0.06) 0%, transparent 40%);
            color: #fafafa;
        }
        
        /* Hide default Streamlit fluff */
        header {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Typography */
        .app-title {
            font-size: 3.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #38bdf8 0%, #8b5cf6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0px;
            padding-bottom: 0px;
            line-height: 1.1;
            letter-spacing: -1px;
        }
        .app-subtitle {
            font-size: 1.1rem;
            color: #a1a1aa;
            font-weight: 300;
            margin-top: 8px;
            letter-spacing: 0.5px;
        }
        
        /* The Glass Card Effect (Targeting st.container with border=True) */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(24, 24, 27, 0.5) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 20px !important;
            padding: 1.5rem 2rem !important;
            box-shadow: 0 10px 40px -10px rgba(0, 0, 0, 0.5) !important;
            backdrop-filter: blur(20px) !important;
            -webkit-backdrop-filter: blur(20px) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: rgba(139, 92, 246, 0.2) !important;
            box-shadow: 0 10px 40px -10px rgba(139, 92, 246, 0.1) !important;
        }

        /* Sliders Customization */
        .stSlider div[data-testid="stThumbValue"] {
            background: #38bdf8 !important;
            color: #09090b !important;
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            border-radius: 6px;
            padding: 2px 8px;
            font-size: 0.85rem;
        }
        .stSlider div[data-baseweb="slider"] div[data-testid="stTickBar"] {
            background: rgba(255, 255, 255, 0.1) !important;
        }
        
        /* Metric Box Customization */
        div[data-testid="metric-container"] {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 1.5rem;
            border-radius: 16px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        div[data-testid="metric-container"] label {
            color: #a1a1aa !important;
            font-weight: 500;
            font-size: 1rem;
        }
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
            color: #fafafa !important;
            font-size: 2.2rem;
            font-weight: 700;
        }
        div[data-testid="stMetricDelta"] svg {
            display: none; /* Hide default arrow to keep it clean */
        }

        /* Primary Action Button */
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #38bdf8 0%, #8b5cf6 100%);
            color: white;
            font-size: 1.1rem;
            font-weight: 600;
            border-radius: 12px;
            border: none;
            height: 54px;
            width: 100%;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 8px 20px -8px rgba(139, 92, 246, 0.6);
        }
        div.stButton > button:first-child:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 25px -8px rgba(139, 92, 246, 0.8);
            color: white;
        }
        div.stButton > button:first-child:active {
            transform: scale(0.98);
        }
        
        /* Custom Labels */
        .section-title {
            font-weight: 600;
            font-size: 1.4rem;
            color: #fafafa;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .insight-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 1.5rem;
            margin-top: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

# --- VISUALIZATIONS ---
def draw_arc_gauge(score, threshold):
    color = "#34d399" if score >= threshold else "#f87171"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        number = {'font': {'size': 56, 'color': '#fafafa', 'family': 'Outfit'}, 'suffix': "%"},
        title = {'text': "Predicted Score", 'font': {'size': 14, 'color': '#a1a1aa', 'family': 'Outfit'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 0, 'showticklabels': False},
            'bar': {'color': color, 'thickness': 0.12},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 0,
            'threshold': {
                'line': {'color': "#38bdf8", 'width': 3},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=250
    )
    return fig

def draw_radar_analysis(study, sleep, attendance, stress):
    categories = ['Focus', 'Recovery', 'Commitment', 'Composure']
    student_vals = [(study/24)*100, (sleep/16)*100, attendance, 100-stress]
    ideal_vals = [(8/24)*100, (8/16)*100, 95, 80]
    
    fig = go.Figure()
    
    # Target Profile
    fig.add_trace(go.Scatterpolar(
        r=ideal_vals, theta=categories,
        fill='toself', fillcolor='rgba(255, 255, 255, 0.03)',
        line=dict(color='rgba(255, 255, 255, 0.2)', width=1.5, dash='dot'),
        name='Target', hoverinfo='none'
    ))
    
    # Student Profile
    fig.add_trace(go.Scatterpolar(
        r=student_vals, theta=categories,
        fill='toself', fillcolor='rgba(139, 92, 246, 0.25)',
        line=dict(color='#8b5cf6', width=2.5),
        name='Student'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 100]),
            angularaxis=dict(
                tickfont=dict(size=12, color="#a1a1aa", family="Outfit"),
                gridcolor="rgba(255,255,255,0.05)",
                linecolor="rgba(0,0,0,0)"
            ),
            bgcolor="rgba(0,0,0,0)"
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=30, t=30, b=10),
        height=250
    )
    return fig

def check_backend_health():
    try:
        res = requests.get(HEALTH_URL, timeout=1)
        return res.status_code == 200
    except:
        return False

# --- MAIN APP UI ---
def main():
    inject_custom_css()
    
    # Header Area
    col_logo, col_status = st.columns([1, 1])
    
    with col_logo:
        st.markdown("<div class='app-title'>Cognitive Core.</div>", unsafe_allow_html=True)
        st.markdown("<div class='app-subtitle'>Hybrid AI academic analytics engine.</div>", unsafe_allow_html=True)
        
    with col_status:
        is_online = check_backend_health()
        status_color = "#34d399" if is_online else "#f87171"
        status_text = "ENGINE ONLINE" if is_online else "ENGINE OFFLINE"
        st.markdown(f"""
            <div style='display: flex; justify-content: flex-end; align-items: center; gap: 8px; height: 100%; margin-top: 15px;'>
                <span style='color: #a1a1aa; font-size: 0.8rem; font-weight: 600; letter-spacing: 1px;'>{status_text}</span>
                <div style='width: 8px; height: 8px; border-radius: 50%; background-color: {status_color}; box-shadow: 0 0 10px {status_color};'></div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    
    # Navigation Tabs
    tab_predict, tab_arch, tab_pipeline = st.tabs(["🔮 Synthesis", "🏗️ Architecture", "🧬 Pipeline"])
    
    with tab_predict:
        col_input, col_spacer, col_dashboard = st.columns([1, 0.05, 1.5])
        
        with col_input:
            with st.container(border=True):
                st.markdown("<div class='section-title'><span>🧬</span> Biometrics Input</div>", unsafe_allow_html=True)
                
                study = st.slider("Daily Study Focus (Hrs)", min_value=0.0, max_value=24.0, value=6.0, step=0.5)
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                
                sleep = st.slider("REM Sleep Cycle (Hrs)", min_value=0.0, max_value=16.0, value=7.5, step=0.5)
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                
                attendance = st.slider("Institutional Presence (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                
                stress = st.slider("Cognitive Load / Stress", min_value=0.0, max_value=100.0, value=40.0, step=1.0)
                
                st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
                predict_btn = st.button("Synthesize Prediction")

        with col_dashboard:
            with st.container(border=True):
                st.markdown("<div class='section-title'><span>📊</span> Synthesis Dashboard</div>", unsafe_allow_html=True)
                
                if predict_btn:
                    if not is_online:
                        st.error("Backend API is currently unreachable. Please ensure it is running.")
                    else:
                        with st.spinner("Executing prediction matrices..."):
                            time.sleep(0.5)
                            payload = {"StudyHours": study, "SleepHours": sleep, "Attendance": attendance, "StressLevel": stress}
                            
                            try:
                                res = requests.post(API_URL, json=payload, timeout=5)
                                res.raise_for_status()
                                data = res.json()
                                
                                score = data.get("prediction", 0)
                                threshold = data.get("threshold", 60)
                                label = data.get("label", "Unknown")
                                m_type = data.get("model_type", "ML")
                                m_name = data.get("model_name", "Unknown")
                                
                                # Top Row: Charts
                                c1, c2 = st.columns(2)
                                with c1: st.plotly_chart(draw_arc_gauge(score, threshold), use_container_width=True, config={'displayModeBar': False})
                                with c2: st.plotly_chart(draw_radar_analysis(study, sleep, attendance, stress), use_container_width=True, config={'displayModeBar': False})
                                
                                st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
                                
                                # Bottom Row: Metrics & Insights
                                m1, m2, m3 = st.columns(3)
                                m1.metric(label="Predicted Trajectory", value=label)
                                
                                delta = score - threshold
                                m2.metric(label="Delta from Threshold", value=f"{abs(delta):.1f} pts", delta="Above" if delta >= 0 else "Below", delta_color="normal" if delta >= 0 else "inverse")
                                
                                m3.metric(label="Active Core", value=m_name)
                                
                                # Insight Footer
                                if label == "Pass":
                                    st.markdown(f"""
                                        <div class='insight-card' style='border-left: 3px solid #34d399;'>
                                            <h5 style='color: #34d399; margin: 0 0 5px 0;'>Optimal Profile Confirmed (via {m_type})</h5>
                                            <p style='color: #a1a1aa; margin: 0; font-size: 0.95rem;'>Subject exhibits behavioral patterns heavily correlated with academic success. Continue standard protocols.</p>
                                        </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                        <div class='insight-card' style='border-left: 3px solid #f87171;'>
                                            <h5 style='color: #f87171; margin: 0 0 5px 0;'>Critical Intervention Advised (via {m_type})</h5>
                                            <p style='color: #a1a1aa; margin: 0; font-size: 0.95rem;'>Subject deviating from success parameters. Recommend immediate adjustment to institutional presence.</p>
                                        </div>
                                    """, unsafe_allow_html=True)
                                    
                            except Exception as e:
                                st.error(f"Synthesis failed: {str(e)}")
                else:
                    # Awaiting State
                    st.markdown("""
                        <div style='height: 480px; display: flex; align-items: center; justify-content: center; flex-direction: column;'>
                            <div style='font-size: 4.5rem; opacity: 0.2; filter: drop-shadow(0 0 15px #8b5cf6);'>✨</div>
                            <h3 style='color: #a1a1aa; font-weight: 300; margin-top: 25px;'>Awaiting Input</h3>
                            <p style='color: #71717a; font-size: 0.95rem;'>Adjust the parameters and initialize synthesis.</p>
                        </div>
                    """, unsafe_allow_html=True)

    with tab_arch:
        with st.container(border=True):
            st.markdown("<div class='section-title'><span>🏗️</span> Model Selection Strategy</div>", unsafe_allow_html=True)
            
            c1, c2 = st.columns([1, 1.5])
            with c1:
                st.write("Our engine dynamically selects the most accurate architecture by comparing traditional Machine Learning with Deep Learning Multi-Layer Perceptrons.")
                
                metrics_path = "models/model_metrics.csv"
                if os.path.exists(metrics_path):
                    import pandas as pd
                    df_metrics = pd.read_csv(metrics_path)
                    st.dataframe(df_metrics, hide_index=True, use_container_width=True)
                else:
                    st.info("Metrics registry not found. Run training to generate performance benchmarks.")
            
            with c2:
                curves_path = "models/dl_training_curves.png"
                if os.path.exists(curves_path):
                    st.image(curves_path, caption="Deep Learning Convergence Curves (MSE)", use_container_width=True)
                else:
                    st.info("No Deep Learning training curves available in artifacts.")

    with tab_pipeline:
        with st.container(border=True):
            st.markdown("<div class='section-title'><span>🧬</span> End-to-End Processing Pipeline</div>", unsafe_allow_html=True)
            
            st.markdown("""
            ### 1. Data Sanitization
            - Missing value imputation using median strategy for numeric features.
            - Outlier detection and removal for target variables.
            - Robust categorical encoding.

            ### 2. Feature Engineering
            - **Academic Stress Index**: Derived from study hours, attendance, and sleep quality.
            - **Digital Access Score**: Composite metric representing institutional support and internet availability.

            ### 3. Preprocessing Graph
            - **Standardization**: Z-score scaling for deep learning compatibility.
            - **One-Hot Encoding**: Nominal variable expansion.
            - **Ordinal Mapping**: Preserving hierarchy in qualitative data (e.g., Parental Involvement).

            ### 4. Hybrid Competition
            - Parallel training of **Linear Regressors**, **Ensemble Trees** (Random Forest, Gradient Boosting), and **Deep Neural Networks**.
            - Automatic serialization of the champion model for production deployment.
            """)

if __name__ == "__main__":
    main()
