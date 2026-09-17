"""
Interactive Streamlit Web Application for Multi-Brand AI Support Agent.
Powered by Agentic RAG, TWCS multi-brand vector database, and Gemini LLM.
"""

import os
import sys
import json
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.agent import CompanySupportAgent

st.set_page_config(
    page_title="Multi-Brand AI Support Agent (TWCS RAG)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Rich Aesthetics CSS
st.markdown("""
<style>
    .main-header {
        font-family: -Company-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .brand-badge {
        background: #1da1f2;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .handoff-alert {
        background-color: #fff2f2;
        border-left: 5px solid #ff3b30;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .autohandle-alert {
        background-color: #f0fdf4;
        border-left: 5px solid #22c55e;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .evidence-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_agent():
    return CompanySupportAgent()

agent = load_agent()

# Sidebar Brand Selection & Settings
st.sidebar.image("https://img.icons8.com/color/96/bot.png", width=70)
st.sidebar.title("Agent Controls")

available_brands = agent.get_available_brands()
selected_brand = st.sidebar.selectbox("🎯 Select Target Brand:", available_brands, index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset & Engine Status")
st.sidebar.markdown("- **Primary Corpus:** TWCS (71k+ Index)")
st.sidebar.markdown("- **Secondary Intent:** Banking77 (77 Intents)")
st.sidebar.markdown("- **Vector Store:** ChromaDB (Dense Local Embeddings)")
st.sidebar.markdown("- **Generative LLM:** Gemini 3.6 Flash")

# Header
st.markdown('<div class="main-header">🤖 Multi-Brand Customer Support AI Agent</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-header">Agentic RAG Engine trained on Twitter Customer Support (TWCS) • Active Brand Filter: <b>@{selected_brand}</b></div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs([
    "💬 Live Multi-Brand Triage",
    "📚 Knowledge & Intent Explorer",
    "🛡️ Human Handoff Rules"
])

# TAB 1: Live Interactive Triage
with tab1:
    st.markdown("### Test Customer Query Resolution")
    col1, col2 = st.columns([1.1, 1])

    with col1:
        # Dynamic Presets based on selected brand
        preset_dict = {
            "AmazonHelp": [
                "Custom Query",
                "My package was marked as delivered today but I never received it!",
                "Order cancellation request for item shipped by mistake",
                "Received a damaged package with broken items inside"
            ],
            "CompanySupport": [
                "Custom Query",
                "My device battery drains in 2 hours since updating to latest OS",
                "Screen cracked and battery feels hot and swollen",
                "Locked out of my Account ID account and 2FA is not sending codes"
            ],
            "Uber_Support": [
                "Custom Query",
                "Driver took a much longer route than expected and charged extra",
                "Left my wallet in the back seat of the Uber car last night"
            ],
            "SpotifyCares": [
                "Custom Query",
                "My Premium subscription charged me twice this month",
                "Music keeps pausing automatically every time screen turns off"
            ],
            "BofA_Help": [
                "Custom Query",
                "I lost my debit card and need to freeze my account immediately!",
                "Unauthorized transaction appearing on my bank statement"
            ]
        }

        default_presets = [
            "Custom Query",
            "My order was supposed to arrive yesterday but tracking hasn't updated!",
            "I lost my card and suspect someone is trying to make unauthorized charges!",
            "Battery gets hot and screen popped out, looks swollen!",
            "My subscription renewed automatically, can I get a refund?"
        ]

        active_presets = preset_dict.get(selected_brand, default_presets)
        selected_preset = st.selectbox("Choose a sample query or type your own:", active_presets)

        default_text = ""
        if selected_preset != "Custom Query":
            default_text = selected_preset

        user_input = st.text_area(
            f"Incoming Customer Tweet (@{selected_brand}):",
            value=default_text,
            placeholder=f"Type a customer message directed at @{selected_brand}...",
            height=120
        )

        run_btn = st.button("🚀 Process Query with Agentic RAG", type="primary", use_container_width=True)

    with col2:
        if run_btn or user_input:
            text_to_eval = user_input if user_input else default_text
            if text_to_eval:
                with st.spinner("Analyzing intent, evaluating safety handoff rules, and querying ChromaDB..."):
                    pred = agent.process_query(text_to_eval, selected_brand=selected_brand)

                st.markdown("#### ⚡ Real-Time Agent Analysis")
                
                # Escalation / Handoff Status
                if pred.escalation_action == "escalate":
                    st.markdown(
                        f"""<div class="handoff-alert">
                            <b>🚨 HUMAN HANDOFF TRIGGERED ({pred.risk_level.upper()} RISK)</b><br>
                            <i>Reason:</i> <code>{pred.escalation_reason}</code><br>
                            <small>{pred.escalation_explanation}</small>
                        </div>""", 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""<div class="autohandle-alert">
                            <b>✅ AUTONOMOUS RAG HANDLE ({pred.risk_level.upper()} RISK)</b><br>
                            <i>Status:</i> <code>{getattr(pred, 'grounding_status', 'synthesized')}</code>
                        </div>""", 
                        unsafe_allow_html=True
                    )

                st.markdown(f"**Target Brand Filter:** `@ {selected_brand}`")
                st.markdown(f"**Latency:** `{pred.latency_ms} ms`")

                st.markdown("#### 🤖 Synthesized Support Response")
                st.info(pred.generated_reply)

    # Evidence Drawer
    if run_btn or user_input:
        text_to_eval = user_input if user_input else default_text
        if text_to_eval and pred.retrieved_evidence:
            with st.expander("🔍 View Retrieved TWCS Historical Conversations (ChromaDB Evidence)", expanded=True):
                for i, ev in enumerate(pred.retrieved_evidence, 1):
                    st.markdown(
                        f"""<div class="evidence-card">
                            <b>Exemplar #{i}</b> | Brand: <code>@{ev.get('brand', 'Unknown')}</code> | Similarity Distance: <code>{ev.get('similarity_score', 0.0):.4f}</code><br>
                            <b>Customer Question:</b> {ev.get('customer_text', '')}<br>
                            <b>Historical Brand Reply:</b> <i>{ev.get('agent_text', '')}</i>
                        </div>""",
                        unsafe_allow_html=True
                    )

# TAB 2: Knowledge & Intent Explorer
with tab2:
    st.markdown("### 📚 Integrated Knowledge Corpus & Intent Taxonomy")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Primary TWCS Multi-Brand Index")
        st.write("Over **71,000+** indexed customer support threads across 108 top brands on Twitter.")
        st.markdown("- **AmazonHelp:** 20,000+ pairs")
        st.markdown("- **CompanySupport:** 7,200+ pairs")
        st.markdown("- **Uber_Support:** 4,600+ pairs")
        st.markdown("- **SpotifyCares:** 3,150+ pairs")
        st.markdown("- **Delta & Airlines:** 8,000+ pairs")
    with colB:
        st.markdown("#### Secondary Intent Taxonomy (Banking77)")
        st.write("Integrated **13,000+** labeled queries across **77 financial intents** for precise categorization.")

# TAB 3: Human Handoff Rules
with tab3:
    st.markdown("### 🛡️ Human Handoff Safety Policy & Triggers")
    st.markdown(
        "To protect customer accounts and ensure physical safety, the agent automatically "
        "bypasses AI generation and transfers the user to a human specialist when any of the following safety rules trigger:"
    )
    st.markdown("---")
    
    rules_path = os.path.join(BASE_DIR, "src", "rules", "handoff_rules.json")
    if os.path.exists(rules_path):
        with open(rules_path) as f:
            rules_data = json.load(f)
            
        triggers = rules_data.get("handoff_triggers", [])
        for rule in triggers:
            cat = rule.get("category", "General").replace("_", " ").title()
            risk = rule.get("risk_level", "HIGH").upper()
            explanation = rule.get("explanation", "")
            keywords = ", ".join([f"`{kw}`" for kw in rule.get("keywords", [])])
            
            icon = "🚨" if risk == "CRITICAL" else "⚠️"
            
            st.markdown(f"#### {icon} {cat} (`{risk} RISK`)")
            st.markdown(f"• **Policy Protocol:** {explanation}")
            st.markdown(f"• **Trigger Keywords:** {keywords}")
            st.markdown("&nbsp;")

