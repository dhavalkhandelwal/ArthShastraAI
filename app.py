import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
import traceback
import finance_toolkit as ftk

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    st.warning("LangChain libraries not available. Chat functionality will be disabled.")

st.set_page_config(page_title="ArthShastraAI", layout="wide")

GEMINI_API_KEY = "YOUR API KEY"

os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY

DATA_DIR = "data"
VECTOR_STORE_PATH = os.path.join(DATA_DIR, "faiss_index")
CHAT_HISTORY_PATH = os.path.join(DATA_DIR, "chat_history.json")
os.makedirs(DATA_DIR, exist_ok=True)

@st.cache_resource
def load_ai_resources():
    if not LANGCHAIN_AVAILABLE or not GEMINI_API_KEY:
        return None, None
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.6)
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return llm, embedding_model
    except Exception as e:
        st.error(f"Error loading AI resources: {e}")
        return None, None

@st.cache_resource
def load_vector_store(_embedding_model):
    if not _embedding_model:
        return None
    required_file = os.path.join(VECTOR_STORE_PATH, "index.faiss")
    try:
        if os.path.exists(required_file):
            return FAISS.load_local(VECTOR_STORE_PATH, _embedding_model, allow_dangerous_deserialization=True)
        else:
            st.info("Knowledge base not found. Creating a new one...")
            placeholder_texts = ["Welcome to ArthShastraAI! You can add text documents to the knowledge base in the 'Upload Knowledge' section."]
            new_index = FAISS.from_texts(placeholder_texts, _embedding_model)
            new_index.save_local(VECTOR_STORE_PATH)
            return new_index
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None

def load_chat_history():
    if os.path.exists(CHAT_HISTORY_PATH):
        try:
            with open(CHAT_HISTORY_PATH, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_chat_history(history):
    try:
        with open(CHAT_HISTORY_PATH, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        st.error(f"Error saving chat history: {e}")

if LANGCHAIN_AVAILABLE and GEMINI_API_KEY:
    llm, embedding_model = load_ai_resources()
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = load_vector_store(embedding_model)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = load_chat_history()
else:
    llm, embedding_model = None, None

def detect_data_frequency(df_index):
    periods_per_year = ftk.detect_frequency(pd.Series(index=df_index))
    freq_map = {252: "Daily", 52: "Weekly", 12: "Monthly", 4: "Quarterly", 1: "Annual"}
    return freq_map.get(periods_per_year, "Unknown"), periods_per_year

def general_csv_preprocessor(df):
    st.write("#### 1. Data Preview")
    st.dataframe(df.head(10))
    st.info("Please ensure your CSV has clear headers for each column.")
    
    st.write("#### 2. Configure Your Data")
    col1, col2 = st.columns(2)
    
    date_col = col1.selectbox("Select the column containing dates", [''] + list(df.columns))
    if not date_col:
        st.warning("Please select a date column.")
        return None, None
        
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception as e:
        st.error(f"Could not convert '{date_col}' to dates. Error: {e}")
        return None, None
        
    df_sorted = df.set_index(date_col).sort_index()
    
    freq_name, periods_per_year = detect_data_frequency(df_sorted.index)
    st.info(f"📅 Detected data frequency: **{freq_name}** ({periods_per_year} periods per year)")

    potential_asset_cols = [col for col in df_sorted.columns if pd.api.types.is_numeric_dtype(df_sorted[col])]
    if not potential_asset_cols:
        st.error("No numeric columns found for analysis.")
        return None, None
        
    asset_cols = col2.multiselect("Select asset/ticker columns (containing prices/values)", 
                                  potential_asset_cols, 
                                  default=potential_asset_cols[:min(10, len(potential_asset_cols))])
    if not asset_cols:
        st.warning("Please select at least one asset column.")
        return None, None
    
    st.write("#### 3. Data Processing Options")
    col3, col4 = st.columns(2)
    
    fill_method = col3.selectbox("Handle missing values:", 
                                 ["Forward fill", "Drop rows", "Linear interpolation"])
    
    return_type = col4.selectbox("Return calculation:", 
                                 ["Simple returns", "Log returns"])
    
    prices_df = df_sorted[asset_cols].copy()
    
    if fill_method == "Forward fill":
        prices_df = prices_df.ffill()
    elif fill_method == "Linear interpolation":
        prices_df = prices_df.interpolate()
    
    prices_df = prices_df.dropna()
    
    if prices_df.empty:
        st.error("No data remaining after processing missing values.")
        return None, None
    
    if return_type == "Simple returns":
        returns_df = prices_df.pct_change().dropna()
    else:
        returns_df = np.log(prices_df / prices_df.shift(1)).dropna()
    
    st.write("#### 4. Processed Returns Data")
    st.dataframe(returns_df.head(10))
    st.success(f"✅ Successfully processed {len(returns_df)} periods of data for {len(asset_cols)} assets")
    
    if returns_df.empty:
        st.error("Processed data is empty. Please check your data.")
        return None, None
        
    return returns_df, periods_per_year

def create_visualizations(returns_df, summary_stats, periods_per_year):
    st.subheader("📈 Cumulative Performance")
    cumulative_returns = (1 + returns_df).cumprod()
    
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    for col in cumulative_returns.columns:
        ax1.plot(cumulative_returns.index, cumulative_returns[col], label=col, linewidth=2)
    ax1.set_title("Cumulative Returns Over Time", fontsize=16)
    ax1.set_ylabel("Cumulative Return")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1)
    
    st.subheader("📊 Risk-Return Analysis")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    scatter = ax2.scatter(summary_stats['Annualized Vol'],
                          summary_stats['Annualized Return'],
                          s=100, alpha=0.7, c=summary_stats['Sharpe Ratio'], 
                          cmap='viridis')
    
    for i, asset in enumerate(summary_stats.index):
        ax2.annotate(asset, 
                     (summary_stats.iloc[i]['Annualized Vol'], 
                      summary_stats.iloc[i]['Annualized Return']),
                     xytext=(5, 5), textcoords='offset points')
    
    ax2.set_xlabel("Annualized Volatility")
    ax2.set_ylabel("Annualized Return")
    ax2.set_title("Risk-Return Profile (Color = Sharpe Ratio)")
    plt.colorbar(scatter, label='Sharpe Ratio')
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    
    st.subheader("🔗 Asset Correlation Matrix")
    correlation_matrix = returns_df.corr()
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, ax=ax3, fmt='.2f')
    ax3.set_title("Asset Correlation Matrix")
    st.pyplot(fig3)
    
    st.subheader("📉 Rolling Risk Analysis")
    window = min(max(len(returns_df) // 10, 10), 60)
    rolling_vol = returns_df.rolling(window=window).std() * np.sqrt(periods_per_year)
    
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    for col in rolling_vol.columns:
        ax4.plot(rolling_vol.index, rolling_vol[col], label=col, alpha=0.8)
    ax4.set_title(f"Rolling {window}-Period Volatility")
    ax4.set_ylabel("Annualized Volatility")
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig4)

def analyze_portfolio_pro(returns_df, riskfree_rate=0.03):
    try:
        summary = ftk.summary_stats(returns_df, riskfree_rate=riskfree_rate)
        er = ftk.annualize_rets(returns_df)
        cov = returns_df.cov() * ftk.detect_frequency(returns_df)
        
        insights = ftk.generate_portfolio_insights(returns_df, summary)
        
        return summary, er, cov, insights
        
    except Exception as e:
        st.error(f"Error in portfolio analysis: {e}")
        return None, None, None, []

def get_portfolio_advice_pro(summary_df, insights_list, returns_df):
    if not llm:
        return generate_basic_advice(summary_df, insights_list)
    
    prompt_template = """
    You are an expert financial advisor analyzing a portfolio. Based on the following data, provide detailed investment insights and recommendations.

    ## Portfolio Summary Statistics:
    {summary_markdown}

    ## Key Insights:
    {insights}

    ## Additional Context:
    - Number of assets: {num_assets}
    - Data period: {start_date} to {end_date}
    - Total observations: {num_observations}

    Please provide a comprehensive analysis covering:

    1. **Performance Assessment**: Evaluate the overall portfolio performance and individual asset performance
    2. **Risk Analysis**: Assess the risk profile and highlight any concerning risk metrics
    3. **Diversification**: Comment on portfolio diversification and correlation patterns
    4. **Recommendations**: Provide specific, actionable investment recommendations
    5. **Risk Management**: Suggest risk management strategies based on the data
    6. **Market Insights**: Provide context based on the observed patterns

    Format your response in clear sections with bullet points where appropriate. Be specific and reference the actual numbers from the data.
    """
    
    insights_text = "\n".join([f"• {insight}" for insight in insights_list])
    
    try:
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        
        with st.spinner("ArthShastraAI is generating detailed analysis..."):
            advice = chain.invoke({
                "summary_markdown": summary_df.to_markdown(),
                "insights": insights_text,
                "num_assets": len(summary_df),
                "start_date": returns_df.index[0].strftime("%Y-%m-%d"),
                "end_date": returns_df.index[-1].strftime("%Y-%m-%d"),
                "num_observations": len(returns_df)
            })
        
        return advice.content
        
    except Exception as e:
        st.error(f"Error generating AI advice: {e}")
        return generate_basic_advice(summary_df, insights_list)

def generate_basic_advice(summary_df, insights_list):
    advice = "## Portfolio Analysis Summary\n\n"
    
    best_return = summary_df['Annualized Return'].max()
    worst_return = summary_df['Annualized Return'].min()
    avg_return = summary_df['Annualized Return'].mean()
    
    advice += f"### Performance Overview\n"
    advice += f"- Average portfolio return: {avg_return:.2%}\n"
    advice += f"- Best performing asset return: {best_return:.2%}\n"
    advice += f"- Worst performing asset return: {worst_return:.2%}\n\n"
    
    avg_vol = summary_df['Annualized Vol'].mean()
    max_vol = summary_df['Annualized Vol'].max()
    avg_sharpe = summary_df['Sharpe Ratio'].mean()
    
    advice += f"### Risk Assessment\n"
    advice += f"- Average volatility: {avg_vol:.2%}\n"
    advice += f"- Maximum volatility: {max_vol:.2%}\n"
    advice += f"- Average Sharpe ratio: {avg_sharpe:.3f}\n\n"
    
    advice += "### Key Insights\n"
    for insight in insights_list:
        advice += f"- {insight}\n"
    
    advice += "\n### Recommendations\n"
    if avg_sharpe > 0.5:
        advice += "- The portfolio shows good risk-adjusted returns.\n"
    else:
        advice += "- Consider reviewing asset allocation to improve risk-adjusted returns.\n"
    
    if max_vol > 0.25:
        advice += "- Some assets show high volatility - consider risk management strategies.\n"
    
    return advice

def create_efficient_frontier_analysis(er, cov, riskfree_rate=0.03):
    try:
        w_msr = ftk.msr(riskfree_rate, er, cov)
        w_gmv = ftk.gmv(cov)
        w_ew = np.repeat(1/len(er), len(er))
        
        portfolios = {
            'Maximum Sharpe Ratio': {
                'weights': w_msr,
                'return': ftk.portfolio_return(w_msr, er),
                'volatility': ftk.portfolio_vol(w_msr, cov),
                'sharpe': (ftk.portfolio_return(w_msr, er) - riskfree_rate) / ftk.portfolio_vol(w_msr, cov)
            },
            'Global Minimum Volatility': {
                'weights': w_gmv,
                'return': ftk.portfolio_return(w_gmv, er),
                'volatility': ftk.portfolio_vol(w_gmv, cov),
                'sharpe': (ftk.portfolio_return(w_gmv, er) - riskfree_rate) / ftk.portfolio_vol(w_gmv, cov)
            },
            'Equal Weight': {
                'weights': w_ew,
                'return': ftk.portfolio_return(w_ew, er),
                'volatility': ftk.portfolio_vol(w_ew, cov),
                'sharpe': (ftk.portfolio_return(w_ew, er) - riskfree_rate) / ftk.portfolio_vol(w_ew, cov)
            }
        }
        
        return portfolios
        
    except Exception as e:
        st.error(f"Error in efficient frontier analysis: {e}")
        return None

st.title("ArthShastraAI: Portfolio & Financial Analysis")
st.markdown("### Your Comprehensive Financial Analysis Assistant")

st.sidebar.title("🔍 ArthShastraAI Navigation")
mode = st.sidebar.radio("Choose a mode:", [
    "📈 Portfolio Optimizer Pro",
    "💬 Ask Anything (Q&A)",
    "📚 Upload Knowledge (RAG)"
])

st.sidebar.markdown("---")
if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR API KEY":
    st.sidebar.warning("⚠️ Enter your Gemini API key to enable AI chat features")
else:
    st.sidebar.success("✅ AI features enabled")

st.sidebar.info("💡 This app uses local storage for chat history and knowledge base.")

if mode == "📈 Portfolio Optimizer Pro":
    st.header("📈 Portfolio Optimizer Pro")
    st.markdown("""
    Upload a CSV file with your portfolio's historical data. ArthShastraAI will:
    - Auto-detect data frequency (daily, weekly, monthly)
    - Calculate comprehensive risk metrics
    - Generate visualizations and insights
    - Provide AI-powered recommendations
    """)
    
    uploaded_file = st.file_uploader("Choose your portfolio CSV file", 
                                     type="csv",
                                     help="CSV should contain a date column and numeric price/value columns")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            
            result = general_csv_preprocessor(df)
            if result and result[0] is not None:
                returns_df, periods_per_year = result
                
                with st.spinner("🔄 Running comprehensive portfolio analysis..."):
                    analysis_result = analyze_portfolio_pro(returns_df)
                    
                if analysis_result and analysis_result[0] is not None:
                    summary_stats, expected_returns, cov_matrix, insights = analysis_result
                    
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "📊 Summary Stats", 
                        "📈 Visualizations", 
                        "🤖 AI Analysis", 
                        "⚡ Efficient Frontier",
                        "📋 Detailed Insights"
                    ])
                    
                    with tab1:
                        st.subheader("📊 Portfolio Summary Statistics")
                        
                        formatted_summary = summary_stats.copy()
                        percentage_cols = ['Annualized Return', 'Annualized Vol', 'Cornish-Fisher VaR (5%)', 
                                         'Historic CVaR (5%)', 'Max Drawdown']
                        
                        for col in percentage_cols:
                            if col in formatted_summary.columns:
                                formatted_summary[col] = formatted_summary[col].apply(lambda x: f"{x:.2%}")
                        
                        if 'Sharpe Ratio' in formatted_summary.columns:
                            formatted_summary['Sharpe Ratio'] = formatted_summary['Sharpe Ratio'].apply(lambda x: f"{x:.3f}")
                        if 'Skewness' in formatted_summary.columns:
                            formatted_summary['Skewness'] = formatted_summary['Skewness'].apply(lambda x: f"{x:.3f}")
                        if 'Kurtosis' in formatted_summary.columns:
                            formatted_summary['Kurtosis'] = formatted_summary['Kurtosis'].apply(lambda x: f"{x:.3f}")
                        
                        st.dataframe(formatted_summary, use_container_width=True)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            best_return = summary_stats['Annualized Return'].max()
                            st.metric("🏆 Best Return", f"{best_return:.2%}")
                        with col2:
                            avg_vol = summary_stats['Annualized Vol'].mean()
                            st.metric("📊 Avg Volatility", f"{avg_vol:.2%}")
                        with col3:
                            best_sharpe = summary_stats['Sharpe Ratio'].max()
                            st.metric("⚡ Best Sharpe", f"{best_sharpe:.3f}")
                        with col4:
                            worst_dd = summary_stats['Max Drawdown'].min()
                            st.metric("📉 Max Drawdown", f"{worst_dd:.2%}")
                    
                    with tab2:
                        create_visualizations(returns_df, summary_stats, periods_per_year)
                    
                    with tab3:
                        st.subheader("🤖 AI-Powered Portfolio Analysis")
                        ai_advice = get_portfolio_advice_pro(summary_stats, insights, returns_df)
                        st.markdown(ai_advice)
                    
                    with tab4:
                        st.subheader("⚡ Efficient Frontier Analysis")
                        fig, ax = plt.subplots(figsize=(12, 8))
                        ftk.plot_ef(n_points=25, 
                                    er=expected_returns, 
                                    cov=cov_matrix,
                                    ax=ax,
                                    show_cml=True, 
                                    show_ew=True, 
                                    show_gmv=True, 
                                    riskfree_rate=0.03)
                        ax.set_title("Efficient Frontier with Optimal Portfolios", fontsize=16)
                        ax.set_xlabel("Annualized Volatility (Risk)", fontsize=12)
                        ax.set_ylabel("Annualized Return", fontsize=12)
                        
                        ax.plot([], [], 'o-', color='blue', label='Efficient Frontier')
                        ax.plot([], [], 'o', color='green', markersize=10, label='Maximum Sharpe Ratio')
                        ax.plot([], [], 'o', color='midnightblue', markersize=10, label='Global Min Volatility')
                        ax.plot([], [], 'o', color='goldenrod', markersize=10, label='Equal Weight')
                        ax.plot([], [], 'o--', color='green', label='Capital Market Line')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
                        
                        portfolios = create_efficient_frontier_analysis(expected_returns, cov_matrix)
                        if portfolios:
                            st.subheader("📋 Optimal Portfolio Compositions")
                            
                            for name, portfolio in portfolios.items():
                                with st.expander(f"{name} Portfolio"):
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write("**Portfolio Metrics:**")
                                        st.write(f"Expected Return: {portfolio['return']:.2%}")
                                        st.write(f"Volatility: {portfolio['volatility']:.2%}")
                                        st.write(f"Sharpe Ratio: {portfolio['sharpe']:.3f}")
                                    
                                    with col2:
                                        st.write("**Asset Allocation:**")
                                        weights_df = pd.DataFrame({
                                            'Asset': expected_returns.index,
                                            'Weight': portfolio['weights']
                                        })
                                        weights_df['Weight %'] = weights_df['Weight'].apply(lambda x: f"{x:.1%}")
                                        st.dataframe(weights_df[['Asset', 'Weight %']], hide_index=True)
                    
                    with tab5:
                        st.subheader("📋 Detailed Portfolio Insights")
                        
                        st.write("#### 🔍 Key Findings:")
                        for insight in insights:
                            st.write(f"• {insight}")
                        
                        st.write("#### 📈 Additional Statistics:")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Return Statistics:**")
                            st.write(f"• Portfolio average return: {summary_stats['Annualized Return'].mean():.2%}")
                            st.write(f"• Return standard deviation: {summary_stats['Annualized Return'].std():.2%}")
                            st.write(f"• Best performer: {summary_stats['Annualized Return'].idxmax()}")
                            st.write(f"• Worst performer: {summary_stats['Annualized Return'].idxmin()}")
                        
                        with col2:
                            st.write("**Risk Statistics:**")
                            st.write(f"• Average volatility: {summary_stats['Annualized Vol'].mean():.2%}")
                            st.write(f"• Volatility range: {summary_stats['Annualized Vol'].min():.2%} - {summary_stats['Annualized Vol'].max():.2%}")
                            st.write(f"• Average Sharpe ratio: {summary_stats['Sharpe Ratio'].mean():.3f}")
                            st.write(f"• Correlation range: {returns_df.corr().min().min():.3f} - {returns_df.corr().max().max():.3f}")
                        
                        csv_data = summary_stats.to_csv()
                        st.download_button(
                            label="📥 Download Analysis Results",
                            data=csv_data,
                            file_name=f"portfolio_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
        except Exception as e:
            st.error(f"❌ An error occurred while processing your file: {str(e)}")
            st.error("Please check that your CSV file has the correct format with date and numeric columns.")
            if st.checkbox("Show detailed error information"):
                st.code(traceback.format_exc())

elif mode == "💬 Ask Anything (Q&A)":
    st.header("💬 Financial Q&A Assistant")
    
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR API KEY":
        st.warning("Please enter your Gemini API key in the sidebar to use the chat feature.")
    elif not LANGCHAIN_AVAILABLE:
        st.error("Required libraries are not installed. Please install langchain and related packages.")
    else:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        if prompt := st.chat_input("Ask about finance, portfolio analysis, or uploaded documents..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            try:
                with st.spinner("ArthShastraAI is thinking..."):
                    if st.session_state.vector_store:
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            chain_type="stuff",
                            retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                        )
                        response = qa_chain.invoke({"query": prompt})
                        answer = response['result']
                    else:
                        response = llm.invoke(prompt)
                        answer = response.content
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                    
                    save_chat_history(st.session_state.chat_history)
                    
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.error(error_msg)
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            save_chat_history([])
            st.rerun()

elif mode == "📚 Upload Knowledge (RAG)":
    st.header("📚 Upload Knowledge for RAG")
    
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR API KEY":
        st.warning("Please enter your Gemini API key in the sidebar to use this feature.")
    elif not LANGCHAIN_AVAILABLE:
        st.error("Required libraries are not installed. Please install langchain and related packages.")
    else:
        st.info("📖 Add text documents to ArthShastraAI's knowledge base. This information will be used to answer questions in the 'Ask Anything' mode.")
        
        uploaded_files = st.file_uploader(
            "Upload text files", 
            type=["txt", "md", "csv"], 
            accept_multiple_files=True,
            help="Supported formats: .txt, .md, .csv"
        )
        
        if uploaded_files:
            for file in uploaded_files:
                try:
                    if file.type == "text/csv":
                        df = pd.read_csv(file)
                        text_content = f"CSV File: {file.name}\nColumns: {', '.join(df.columns)}\nData Summary:\n{df.describe().to_string()}"
                    else:
                        stringio = StringIO(file.getvalue().decode("utf-8"))
                        text_content = stringio.read()
                    
                    with st.spinner(f"📝 Processing '{file.name}' and updating knowledge base..."):
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000, 
                            chunk_overlap=100
                        )
                        docs = text_splitter.split_text(text_content)
                        
                        if st.session_state.vector_store:
                            st.session_state.vector_store.add_texts(docs)
                            st.session_state.vector_store.save_local(VECTOR_STORE_PATH)
                        
                    st.success(f"✅ Successfully added '{file.name}' to the knowledge base!")
                    
                except Exception as e:
                    st.error(f"❌ An error occurred with file '{file.name}': {e}")
        
        if st.session_state.get('vector_store'):
            st.info("💾 Knowledge base is active and ready for queries!")
        else:
            st.warning("⚠️ No knowledge base found. Upload some documents to get started.")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        ArthShastraAI - Your Comprehensive Financial Analysis Assistant<br>
        Built by Dhawal Khandelwal • with Streamlit • Powered by Google Gemini • Enhanced with Advanced Analytics
    </div>
    """, 
    unsafe_allow_html=True
)
