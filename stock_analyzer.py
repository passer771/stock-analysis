"""
=========================================
   📊 智能股票分析系统 v2.0
   🛠 技术栈: Python + Streamlit + Plotly
=========================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import datetime
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# ===================== 页面配置 =====================
st.set_page_config(
    page_title="📊 智能股票分析系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== 样式美化 =====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
    }
    .stButton>button {
        width: 100%;
        background: #1E88E5;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ===================== 辅助函数 =====================

@st.cache_data(ttl=3600)
def load_stock_data(ticker, period="1y"):
    """
    从 Yahoo Finance 加载股票数据
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            return None, f"❌ 未找到股票代码: {ticker}"
        return df, None
    except Exception as e:
        return None, f"❌ 加载失败: {str(e)}"


def calculate_technical_indicators(df):
    """
    计算常用技术指标
    """
    df = df.copy()
    
    # 移动平均线
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    # 布林带
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
    df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
    
    # RSI (相对强弱指标)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # 成交量移动平均
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    
    # 波动率
    df['Volatility'] = df['Close'].pct_change().rolling(window=20).std() * 100
    
    return df


def plot_candlestick_chart(df, ticker, indicators=None):
    """
    绘制K线图和技术指标
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f"{ticker} - K线图", "成交量", "RSI / MACD")
    )
    
    # K线图
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="K线",
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 移动平均线
    if indicators and 'MA' in indicators:
        colors = {'MA5': '#FF6B6B', 'MA10': '#4ECDC4', 'MA20': '#45B7D1', 'MA60': '#96CEB4'}
        for ma in ['MA5', 'MA10', 'MA20', 'MA60']:
            if ma in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df[ma],
                        mode='lines',
                        name=ma,
                        line=dict(color=colors.get(ma, '#888'), width=1)
                    ),
                    row=1, col=1
                )
    
    # 布林带
    if indicators and 'BB' in indicators:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['BB_Upper'],
                mode='lines',
                name='布林上轨',
                line=dict(color='rgba(173, 216, 230, 0.5)', width=1),
                showlegend=True
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['BB_Lower'],
                mode='lines',
                name='布林下轨',
                line=dict(color='rgba(173, 216, 230, 0.5)', width=1),
                fill='tonexty',
                fillcolor='rgba(173, 216, 230, 0.1)',
                showlegend=True
            ),
            row=1, col=1
        )
    
    # 成交量柱状图
    colors_vol = ['red' if close >= open else 'green' 
                  for close, open in zip(df['Close'], df['Open'])]
    fig.add_trace(
        go.Bar(
            x=df.index, y=df['Volume'],
            name='成交量',
            marker_color=colors_vol,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # RSI
    if indicators and 'RSI' in indicators:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='#FF9800', width=2)
            ),
            row=3, col=1
        )
        # RSI 参考线
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # MACD
    if indicators and 'MACD' in indicators:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='#2196F3', width=2)
            ),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['MACD_Signal'],
                mode='lines',
                name='MACD Signal',
                line=dict(color='#FF5722', width=2)
            ),
            row=3, col=1
        )
        # MACD 柱状图
        colors_macd = ['red' if v >= 0 else 'green' for v in df['MACD_Hist']]
        fig.add_trace(
            go.Bar(
                x=df.index, y=df['MACD_Hist'],
                name='MACD Hist',
                marker_color=colors_macd,
                opacity=0.5
            ),
            row=3, col=1
        )
    
    # 更新布局
    fig.update_layout(
        template='plotly_dark',
        height=800,
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode='x unified',
        xaxis_rangeslider_visible=False
    )
    
    fig.update_xaxes(title_text="日期", row=3, col=1)
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    fig.update_yaxes(title_text="指标值", row=3, col=1)
    
    return fig


def generate_signal(df):
    """
    生成交易信号
    """
    signals = []
    
    # 最近的数据
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 1. 均线金叉/死叉
    if latest['MA5'] > latest['MA20'] and prev['MA5'] <= prev['MA20']:
        signals.append(("🔴 买入信号", "5日均线上穿20日均线 (金叉)", "positive"))
    elif latest['MA5'] < latest['MA20'] and prev['MA5'] >= prev['MA20']:
        signals.append(("🟢 卖出信号", "5日均线下穿20日均线 (死叉)", "negative"))
    
    # 2. RSI 超买/超卖
    if latest['RSI'] < 30:
        signals.append(("🔴 超卖反弹信号", f"RSI = {latest['RSI']:.1f} (低于30，超卖区)", "positive"))
    elif latest['RSI'] > 70:
        signals.append(("🟢 超买回调信号", f"RSI = {latest['RSI']:.1f} (高于70，超买区)", "negative"))
    
    # 3. MACD 金叉/死叉
    if latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
        signals.append(("🔴 买入信号", "MACD金叉 (DIF上穿DEA)", "positive"))
    elif latest['MACD'] < latest['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
        signals.append(("🟢 卖出信号", "MACD死叉 (DIF下穿DEA)", "negative"))
    
    # 4. 布林带触及
    if latest['Close'] <= latest['BB_Lower'] * 1.02:
        signals.append(("🔴 超跌信号", "股价触及布林带下轨", "positive"))
    elif latest['Close'] >= latest['BB_Upper'] * 0.98:
        signals.append(("🟢 超涨信号", "股价触及布林带上轨", "negative"))
    
    if not signals:
        signals.append(("⚪ 无明确信号", "当前无强烈交易信号", "neutral"))
    
    return signals


def get_stock_info(ticker):
    """
    获取股票基本面信息
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'name': info.get('longName', info.get('shortName', ticker)),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'dividend_yield': info.get('dividendYield', 0),
            'beta': info.get('beta', 'N/A'),
            '52w_high': info.get('fiftyTwoWeekHigh', 0),
            '52w_low': info.get('fiftyTwoWeekLow', 0),
            'avg_volume': info.get('averageVolume', 0),
            'description': info.get('longBusinessSummary', '暂无简介')
        }
    except:
        return None


# ===================== 主程序 =====================

def main():
    # 侧边栏
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/stock.png", width=80)
        st.markdown("## 📊 股票分析系统")
        st.markdown("---")
        
        # 股票代码输入
        ticker = st.text_input(
            "🔍 输入股票代码",
            value="AAPL",
            help="例如: AAPL (苹果), MSFT (微软), 000001.SZ (平安), 600519.SS (茅台)"
        ).strip().upper()
        
        # 时间周期选择
        period = st.selectbox(
            "📅 选择时间范围",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
            index=3,
            format_func=lambda x: {
                "1mo": "1个月", "3mo": "3个月", "6mo": "6个月",
                "1y": "1年", "2y": "2年", "5y": "5年", "max": "全部"
            }.get(x, x)
        )
        
        # 技术指标选择
        st.markdown("### 📐 技术指标")
        indicators = []
        if st.checkbox("移动平均线 (MA)", value=True):
            indicators.extend(['MA5', 'MA10', 'MA20', 'MA60'])
        if st.checkbox("布林带 (BB)", value=True):
            indicators.append('BB')
        if st.checkbox("RSI", value=True):
            indicators.append('RSI')
        if st.checkbox("MACD", value=True):
            indicators.append('MACD')
        
        # 启动按钮
        analyze_btn = st.button("🚀 开始分析", use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📌 快速示例")
        quick_stocks = {
            "🇺🇸 美股": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"],
            "🇨🇳 A股": ["600519.SS", "000001.SZ", "300750.SZ", "000858.SZ"]
        }
        for market, stocks in quick_stocks.items():
            st.markdown(f"**{market}**")
            cols = st.columns(len(stocks[:3]))
            for i, s in enumerate(stocks[:3]):
                with cols[i]:
                    if st.button(s, key=f"quick_{s}", use_container_width=True):
                        ticker = s
                        analyze_btn = True
    
    # 主界面
    st.markdown('<h1 class="main-header">📊 智能股票分析系统</h1>', unsafe_allow_html=True)
    
    if not analyze_btn:
        # 欢迎页面
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("📈 **实时行情**\n\n获取全球股票实时数据")
        with col2:
            st.success("📐 **技术分析**\n\nMA, MACD, RSI, 布林带等")
        with col3:
            st.warning("🔔 **智能信号**\n\n自动识别买卖信号")
        
        st.markdown("---")
        st.markdown("### 🏆 热门股票速览")
        
        hot_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK-B", "JPM", "V"]
        cols = st.columns(5)
        for i, stock in enumerate(hot_stocks):
            with cols[i % 5]:
                try:
                    data, err = load_stock_data(stock, "5d")
                    if data is not None:
                        price = data['Close'].iloc[-1]
                        change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
                        pct = (change / data['Close'].iloc[-2]) * 100
                        color = "green" if change >= 0 else "red"
                        st.markdown(f"""
                        <div style="text-align:center; padding:10px; background:#f0f2f6; border-radius:10px; margin:5px;">
                            <h4>{stock}</h4>
                            <h3>${price:.2f}</h3>
                            <p style="color:{color};">{change:+.2f} ({pct:+.2f}%)</p>
                        </div>
                        """, unsafe_allow_html=True)
                except:
                    pass
        
        return
    
    # ========== 核心分析逻辑 ==========
    
    # 显示加载状态
    with st.spinner(f"🔍 正在加载 {ticker} 数据..."):
        df, error = load_stock_data(ticker, period)
    
    if error:
        st.error(error)
        st.info("💡 提示：A股代码需要添加后缀，如：贵州茅台 → 600519.SS，平安银行 → 000001.SZ")
        return
    
    if df is None or df.empty:
        st.error("❌ 未获取到数据，请检查股票代码")
        return
    
    # 计算技术指标
    df = calculate_technical_indicators(df)
    
    # 获取公司信息
    info = get_stock_info(ticker)
    
    # ==================== 顶部信息卡片 ====================
    
    st.markdown(f"## 🏢 {info['name'] if info else ticker} ({ticker})")
    
    # 基本信息行
    col1, col2, col3, col4, col5 = st.columns(5)
    
    latest_price = df['Close'].iloc[-1]
    price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
    price_pct = (price_change / df['Close'].iloc[-2]) * 100
    
    with col1:
        st.metric(
            "当前价格",
            f"${latest_price:.2f}" if ticker.isalpha() else f"¥{latest_price:.2f}",
            f"{price_change:+.2f} ({price_pct:+.2f}%)",
            delta_color="normal"
        )
    
    with col2:
        st.metric("开盘价", f"${df['Open'].iloc[-1]:.2f}")
    
    with col3:
        st.metric("最高价", f"${df['High'].iloc[-1]:.2f}")
    
    with col4:
        st.metric("最低价", f"${df['Low'].iloc[-1]:.2f}")
    
    with col5:
        st.metric("成交量", f"{df['Volume'].iloc[-1]:,.0f}")
    
    # 基本面信息
    if info:
        with st.expander("📋 公司基本面信息", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("行业", info['sector'])
            col2.metric("市值", f"${info['market_cap']/1e9:.2f}B" if info['market_cap'] else "N/A")
            col3.metric("市盈率 (PE)", f"{info['pe_ratio']:.2f}" if info['pe_ratio'] != 'N/A' else "N/A")
            col4.metric("Beta值", f"{info['beta']:.2f}" if info['beta'] != 'N/A' else "N/A")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("52周最高", f"${info['52w_high']:.2f}")
            col2.metric("52周最低", f"${info['52w_low']:.2f}")
            col3.metric("股息率", f"{info['dividend_yield']*100:.2f}%" if info['dividend_yield'] else "N/A")
            col4.metric("平均成交量", f"{info['avg_volume']:,.0f}")
            
            if info['description'] != '暂无简介':
                st.markdown(f"**公司简介:** {info['description'][:500]}...")
    
    # ==================== K线图 ====================
    
    st.markdown("---")
    st.markdown("### 📈 股价走势分析")
    
    fig = plot_candlestick_chart(df, ticker, indicators)
    st.plotly_chart(fig, use_container_width=True)
    
    # ==================== 交易信号 ====================
    
    st.markdown("---")
    st.markdown("### 🔔 智能交易信号")
    
    signals = generate_signal(df)
    
    cols = st.columns(len(signals))
    for i, (title, desc, signal_type) in enumerate(signals):
        with cols[i]:
            if signal_type == "positive":
                st.success(f"**{title}**\n\n{desc}")
            elif signal_type == "negative":
                st.error(f"**{title}**\n\n{desc}")
            else:
                st.info(f"**{title}**\n\n{desc}")
    
    # ==================== 数据表和统计数据 ====================
    
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 历史数据", "📐 指标详情", "📈 收益率分析", "📋 统计分析"])
    
    with tab1:
        # 显示最新数据
        display_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        display_df.columns = ['开盘', '最高', '最低', '收盘', '成交量']
        display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d')
        
        st.dataframe(
            display_df.tail(30).style.format({
                '开盘': '{:.2f}', '最高': '{:.2f}', 
                '最低': '{:.2f}', '收盘': '{:.2f}', '成交量': '{:,.0f}'
            }).background_gradient(cmap='RdYlGn', subset=['收盘']),
            use_container_width=True,
            height=400
        )
        
        # 下载按钮
        csv = df.to_csv().encode('utf-8')
        st.download_button(
            label="📥 下载CSV数据",
            data=csv,
            file_name=f"{ticker}_数据.csv",
            mime="text/csv",
        )
    
    with tab2:
        # 技术指标详细表格
        tech_cols = [col for col in ['MA5', 'MA10', 'MA20', 'MA60', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'Volatility'] if col in df.columns]
        tech_df = df[tech_cols].tail(20].copy()
        tech_df.index = pd.to_datetime(tech_df.index).strftime('%Y-%m-%d')
        
        st.dataframe(
            tech_df.style.format('{:.2f}').background_gradient(cmap='coolwarm'),
            use_container_width=True,
            height=400
        )
    
    with tab3:
        # 收益率分析
        st.markdown("#### 不同时间周期收益率")
        
        returns = {}
        periods = {
            '5天': 5, '10天': 10, '20天': 20, 
            '60天': 60, '120天': 120, '250天': 250
        }
        
        for name, days in periods.items():
            if len(df) > days:
                ret = (df['Close'].iloc[-1] / df['Close'].iloc[-days] - 1) * 100
                returns[name] = ret
        
        cols = st.columns(len(returns))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        for i, (name, ret) in enumerate(returns.items()):
            with cols[i]:
                st.markdown(f"""
                <div style="text-align:center; padding:15px; background:{colors[i]}; border-radius:10px; opacity:0.9;">
                    <h4>{name}</h4>
                    <h2 style="color:{'green' if ret > 0 else 'red'}">{ret:+.2f}%</h2>
                </div>
                """, unsafe_allow_html=True)
        
        # 日收益率分布
        df['Daily_Return'] = df['Close'].pct_change() * 100
        daily_returns = df['Daily_Return'].dropna()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 日收益率直方图")
            fig_ret = go.Figure()
            fig_ret.add_trace(go.Histogram(
                x=daily_returns,
                nbinsx=50,
                name='日收益率分布',
                marker_color='#1E88E5',
                opacity=0.7
            ))
            fig_ret.update_layout(
                template='plotly_dark',
                title=f"日收益率分布 (均值: {daily_returns.mean():.2f}%, 标准差: {daily_returns.std():.2f}%)",
                xaxis_title="收益率 (%)",
                yaxis_title="频次",
                height=400
            )
            st.plotly_chart(fig_ret, use_container_width=True)
        
        with col2:
            st.markdown("#### 累计收益率曲线")
            cumulative_ret = (1 + df['Daily_Return'] / 100).cumprod() * 100 - 100
            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(
                x=df.index,
                y=cumulative_ret,
                mode='lines',
                name='累计收益率',
                line=dict(color='#FF9800', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 152, 0, 0.1)'
            ))
            fig_cum.update_layout(
                template='plotly_dark',
                title="累计收益率",
                xaxis_title="日期",
                yaxis_title="累计收益率 (%)",
                height=400
            )
            st.plotly_chart(fig_cum, use_container_width=True)
    
    with tab4:
        # 统计分析
        st.markdown("#### 描述性统计")
        
        stats_data = {
            '统计量': ['均值', '中位数', '标准差', '最大值', '最小值', '偏度', '峰度', '25%分位', '75%分位'],
            '收盘价': [
                df['Close'].mean(),
                df['Close'].median(),
                df['Close'].std(),
                df['Close'].max(),
                df['Close'].min(),
                df['Close'].skew(),
                df['Close'].kurtosis(),
                df['Close'].quantile(0.25),
                df['Close'].quantile(0.75)
            ],
            '收益率 (%)': [
                daily_returns.mean(),
                daily_returns.median(),
                daily_returns.std(),
                daily_returns.max(),
                daily_returns.min(),
                daily_returns.skew(),
                daily_returns.kurtosis(),
                daily_returns.quantile(0.25),
                daily_returns.quantile(0.75)
            ],
            '成交量': [
                df['Volume'].mean(),
                df['Volume'].median(),
                df['Volume'].std(),
                df['Volume'].max(),
                df['Volume'].min(),
                df['Volume'].skew(),
                df['Volume'].kurtosis(),
                df['Volume'].quantile(0.25),
                df['Volume'].quantile(0.75)
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        stats_df = stats_df.set_index('统计量')
        
        st.dataframe(
            stats_df.style.format('{:.4f}').background_gradient(cmap='viridis'),
            use_container_width=True
        )
        
        # 相关性分析
        st.markdown("#### 价格与成交量相关性")
        corr = df[['Close', 'Volume', 'Daily_Return']].dropna().corr()
        st.dataframe(
            corr.style.format('{:.4f}').background_gradient(cmap='coolwarm'),
            use_container_width=True
        )


# ===================== 运行入口 =====================

if __name__ == "__main__":
    main()
