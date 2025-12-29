import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox

# 添加以下两行解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def calculate_rsi(data, window=10, rsi_params=None):
    """
    计算RSI指标
    :param data: 股票数据
    :param window: RSI周期，默认10天
    :param rsi_params: RSI参数列表，格式为 [周期, 参数3, 参数97, 参数20, 参数80]
    """
    if rsi_params is None:
        window = 10  # 使用传入的默认值
    else:
        window = rsi_params[0] if len(rsi_params) > 0 else 10
    
    delta = data['收盘'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=3, slow=33, signal=17):
    """
    计算MACD指标
    :param data: 股票数据
    :param fast: 快线周期，默认3天
    :param slow: 慢线周期，默认33天
    :param signal: 信号线周期，默认17天
    """
    exp1 = data['收盘'].ewm(span=fast).mean()
    exp2 = data['收盘'].ewm(span=slow).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_kdj(data, n=10, k_period=5, d_period=4):
    """
    计算KDJ指标
    :param data: 股票数据
    :param n: RSV周期，默认10天
    :param k_period: K值平滑参数，默认5天
    :param d_period: D值平滑参数，默认4天
    """
    low_min = data['最低'].rolling(window=n).min()
    high_max = data['最高'].rolling(window=n).max()
    
    rsv = 100 * ((data['收盘'] - low_min) / (high_max - low_min))
    # 使用传入的参数进行平滑处理
    k = rsv.ewm(com=k_period-1).mean()  # ewm的com参数为alpha=1/(1+com)
    d = k.ewm(com=d_period-1).mean()
    j = 3 * k - 2 * d
    
    return k, d, j

def calculate_moving_averages(data, ma_periods=[6, 12, 24]):
    """
    计算移动平均线
    """
    mas = {}
    for period in ma_periods:
        mas[f'MA_{period}'] = data['收盘'].rolling(window=period).mean()
    return mas

def get_stock_data_with_indicators(stock_code, period_days=180, period="daily"):
    """
    获取股票数据并计算技术指标
    :param stock_code: 股票代码
    :param period_days: 获取数据的天数
    :param period: 数据周期，"daily"表示日线，"weekly"表示周线，"monthly"表示月线
    """
    try:
        # 根据周期调整数据获取天数
        if period == "weekly":
            # 周线数据，需要获取更多历史数据以确保有足够的周数据
            start_date = (datetime.now() - timedelta(days=period_days*7)).strftime('%Y%m%d')
        elif period == "monthly":
            # 月线数据，需要获取更多历史数据以确保有足够的月数据
            start_date = (datetime.now() - timedelta(days=period_days*30)).strftime('%Y%m%d')
        else:
            # 日线数据
            start_date = (datetime.now() - timedelta(days=period_days+60)).strftime('%Y%m%d')
        
        end_date = datetime.now().strftime('%Y%m%d')
        
        stock_zh_a_hist = ak.stock_zh_a_hist(
            symbol=stock_code, 
            period=period,  # 使用传入的周期参数
            start_date=start_date, 
            end_date=end_date, 
            adjust=""
        )
        
        if stock_zh_a_hist.empty:
            print(f"无法获取股票 {stock_code} 的{period}数据")
            return None
        
        # 重命名列名以匹配常用名称
        stock_zh_a_hist.rename(columns={
            '日期': '日期',
            '开盘': '开盘',
            '收盘': '收盘', 
            '最高': '最高',
            '最低': '最低',
            '成交量': '成交量'
        }, inplace=True)
        
        # 确保数据列存在
        required_columns = ['开盘', '收盘', '最高', '最低', '成交量']
        for col in required_columns:
            if col not in stock_zh_a_hist.columns:
                print(f"缺少必要列: {col}")
                return None
        
        # 计算移动平均线
        ma_data = calculate_moving_averages(stock_zh_a_hist)
        for ma_name, ma_values in ma_data.items():
            stock_zh_a_hist[ma_name] = ma_values
        
        # 计算技术指标
        stock_zh_a_hist['RSI'] = calculate_rsi(stock_zh_a_hist, window=10)
        macd_line, signal_line, histogram = calculate_macd(stock_zh_a_hist, fast=3, slow=33, signal=17)
        stock_zh_a_hist['MACD'] = macd_line
        stock_zh_a_hist['MACD_Signal'] = signal_line
        stock_zh_a_hist['MACD_Histogram'] = histogram
        k, d, j = calculate_kdj(stock_zh_a_hist, n=10, k_period=5, d_period=4)
        stock_zh_a_hist['K'] = k
        stock_zh_a_hist['D'] = d
        stock_zh_a_hist['J'] = j
        
        # 按日期排序并重置索引
        stock_zh_a_hist = stock_zh_a_hist.sort_values('日期').reset_index(drop=True)
        
        return stock_zh_a_hist
        
    except Exception as e:
        print(f"获取股票数据时发生错误: {e}")
        return None

def display_current_indicators(stock_data, stock_code, period="daily"):
    """
    显示当前最新的技术指标值
    """
    if stock_data is None or stock_data.empty:
        print("没有数据可显示")
        return
    
    latest = stock_data.iloc[-1]
    date = latest['日期'] if '日期' in stock_data.columns else stock_data.index[-1]
    
    period_name = "日线" if period == "daily" else ("周线" if period == "weekly" else "月线")
    print(f"\n{'='*60}")
    print(f"{stock_code} 最新{period_name}技术指标 ({date})")
    print(f"{'='*60}")
    
    print(f"收盘价: {latest['收盘']:.2f}")
    
    # 移动平均线
    print("\n移动平均线:")
    ma_periods = [6, 12, 24]  # 修改为新的均线周期
    for period in ma_periods:
        ma_col = f'MA_{period}'
        if ma_col in latest:
            ma_value = latest[ma_col]
            if not pd.isna(ma_value):
                print(f"  MA{period}: {ma_value:.2f}")
            else:
                print(f"  MA{period}: N/A (数据不足)")
    
    # RSI
    rsi = latest['RSI']
    rsi_level = ""
    if pd.isna(rsi):
        print(f"RSI: N/A (数据不足)")
    else:
        # 使用RSI10 3 97 20 80的参数来判断
        if rsi > 80:  # 第5个参数：80
            rsi_level = "严重超买"
        elif rsi > 20:  # 第4个参数：20
            rsi_level = "正常"
        else:  # 小于20
            rsi_level = "严重超卖"
        print(f"RSI: {rsi:.2f} ({rsi_level})")
    
    # MACD
    macd = latest['MACD']
    signal = latest['MACD_Signal']
    histogram = latest['MACD_Histogram']
    if pd.isna(macd) or pd.isna(signal):
        print("MACD: N/A (数据不足)")
    else:
        print(f"MACD: {macd:.4f}")
        print(f"Signal: {signal:.4f}")
        print(f"Histogram: {histogram:.4f}")
    
    # KDJ
    k = latest['K']
    d = latest['D']
    j = latest['J']
    if pd.isna(k) or pd.isna(d) or pd.isna(j):
        print("KDJ: N/A (数据不足)")
    else:
        print(f"K: {k:.2f}")
        print(f"D: {d:.2f}")
        print(f"J: {j:.2f}")

def plot_indicators_with_scroll(stock_data, stock_code, period="daily"):
    """
    绘制带滚动条的技术指标图表
    """
    if stock_data is None or stock_data.empty:
        print("没有数据可绘制")
        return
    
    period_name = "日线" if period == "daily" else ("周线" if period == "weekly" else "月线")
    # 设置图表
    fig = plt.figure(figsize=(14, 10))  # 减少高度，因为去掉了成交量图
    fig.suptitle(f'{stock_code} {period_name}技术指标分析 - 使用滑块滚动查看不同时间段', fontsize=16)
    
    # 计算滑块范围
    total_points = len(stock_data)
    window_size = min(60, total_points)  # 显示窗口大小
    max_start_idx = max(0, total_points - window_size)
    
    # 创建子图 - 移除了成交量图，现在是4个子图
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)
    ax_main = fig.add_subplot(gs[0])
    ax_rsi = fig.add_subplot(gs[1])
    ax_macd = fig.add_subplot(gs[2])
    ax_kdj = fig.add_subplot(gs[3])
    
    # 创建滑块
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
    slider = Slider(ax_slider, '滚动', 0, max_start_idx, valinit=max_start_idx, valfmt='%d')
    
    def update_plot(val):
        start_idx = int(slider.val)
        end_idx = min(start_idx + window_size, total_points)
        data_slice = stock_data.iloc[start_idx:end_idx]
        
        # 清除子图
        ax_main.clear()
        ax_rsi.clear()
        ax_macd.clear()
        ax_kdj.clear()
        
        # 主图：绘制K线图
        # 使用日期索引而不是整数索引，使x轴更清晰
        x_indices = range(len(data_slice))
        
        for i in range(len(data_slice)):
            # 获取当前K线数据
            open_price = data_slice['开盘'].iloc[i]
            close_price = data_slice['收盘'].iloc[i]
            high_price = data_slice['最高'].iloc[i]
            low_price = data_slice['最低'].iloc[i]
            
            # 确定K线颜色
            color = 'red' if close_price >= open_price else 'green'
            
            # 绘制上下影线
            ax_main.plot([x_indices[i], x_indices[i]], [low_price, high_price], color=color, linewidth=1)
            
            # 绘制实体
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            ax_main.bar(x_indices[i], body_height, bottom=body_bottom, width=0.8, color=color, alpha=0.8)
        
        # 绘制移动平均线 - 使用实际数据点索引
        colors = ['blue', 'red', 'green']  # 只有3条均线
        ma_periods = [6, 12, 24]  # 修改为新的均线周期
        
        # 确保只绘制当前数据切片范围内的MA值
        for i, period in enumerate(ma_periods):
            ma_col = f'MA_{period}'
            if ma_col in data_slice.columns:
                # 获取有效的MA值（非NaN）
                ma_values = data_slice[ma_col]
                valid_indices = x_indices[:len(ma_values)]
                
                # 只绘制有效的MA值
                valid_ma_mask = ~pd.isna(ma_values)
                if valid_ma_mask.any():
                    ax_main.plot(
                        [x for x, valid in zip(valid_indices, valid_ma_mask) if valid],
                        [val for val, valid in zip(ma_values, valid_ma_mask) if valid],
                        label=f'MA{period}',
                        color=colors[i % len(colors)],
                        linewidth=1.5
                    )
        
        # 设置x轴标签为日期（如果原数据有日期列）
        ax_main.set_title(f'{stock_code} K线及移动平均线 ({period_name}) (数据点: {start_idx+1}-{end_idx})')
        ax_main.legend(loc='upper left')
        ax_main.grid(True, alpha=0.3)
        
        # RSI图 - 使用RSI10 3 97 20 80的参数
        if 'RSI' in data_slice.columns:
            ax_rsi.plot(x_indices, data_slice['RSI'], label='RSI', color='blue')
            ax_rsi.axhline(y=80, color='r', linestyle='--', label='超买线(80)')  # 参数80
            ax_rsi.axhline(y=20, color='g', linestyle='--', label='超卖线(20)')  # 参数20
            ax_rsi.set_title('RSI指标 (RSI10 3 97 20 80)')
            ax_rsi.legend(loc='upper left')
            ax_rsi.grid(True, alpha=0.3)
            ax_rsi.set_ylim(0, 100)
        
        # MACD图 - 使用MACD 3 33 17的参数
        if all(col in data_slice.columns for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
            ax_macd.plot(x_indices, data_slice['MACD'], label='MACD(3,33,17)', color='blue')
            ax_macd.plot(x_indices, data_slice['MACD_Signal'], label='Signal(17)', color='red')
            ax_macd.bar(x_indices, data_slice['MACD_Histogram'], label='Histogram', alpha=0.3, color='gray')
            ax_macd.set_title('MACD指标 (MACD 3 33 17)')
            ax_macd.legend(loc='upper left')
            ax_macd.grid(True, alpha=0.3)
        
        # KDJ图 - 使用KDJ 10 5 4的参数
        if all(col in data_slice.columns for col in ['K', 'D', 'J']):
            ax_kdj.plot(x_indices, data_slice['K'], label='K(10,5,4)', color='blue')
            ax_kdj.plot(x_indices, data_slice['D'], label='D(10,5,4)', color='red')
            ax_kdj.plot(x_indices, data_slice['J'], label='J(10,5,4)', color='green')
            ax_kdj.axhline(y=80, color='r', linestyle='--', label='超买线(80)')
            ax_kdj.axhline(y=20, color='g', linestyle='--', label='超卖线(20)')
            ax_kdj.set_title('KDJ指标 (KDJ 10 5 4)')
            ax_kdj.legend(loc='upper left')
            ax_kdj.grid(True, alpha=0.3)
            ax_kdj.set_ylim(0, 100)
        
        # 设置x轴标签（在最后一个子图上）
        ax_kdj.set_xlabel('时间索引')
        
        # 刷新图表
        fig.canvas.draw()
    
    # 连接滑块事件
    slider.on_changed(update_plot)
    
    # 初始绘制
    update_plot(0)
    
    # 显示图表
    plt.show()


def analyze_ma_trend(stock_data, period="daily"):
    """
    分析移动平均线趋势
    """
    if stock_data is None or len(stock_data) < 2:
        return
    
    latest = stock_data.iloc[-1]
    prev = stock_data.iloc[-2]
    
    period_name = "日线" if period == "daily" else ("周线" if period == "weekly" else "月线")
    print(f"\n{'='*50}")
    print(f"移动平均线趋势分析 ({period_name})")
    print(f"{'='*50}")
    
    # 检查短期MA是否上穿中期MA（金叉）
    if ('MA_6' in stock_data.columns and 'MA_12' in stock_data.columns and
        not pd.isna(latest['MA_6']) and not pd.isna(latest['MA_12']) and
        not pd.isna(prev['MA_6']) and not pd.isna(prev['MA_12'])):
        
        if latest['MA_6'] > latest['MA_12'] and prev['MA_6'] <= prev['MA_12']:
            print("MA6上穿MA12: 短期金叉信号")
        elif latest['MA_6'] < latest['MA_12'] and prev['MA_6'] >= prev['MA_12']:
            print("MA6下穿MA12: 短期死叉信号")
    
    # 检查中期MA与长期MA趋势
    if all(col in stock_data.columns for col in ['MA_12', 'MA_24']) and \
       all(not pd.isna(latest[col]) for col in ['MA_12', 'MA_24']) and \
       all(not pd.isna(prev[col]) for col in ['MA_12', 'MA_24']):
        
        if latest['MA_12'] > latest['MA_24'] and prev['MA_12'] <= prev['MA_24']:
            print("MA12上穿MA24: 中期金叉信号")
        elif latest['MA_12'] < latest['MA_24'] and prev['MA_12'] >= prev['MA_24']:
            print("MA12下穿MA24: 中期死叉信号")
    
    # 检查价格与MA的关系
    if all(col in stock_data.columns for col in ['MA_6', 'MA_12', 'MA_24']) and \
       all(not pd.isna(latest[col]) for col in ['MA_6', 'MA_12', 'MA_24']):
        
        current_price = latest['收盘']
        ma6, ma12, ma24 = latest['MA_6'], latest['MA_12'], latest['MA_24']
        
        if current_price > ma6 > ma12 > ma24:
            print("价格强势: 价格 > MA6 > MA12 > MA24 (多头排列)")
        elif current_price < ma6 < ma12 < ma24:
            print("价格弱势: 价格 < MA6 < MA12 < MA24 (空头排列)")

def analyze_signal(stock_data, period="daily"):
    """
    简单的技术信号分析
    """
    if stock_data is None or len(stock_data) < 2:
        return
    
    latest = stock_data.iloc[-1]
    prev = stock_data.iloc[-2]
    
    period_name = "日线" if period == "daily" else ("周线" if period == "weekly" else "月线")
    print(f"\n{'='*50}")
    print(f"技术信号分析 ({period_name})")
    print(f"{'='*50}")
    
    # 检查是否为有效数值
    if pd.isna(latest['RSI']) or pd.isna(prev['RSI']):
        print("RSI信号: 数据不足")
    else:
        # RSI信号 - 使用RSI10 3 97 20 80的参数
        if latest['RSI'] > 80:  # 超买线80
            print("RSI信号: 严重超买，可能回调")
        elif latest['RSI'] < 20:  # 超卖线20
            print("RSI信号: 严重超卖，可能反弹")
        elif prev['RSI'] < 20 and latest['RSI'] >= 20:
            print("RSI信号: 从严重超卖回升，买入信号")
        elif prev['RSI'] > 80 and latest['RSI'] <= 80:
            print("RSI信号: 从严重超买回落，卖出信号")
    
    # MACD信号 - 使用MACD 3 33 17的参数
    if pd.isna(latest['MACD']) or pd.isna(prev['MACD']):
        print("MACD信号: 数据不足")
    else:
        if (latest['MACD'] > latest['MACD_Signal'] and 
            not pd.isna(prev['MACD']) and prev['MACD'] <= prev['MACD_Signal']):
            print("MACD信号: 金叉，买入信号")
        elif (latest['MACD'] < latest['MACD_Signal'] and 
              not pd.isna(prev['MACD']) and prev['MACD'] >= prev['MACD_Signal']):
            print("MACD信号: 死叉，卖出信号")
    
    # KDJ信号 - 使用KDJ 10 5 4的参数
    if pd.isna(latest['K']) or pd.isna(prev['K']):
        print("KDJ信号: 数据不足")
    else:
        if latest['K'] > latest['D'] and prev['K'] <= prev['D']:
            print("KDJ信号: K线上穿D线，买入信号")
        elif latest['K'] < latest['D'] and prev['K'] >= prev['D']:
            print("KDJ信号: K线下穿D线，卖出信号")
        
        if (latest['K'] < 20 and latest['D'] < 20 and latest['J'] < 20):
            print("KDJ信号: 三线严重超卖，强烈反弹信号")
        elif (latest['K'] > 80 and latest['D'] > 80 and latest['J'] > 80):
            print("KDJ信号: 三线严重超买，强烈回调信号")

def get_and_analyze_stock_indicators(stock_code, period="daily"):
    """
    获取股票数据并分析技术指标
    :param stock_code: 股票代码
    :param period: 数据周期，"daily"表示日线，"weekly"表示周线，"monthly"表示月线
    """
    period_name = "日线" if period == "daily" else ("周线" if period == "weekly" else "月线")
    print(f"正在获取 {stock_code} 的{period_name}技术指标数据...")
    print(f"使用指标参数: RSI10 3 97 20 80, MACD 3 33 17, KDJ 10 5 4")
    
    # 获取带指标的股票数据
    stock_data_with_indicators = get_stock_data_with_indicators(stock_code, period=period)
    
    if stock_data_with_indicators is None:
        print("获取数据失败")
        return None
    
    # 显示当前指标
    display_current_indicators(stock_data_with_indicators, stock_code, period)
    
    # 移动平均线趋势分析
    analyze_ma_trend(stock_data_with_indicators, period)
    
    # 技术信号分析
    analyze_signal(stock_data_with_indicators, period)
    
    # 绘制带滚动条的图表（可选）
    try:
        plot_indicators_with_scroll(stock_data_with_indicators, stock_code, period)
    except ImportError:
        print("matplotlib未安装，跳过图表绘制")
    
    return stock_data_with_indicators

def interactive_analysis():
    """
    交互式分析，允许用户选择日线、周线或月线
    """
    root = tk.Tk()
    root.title("股票技术指标分析工具")
    root.geometry("400x300")
    
    # 股票代码输入
    tk.Label(root, text="股票代码:").grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
    stock_code_entry = tk.Entry(root)
    stock_code_entry.insert(0, "600724")  # 默认值
    stock_code_entry.grid(row=0, column=1, padx=10, pady=10)
    
    # 周期选择
    tk.Label(root, text="数据周期:").grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
    period_var = tk.StringVar(value="daily")
    period_frame = tk.Frame(root)
    period_frame.grid(row=1, column=1, padx=10, pady=10)
    
    tk.Radiobutton(period_frame, text="日线", variable=period_var, value="daily").pack(side=tk.LEFT)
    tk.Radiobutton(period_frame, text="周线", variable=period_var, value="weekly").pack(side=tk.LEFT)
    tk.Radiobutton(period_frame, text="月线", variable=period_var, value="monthly").pack(side=tk.LEFT)
    
    def start_analysis():
        stock_code = stock_code_entry.get().strip()
        if not stock_code:
            messagebox.showerror("错误", "请输入股票代码")
            return
        
        period = period_var.get()
        root.destroy()
        
        print("股票技术指标分析工具")
        print("使用指标参数: RSI10 3 97 20 80, MACD 3 33 17, KDJ 10 5 4")
        print("图表支持滚动查看历史数据")
        
        # 获取并分析股票指标
        stock_data = get_and_analyze_stock_indicators(stock_code, period=period)
        
        if stock_data is not None:
            print(f"\n{'='*50}")
            print("分析完成")
            print("="*50)
        else:
            print("分析失败")
    
    # 开始分析按钮
    tk.Button(root, text="开始分析", command=start_analysis).grid(row=2, column=0, columnspan=2, pady=20)
    
    # 说明文字
    tk.Label(root, text="说明: 使用RSI10 3 97 20 80, MACD 3 33 17, KDJ 10 5 4参数", fg="gray").grid(row=3, column=0, columnspan=2)
    
    root.mainloop()

if __name__ == "__main__":
    interactive_analysis()