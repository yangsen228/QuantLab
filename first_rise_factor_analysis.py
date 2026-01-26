#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
首次拉升因子分析脚本

1. 读取data/price_volume中所有个股数据（不包括指数和etf）
2. 过滤得到所有个股在2008年1月1日以后的数据
3. 计算首次拉升因子，当满足如下条件时打标为1：
   - t-20到t-1横盘震荡
   - t日收盘价突破t-20到t-1的最高价
   - t日成交量大于t-20到t-1平均成交量的2倍
4. 计算t+1开盘买入到t+2收盘卖出的收益率
5. 评估首次拉升因子和收益率的相关性以及胜率
6. 获取date between '2026-01-15' and '2026-01-23'首次拉升因子达标为1的股票代码输出
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import glob
import warnings
warnings.filterwarnings('ignore')

def is_individual_stock(filename):
    """判断是否为个股文件（非指数和ETF）"""
    # 排除包含"指数"、"ETF"等关键词的文件
    exclude_keywords = ['指数', 'ETF', '基金', '债', '国债', '企业债', '可转债']
    return not any(keyword in filename for keyword in exclude_keywords)

def load_stock_data(data_path):
    """加载所有个股数据"""
    print("正在加载个股数据...")
    
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    
    # 过滤出个股文件
    stock_files = [f for f in csv_files if is_individual_stock(os.path.basename(f))]
    
    print(f"找到 {len(stock_files)} 个个股文件")
    
    # 存储所有股票数据
    all_data = []
    
    for i, file in enumerate(stock_files):
        if i % 100 == 0:
            print(f"正在处理第 {i+1}/{len(stock_files)} 个文件")
        
        try:
            # 读取数据
            df = pd.read_csv(file)
            
            # 过滤2008年1月1日以后的数据
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'] >= '2008-01-01']
            
            if len(df) > 0:
                all_data.append(df)
        except Exception as e:
            print(f"处理文件 {file} 时出错: {e}")
    
    if all_data:
        # 合并所有数据
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"共加载 {len(combined_data)} 条数据")
        return combined_data
    else:
        print("未找到有效数据")
        return pd.DataFrame()

def calculate_first_rise_factor(df):
    """计算首次拉升因子"""
    print("正在计算首次拉升因子...")
    
    # 按股票代码和日期排序
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    
    # 初始化首次拉升因子
    df['first_rise_factor'] = 0
    
    # 计算每只股票的最小日期（上市日期）
    df['listing_date'] = df.groupby('code')['date'].transform('min')
    # 计算上市天数
    df['listing_days'] = (df['date'] - df['listing_date']).dt.days
    
    # 计算横盘震荡条件（t-20到t-1价格振幅较小）
    df['high_20'] = df.groupby('code')['high'].rolling(window=20, min_periods=20).max().shift(1).reset_index(0, drop=True)
    df['low_20'] = df.groupby('code')['low'].rolling(window=20, min_periods=20).min().shift(1).reset_index(0, drop=True)
    df['amplitude_20'] = (df['high_20'] - df['low_20']) / df['low_20']
    
    # 计算t-20到t-1的最高价
    df['max_price_20'] = df.groupby('code')['high'].rolling(window=20, min_periods=20).max().shift(1).reset_index(0, drop=True)
    
    # 计算t-20到t-1的平均成交量
    df['avg_volume_20'] = df.groupby('code')['volume'].rolling(window=20, min_periods=20).mean().shift(1).reset_index(0, drop=True)
    
    # 计算t-1的收盘价
    df['close_t1'] = df.groupby('code')['close'].shift(1)
    
    # 计算t-20的收盘价
    df['close_t20'] = df.groupby('code')['close'].shift(20)
    
    # 计算t-20到t-1收盘价的变化率
    df['price_change_rate_t20_t1'] = (df['close_t1'] - df['close_t20']) / df['close_t20']
    
    # 使用向量化操作优化计算效率
    # 创建条件掩码
    condition_mask = (
        (df['listing_days'] > 365) &  # 上市天数大于一年
        (df['amplitude_20'] < 0.06) &  # t-20到t-1横盘震荡（振幅小于6%）
        (df['close'] > df['max_price_20']) &  # t日收盘价突破t-20到t-1的最高价
        (df['volume'] > df['avg_volume_20'] * 1.5) &  # t日成交量大于t-20到t-1平均成交量的1.5倍
        (df['volume'] < df['avg_volume_20'] * 3) &  # t日成交量小于t-20到t-1平均成交量的3倍
        (df['volume'] > df.groupby('code')['volume'].shift(1)) &  # t的成交量大于t-1成交量
        (df['close_t1'] <= df['max_price_20']) &  # t-1的收盘价小于等于t-20到t-1的最高价
        (df['price_change_rate_t20_t1'].abs() < 0.03)  # t-20和t-1收盘价的变化率小于3%
    )
    
    # 应用条件
    df.loc[condition_mask, 'first_rise_factor'] = 1
    
    return df

def calculate_returns(df):
    """计算t+1开盘买入到t+2收盘卖出的收益率"""
    print("正在计算收益率...")
    
    # 使用向量化操作计算t+1开盘到t+2收盘的收益率
    t1_open = df.groupby('code')['open'].shift(-1)
    t2_close = df.groupby('code')['close'].shift(-2)
    
    # 计算收益率并处理可能的除零情况
    df['return_t1_open_t2_close'] = np.where(
        t1_open != 0,
        (t2_close - t1_open) / t1_open * 100,
        np.nan
    )
    
    return df

def analyze_correlation(df):
    """评估首次拉升因子和收益率的相关性"""
    print("正在评估相关性...")
    
    # 过滤掉NaN值
    valid_data = df.dropna(subset=['first_rise_factor', 'return_t1_open_t2_close'])
    
    if len(valid_data) == 0:
        print("没有有效数据进行相关性分析")
        return None
    
    # 计算相关系数
    correlation = valid_data['first_rise_factor'].corr(valid_data['return_t1_open_t2_close'])
    
    # 分组统计
    factor_0 = valid_data[valid_data['first_rise_factor'] == 0]['return_t1_open_t2_close']
    factor_1 = valid_data[valid_data['first_rise_factor'] == 1]['return_t1_open_t2_close']
    
    # 计算因子为1的胜率（收益率大于0的比例）
    if len(factor_1) > 0:
        win_rate_factor_1 = (factor_1 > 0).mean() * 100
    else:
        win_rate_factor_1 = 0
    
    print(f"首次拉升因子和收益率的相关系数: {correlation:.4f}")
    print(f"因子为0的样本数: {len(factor_0)}, 平均收益率: {factor_0.mean():.4f}%")
    print(f"因子为1的样本数: {len(factor_1)}, 平均收益率: {factor_1.mean():.4f}%")
    print(f"因子为1的胜率（收益率>0）: {win_rate_factor_1:.2f}%")
    
    return {
        'correlation': correlation,
        'factor_0_count': len(factor_0),
        'factor_0_mean_return': factor_0.mean(),
        'factor_1_count': len(factor_1),
        'factor_1_mean_return': factor_1.mean(),
        'factor_1_win_rate': win_rate_factor_1
    }

def get_qualified_stocks(df, start_date, end_date):
    """获取指定日期范围内首次拉升因子达标为1的股票代码"""
    print(f"正在获取 {start_date} 到 {end_date} 首次拉升因子达标为1的股票...")
    
    # 转换日期格式
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # 过滤指定日期范围内且因子为1的股票
    qualified_stocks = df[(df['date'] >= start_date) & (df['date'] <= end_date) & (df['first_rise_factor'] == 1)]
    
    if len(qualified_stocks) > 0:
        print(f"{start_date} 到 {end_date} 首次拉升因子达标为1的股票代码:")
        # 按日期分组显示
        for date in sorted(qualified_stocks['date'].unique()):
            date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
            daily_stocks = qualified_stocks[qualified_stocks['date'] == date]['code'].unique()
            print(f"{date_str}: {', '.join(daily_stocks)}")
        
        return qualified_stocks[['date', 'code']].drop_duplicates()
    else:
        print(f"{start_date} 到 {end_date} 没有首次拉升因子达标为1的股票")
        return pd.DataFrame(columns=['date', 'code'])

def main():
    """主函数"""
    # 数据路径
    data_path = "data/price_volume"
    
    # 1. 读取个股数据
    df = load_stock_data(data_path)
    
    if df.empty:
        print("没有数据可供分析")
        return
    
    # 2. 计算首次拉升因子
    df = calculate_first_rise_factor(df)
    
    # 3. 计算收益率
    df = calculate_returns(df)
    
    # 4. 评估相关性
    correlation_result = analyze_correlation(df)
    
    # 5. 获取指定日期范围内达标股票
    # 动态计算日期：结束时间为当前日期，开始时间为10天前
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - pd.Timedelta(days=10)).strftime('%Y-%m-%d')
    qualified_stocks = get_qualified_stocks(df, start_date, end_date)
    
    # 保存结果
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存分析结果
    if correlation_result:
        analysis_result = pd.DataFrame({
            'metric': ['correlation', 'factor_0_count', 'factor_0_mean_return', 'factor_1_count', 'factor_1_mean_return', 'factor_1_win_rate'],
            'value': [
                correlation_result['correlation'],
                correlation_result['factor_0_count'],
                correlation_result['factor_0_mean_return'],
                correlation_result['factor_1_count'],
                correlation_result['factor_1_mean_return'],
                correlation_result['factor_1_win_rate']
            ]
        })
        
        analysis_result.to_csv(os.path.join(output_dir, "first_rise_correlation_analysis_t1_open_t2_close.csv"), index=False, encoding='utf-8-sig')
    
    # 保存达标股票列表
    if not qualified_stocks.empty:
        qualified_stocks.to_csv(os.path.join(output_dir, "first_rise_qualified_stocks_20260115_20260123.csv"), index=False, encoding='utf-8-sig')
    
    # 保存因子为1的样本数据
    factor_1_samples = df[df['first_rise_factor'] == 1]
    if not factor_1_samples.empty:
        factor_1_samples.to_csv(os.path.join(output_dir, "first_rise_factor_1_samples_t1_open_t2_close.csv"), index=False, encoding='utf-8-sig')
        print(f"因子为1的样本已保存，共{len(factor_1_samples)}条记录")
    
    # 保存完整数据（可选，文件可能较大）
    # df.to_csv(os.path.join(output_dir, "first_rise_full_data.csv"), index=False, encoding='utf-8-sig')
    
    print("首次拉升因子分析完成，结果已保存到output目录")

if __name__ == "__main__":
    main()