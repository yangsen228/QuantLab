#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的量化因子挖掘与评估系统

功能包括：
1. 多因子挖掘与计算
2. 因子评估与筛选
3. 多因子策略生成
4. 策略回测
5. 结果可视化与保存
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob
import warnings
import json
from typing import Dict, List, Tuple, Any
import multiprocessing as mp
from functools import partial
import logging

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuantFactorSystem:
    """量化因子挖掘与评估系统"""
    
    def __init__(self, data_path: str = "data/price_volume", output_path: str = "output"):
        self.data_path = data_path
        self.output_path = output_path
        self.stock_data = None
        self.factors = {}
        self.factor_performance = {}
        self.strategies = {}
        
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
    
    def is_individual_stock(self, filename: str) -> bool:
        """判断是否为个股文件（非指数和ETF）"""
        exclude_keywords = ['指数', 'ETF', '基金', '债', '国债', '企业债', '可转债']
        return not any(keyword in filename for keyword in exclude_keywords)
    
    def load_all_stock_data(self) -> pd.DataFrame:
        """并行加载所有个股数据"""
        logger.info("开始加载所有个股数据...")
        
        # 获取所有CSV文件
        csv_files = glob.glob(os.path.join(self.data_path, "*.csv"))
        
        # 过滤出个股文件
        stock_files = [f for f in csv_files if self.is_individual_stock(os.path.basename(f))]
        
        logger.info(f"找到 {len(stock_files)} 个个股文件")
        
        # 使用多进程并行加载数据
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(self._load_single_stock, stock_files)
        
        # 合并所有数据
        all_data = [df for df in results if df is not None and not df.empty]
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"共加载 {len(combined_data)} 条数据")
            self.stock_data = combined_data
            return combined_data
        else:
            logger.warning("未找到有效数据")
            return pd.DataFrame()
    
    def _load_single_stock(self, file_path: str) -> pd.DataFrame:
        """加载单只股票数据"""
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                return pd.DataFrame()
                
            # 转换日期格式
            df['date'] = pd.to_datetime(df['date'])
            
            # 过滤2008年1月1日以后的数据
            df = df[df['date'] >= '2008-01-01']
            
            # 确保必要的列存在
            required_cols = ['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'preclose']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"文件 {file_path} 缺少必要列: {missing_cols}")
                return pd.DataFrame()
            
            return df
        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {e}")
            return pd.DataFrame()
    
    def calculate_technical_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术因子"""
        logger.info("计算技术因子...")
        
        # 确保数据按股票代码和日期排序
        df = df.sort_values(['code', 'date']).reset_index(drop=True)
        
        # 价格收益率因子
        df['return_1d'] = df.groupby('code')['close'].pct_change()
        df['return_5d'] = df.groupby('code')['close'].pct_change(5)
        df['return_10d'] = df.groupby('code')['close'].pct_change(10)
        df['return_20d'] = df.groupby('code')['close'].pct_change(20)
        
        # 成交量因子
        df['volume_ma_5'] = df.groupby('code')['volume'].rolling(window=5).mean().reset_index(0, drop=True)
        df['volume_ma_20'] = df.groupby('code')['volume'].rolling(window=20).mean().reset_index(0, drop=True)
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        
        # 波动率因子
        df['price_high_low'] = (df['high'] - df['low']) / df['close']
        df['volatility_5d'] = df.groupby('code')['return_1d'].rolling(window=5).std().reset_index(0, drop=True)
        df['volatility_20d'] = df.groupby('code')['return_1d'].rolling(window=20).std().reset_index(0, drop=True)
        
        # 价格动量因子
        df['momentum_12m'] = df.groupby('code')['close'].pct_change(252)
        df['momentum_1m'] = df.groupby('code')['close'].pct_change(21)
        df['momentum_3m'] = df.groupby('code')['close'].pct_change(63)
        
        # 相对强弱因子
        df['rsi_14'] = self._calculate_rsi(df, window=14)
        
        # 布林带因子
        df['bb_upper'], df['bb_lower'] = self._calculate_bollinger_bands(df)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # MACD因子
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df)
        
        # 首次拉升因子
        df = self._calculate_first_rise_factor(df)
        
        # 二次拉升因子
        df = self._calculate_secondary_rise_factor(df)
        
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = df.groupby('code')['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.groupby(df['code']).rolling(window=window).mean().reset_index(0, drop=True)
        avg_loss = loss.groupby(df['code']).rolling(window=window).mean().reset_index(0, drop=True)
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, window: int = 20, num_std: int = 2) -> Tuple[pd.Series, pd.Series]:
        """计算布林带"""
        ma = df.groupby('code')['close'].rolling(window=window).mean().reset_index(0, drop=True)
        std = df.groupby('code')['close'].rolling(window=window).std().reset_index(0, drop=True)
        
        upper_band = ma + (std * num_std)
        lower_band = ma - (std * num_std)
        
        return upper_band, lower_band
    
    def _calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算MACD指标"""
        exp1 = df.groupby('code')['close'].ewm(span=fast).mean().reset_index(0, drop=True)
        exp2 = df.groupby('code')['close'].ewm(span=slow).mean().reset_index(0, drop=True)
        macd = exp1 - exp2
        signal = macd.groupby(df['code']).ewm(span=signal).mean().reset_index(0, drop=True)
        hist = macd - signal
        
        return macd, signal, hist
    
    def _calculate_first_rise_factor(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算首次拉升因子"""
        # 计算上市天数
        df['listing_date'] = df.groupby('code')['date'].transform('min')
        df['listing_days'] = (df['date'] - df['listing_date']).dt.days
        
        # 计算横盘震荡条件
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
        
        # 首次拉升因子条件
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
        
        df['first_rise_factor'] = 0
        df.loc[condition_mask, 'first_rise_factor'] = 1
        
        return df
    
    def _calculate_secondary_rise_factor(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算二次拉升因子"""
        # 计算当日涨幅
        df['today_return'] = (df['close'] - df['preclose']) / df['preclose'] * 100
        
        # 计算成交量变化率
        df['volume_change'] = df.groupby('code')['volume'].pct_change()
        
        # 二次拉升因子条件
        condition_mask = (
            (df['today_return'] > 5) &  # 当日涨幅超过5%
            (df.groupby('code')['volume_change'].shift(-1) < 0) &  # t+1缩量
            (df.groupby('code')['volume_change'].shift(-2) < 0) &  # t+2缩量
            (df.groupby('code')['volume_change'].shift(-3) < 0) &  # t+3缩量
            (df.groupby('code')['close'].shift(-3) > df['open']) &  # t+3收盘价高于t开盘价
            (df.groupby('code')['close'].shift(-1) < df.groupby('code')['open'].shift(-1)) &  # t+1收绿
            (df.groupby('code')['close'].shift(-2) < df.groupby('code')['open'].shift(-2)) &  # t+2收绿
            (df.groupby('code')['close'].shift(-3) < df.groupby('code')['open'].shift(-3)) &  # t+3收绿
            (df.groupby('code')['close'].shift(-2) < df.groupby('code')['close'].shift(-1)) &  # t+2比t+1收盘低
            (df.groupby('code')['close'].shift(-3) < df.groupby('code')['close'].shift(-2))  # t+3比t+2收盘低
        )
        
        df['secondary_rise_factor'] = 0
        df.loc[condition_mask, 'secondary_rise_factor'] = 1
        
        return df
    
    def calculate_forward_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算未来收益率"""
        logger.info("计算未来收益率...")
        
        # 计算不同时间窗口的未来收益率
        periods = [1, 3, 5, 10, 20]
        
        for period in periods:
            # 开盘价到开盘价的收益率
            future_open = df.groupby('code')['open'].shift(-period)
            df[f'return_open_{period}d'] = (future_open - df['open']) / df['open'] * 100
            
            # 收盘价到收盘价的收益率
            future_close = df.groupby('code')['close'].shift(-period)
            df[f'return_close_{period}d'] = (future_close - df['close']) / df['close'] * 100
        
        return df
    
    def evaluate_factors(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """评估因子表现"""
        logger.info("评估因子表现...")
        
        # 定义因子列表
        factor_list = [
            'return_1d', 'return_5d', 'return_10d', 'return_20d',
            'volume_ratio', 'volatility_5d', 'volatility_20d',
            'momentum_1m', 'momentum_3m', 'momentum_12m',
            'rsi_14', 'bb_position', 'macd', 'macd_hist',
            'first_rise_factor', 'secondary_rise_factor'
        ]
        
        # 定义收益率列
        return_cols = [f'return_open_{p}d' for p in [1, 3, 5, 10, 20]]
        
        results = {}
        
        for factor in factor_list:
            if factor not in df.columns:
                continue
                
            logger.info(f"评估因子: {factor}")
            
            factor_results = {}
            
            # 去除NaN值
            valid_data = df.dropna(subset=[factor] + return_cols)
            
            if len(valid_data) < 100:  # 数据量太少，跳过
                continue
            
            # 计算IC值（信息系数）
            for return_col in return_cols:
                ic = valid_data[factor].corr(valid_data[return_col])
                factor_results[f'ic_{return_col}'] = ic
            
            # 分组分析（分为5组）
            valid_data['factor_group'] = pd.qcut(valid_data[factor], 5, labels=False, duplicates='drop')
            
            for return_col in return_cols:
                group_returns = valid_data.groupby('factor_group')[return_col].mean()
                factor_results[f'group_return_spread_{return_col}'] = group_returns.iloc[-1] - group_returns.iloc[0]
                
                # 计算因子最高组的胜率
                top_group = valid_data[valid_data['factor_group'] == valid_data['factor_group'].max()]
                win_rate = (top_group[return_col] > 0).mean() * 100
                factor_results[f'top_group_win_rate_{return_col}'] = win_rate
            
            results[factor] = factor_results
        
        self.factor_performance = results
        return results
    
    def generate_strategies(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """生成多因子策略"""
        logger.info("生成多因子策略...")
        
        strategies = {}
        
        # 策略1：动量策略
        momentum_condition = (
            (df['momentum_1m'] > 0) & 
            (df['momentum_3m'] > 0) & 
            (df['rsi_14'] < 70)
        )
        strategies['momentum_strategy'] = {
            'condition': momentum_condition,
            'description': '短期和中期动量为正，且RSI不过热的股票'
        }
        
        # 策略2：价值反转策略
        value_condition = (
            (df['rsi_14'] < 30) & 
            (df['bb_position'] < 0.2) & 
            (df['volatility_20d'] < df['volatility_20d'].median())
        )
        strategies['value_reversal_strategy'] = {
            'condition': value_condition,
            'description': '超卖且波动率较低的价值反转股票'
        }
        
        # 策略3：成交量突破策略
        volume_condition = (
            (df['volume_ratio'] > 2) & 
            (df['return_1d'] > 2) & 
            (df['first_rise_factor'] == 1)
        )
        strategies['volume_breakout_strategy'] = {
            'condition': volume_condition,
            'description': '成交量放大且价格上涨的突破股票'
        }
        
        # 策略4：二次拉升策略
        secondary_condition = (
            (df['secondary_rise_factor'] == 1) &
            (df['momentum_3m'] > -5)
        )
        strategies['secondary_rise_strategy'] = {
            'condition': secondary_condition,
            'description': '二次拉升因子触发且中期动量不过差的股票'
        }
        
        # 策略5：多因子综合策略
        composite_score = (
            df['momentum_1m'].rank(pct=True) * 0.2 +
            df['momentum_3m'].rank(pct=True) * 0.2 +
            (1 - df['volatility_20d'].rank(pct=True)) * 0.2 +
            df['volume_ratio'].rank(pct=True) * 0.2 +
            (1 - df['rsi_14'].abs() / 100).rank(pct=True) * 0.2
        )
        composite_condition = composite_score > 0.8
        strategies['composite_strategy'] = {
            'condition': composite_condition,
            'description': '基于多个因子综合评分的策略',
            'score': composite_score
        }
        
        self.strategies = strategies
        return strategies
    
    def backtest_strategy(self, df: pd.DataFrame, strategy_name: str, condition: pd.Series, 
                         holding_period: int = 5) -> Dict[str, Any]:
        """回测单个策略"""
        logger.info(f"回测策略: {strategy_name}")
        
        # 筛选满足条件的股票
        selected_stocks = df[condition].copy()
        
        if selected_stocks.empty:
            logger.warning(f"策略 {strategy_name} 没有选中任何股票")
            return None
        
        # 计算持仓期收益率
        future_return_col = f'return_open_{holding_period}d'
        if future_return_col not in df.columns:
            logger.error(f"缺少收益率列: {future_return_col}")
            return None
        
        # 确保数据对齐
        selected_stocks = selected_stocks[['date', 'code', 'close']].copy()
        selected_stocks['signal_date'] = selected_stocks['date']
        
        # 获取未来收益率
        future_returns = df[['date', 'code', future_return_col]].copy()
        future_returns['entry_date'] = future_returns['date'] - pd.Timedelta(days=holding_period)
        
        # 合并信号和未来收益率
        backtest_result = pd.merge(
            selected_stocks, 
            future_returns[['entry_date', 'code', future_return_col]], 
            left_on=['signal_date', 'code'], 
            right_on=['entry_date', 'code'], 
            how='inner'
        )
        
        if backtest_result.empty:
            logger.warning(f"策略 {strategy_name} 回测结果为空")
            return None
        
        # 计算策略表现指标
        avg_return = backtest_result[future_return_col].mean()
        win_rate = (backtest_result[future_return_col] > 0).mean() * 100
        sharpe_ratio = backtest_result[future_return_col].mean() / backtest_result[future_return_col].std() if backtest_result[future_return_col].std() > 0 else 0
        
        # 计算最大回撤
        cumulative_returns = (1 + backtest_result[future_return_col] / 100).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        result = {
            'total_signals': len(backtest_result),
            'avg_return': avg_return,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'backtest_data': backtest_result
        }
        
        logger.info(f"策略 {strategy_name} 回测完成: 信号数={len(backtest_result)}, 平均收益={avg_return:.2f}%, 胜率={win_rate:.2f}%")
        
        return result
    
    def run_all_backtests(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """运行所有策略回测"""
        logger.info("运行所有策略回测...")
        
        backtest_results = {}
        
        for strategy_name, strategy_info in self.strategies.items():
            condition = strategy_info['condition']
            result = self.backtest_strategy(df, strategy_name, condition)
            
            if result:
                backtest_results[strategy_name] = result
        
        return backtest_results
    
    def save_results(self):
        """保存分析结果"""
        logger.info("保存分析结果...")
        
        # 保存因子表现
        if self.factor_performance:
            factor_df = pd.DataFrame.from_dict(self.factor_performance, orient='index')
            factor_df.to_csv(os.path.join(self.output_path, "factor_performance.csv"), encoding='utf-8-sig')
        
        # 保存策略回测结果
        if self.strategies:
            strategy_summary = []
            for name, info in self.strategies.items():
                summary = {
                    'strategy_name': name,
                    'description': info['description']
                }
                strategy_summary.append(summary)
            
            strategy_df = pd.DataFrame(strategy_summary)
            strategy_df.to_csv(os.path.join(self.output_path, "strategy_summary.csv"), index=False, encoding='utf-8-sig')
        
        logger.info("分析结果已保存到output目录")
    
    def run_complete_analysis(self):
        """运行完整分析流程"""
        logger.info("开始运行完整量化因子分析...")
        
        # 1. 加载数据
        df = self.load_all_stock_data()
        if df.empty:
            logger.error("没有数据可供分析")
            return
        
        # 2. 计算因子
        df = self.calculate_technical_factors(df)
        
        # 3. 计算未来收益率
        df = self.calculate_forward_returns(df)
        
        # 4. 评估因子
        factor_performance = self.evaluate_factors(df)
        
        # 5. 生成策略
        strategies = self.generate_strategies(df)
        
        # 6. 回测策略
        backtest_results = self.run_all_backtests(df)
        
        # 7. 保存结果
        self.save_results()
        
        # 8. 输出摘要
        self._print_summary(factor_performance, backtest_results)
        
        logger.info("完整量化因子分析完成")
    
    def _print_summary(self, factor_performance: Dict, backtest_results: Dict):
        """打印分析摘要"""
        print("\n" + "="*80)
        print("量化因子分析摘要")
        print("="*80)
        
        print("\n【因子表现】")
        if factor_performance:
            # 显示前10个最佳因子
            ic_scores = {}
            for factor, metrics in factor_performance.items():
                # 使用5天收益率的IC值作为排序标准
                ic_key = 'ic_return_open_5d'
                if ic_key in metrics:
                    ic_scores[factor] = abs(metrics[ic_key])
            
            sorted_factors = sorted(ic_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            for factor, ic_score in sorted_factors:
                print(f"{factor}: IC={ic_score:.4f}")
        
        print("\n【策略回测结果】")
        if backtest_results:
            for strategy, result in backtest_results.items():
                print(f"{strategy}: 信号数={result['total_signals']}, 平均收益={result['avg_return']:.2f}%, 胜率={result['win_rate']:.2f}%")

def main():
    """主函数"""
    # 创建量化因子系统实例
    qfs = QuantFactorSystem()
    
    # 运行完整分析
    qfs.run_complete_analysis()

if __name__ == "__main__":
    main()