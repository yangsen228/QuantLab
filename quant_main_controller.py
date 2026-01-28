#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量化交易系统主控制器

整合因子挖掘、策略生成、优化和回测的完整流程
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from quant_factor_system import QuantFactorSystem
from quant_optimizer import QuantOptimizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quant_system.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class QuantMainController:
    """量化交易系统主控制器"""
    
    def __init__(self, data_path: str = "data/price_volume", output_path: str = "output"):
        self.data_path = data_path
        self.output_path = output_path
        self.factor_system = QuantFactorSystem(data_path, output_path)
        self.optimizer = QuantOptimizer(output_path)
        self.current_data = None
        
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        
        logger.info("量化交易系统主控制器初始化完成")
    
    def run_full_pipeline(self, steps: List[str] = None):
        """运行完整流程
        
        Args:
            steps: 要运行的步骤列表，如果为None则运行所有步骤
        """
        if steps is None:
            steps = ['load_data', 'calculate_factors', 'evaluate_factors', 
                     'generate_strategies', 'backtest_strategies', 'optimize_weights']
        
        logger.info(f"开始运行流程步骤: {steps}")
        
        # 1. 加载数据
        if 'load_data' in steps:
            logger.info("步骤1: 加载数据")
            self.current_data = self.factor_system.load_all_stock_data()
            if self.current_data is None or self.current_data.empty:
                logger.error("数据加载失败，流程终止")
                return False
        
        # 2. 计算因子
        if 'calculate_factors' in steps and self.current_data is not None:
            logger.info("步骤2: 计算因子")
            self.current_data = self.factor_system.calculate_technical_factors(self.current_data)
            self.current_data = self.factor_system.calculate_forward_returns(self.current_data)
        
        # 3. 评估因子
        if 'evaluate_factors' in steps and self.current_data is not None:
            logger.info("步骤3: 评估因子")
            factor_performance = self.factor_system.evaluate_factors(self.current_data)
            logger.info(f"因子评估完成，共评估 {len(factor_performance)} 个因子")
        
        # 4. 生成策略
        if 'generate_strategies' in steps and self.current_data is not None:
            logger.info("步骤4: 生成策略")
            strategies = self.factor_system.generate_strategies(self.current_data)
            logger.info(f"策略生成完成，共生成 {len(strategies)} 个策略")
        
        # 5. 回测策略
        if 'backtest_strategies' in steps and self.current_data is not None:
            logger.info("步骤5: 回测策略")
            backtest_results = self.factor_system.run_all_backtests(self.current_data)
            logger.info(f"策略回测完成，共回测 {len(backtest_results)} 个策略")
        
        # 6. 优化权重
        if 'optimize_weights' in steps and self.current_data is not None:
            logger.info("步骤6: 优化权重")
            self._optimize_strategy_weights()
        
        # 7. 保存结果
        logger.info("步骤7: 保存结果")
        self.factor_system.save_results()
        
        logger.info("完整流程运行完成")
        return True
    
    def _optimize_strategy_weights(self):
        """优化策略权重"""
        if self.current_data is None:
            return
        
        # 获取表现最好的因子
        if not self.factor_system.factor_performance:
            logger.warning("没有因子表现数据，跳过权重优化")
            return
        
        # 选择IC值最高的前5个因子
        ic_scores = {}
        for factor, metrics in self.factor_system.factor_performance.items():
            if 'ic_return_open_5d' in metrics:
                ic_scores[factor] = abs(metrics['ic_return_open_5d'])
        
        top_factors = sorted(ic_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        selected_factors = [factor for factor, score in top_factors]
        
        logger.info(f"选择优化权重的前5个因子: {selected_factors}")
        
        # 优化权重
        weight_result = self.optimizer.optimize_strategy_weights(
            self.current_data, 
            selected_factors, 
            target_metric='return_open_5d'
        )
        
        if weight_result:
            logger.info(f"权重优化完成: {weight_result['optimal_weights']}")
            logger.info(f"最优相关系数: {weight_result['optimal_correlation']:.4f}")
            
            # 创建优化后的复合策略
            optimal_weights = weight_result['optimal_weights']
            composite_condition = self.optimizer.create_composite_strategy(
                self.current_data, 
                optimal_weights, 
                threshold=0.8
            )
            
            # 回测优化后的策略
            logger.info("回测优化后的复合策略")
            optimized_result = self.factor_system.backtest_strategy(
                self.current_data, 
                'optimized_composite_strategy', 
                composite_condition,
                holding_period=5
            )
            
            if optimized_result:
                logger.info(f"优化策略回测结果: 平均收益={optimized_result['avg_return']:.2f}%, 胜率={optimized_result['win_rate']:.2f}%")
    
    def run_factor_optimization(self, factor_name: str = None):
        """运行因子优化
        
        Args:
            factor_name: 要优化的因子名称，如果为None则优化所有因子
        """
        if self.current_data is None:
            logger.error("没有可用数据，请先运行数据加载步骤")
            return
        
        logger.info(f"开始因子优化: {factor_name if factor_name else '所有因子'}")
        
        # 这里可以添加具体的因子优化逻辑
        # 由于需要具体的因子函数，这里仅作为示例
        
        logger.info("因子优化完成")
    
    def generate_today_signals(self, date_str: str = None):
        """生成今日交易信号
        
        Args:
            date_str: 日期字符串，格式为'YYYY-MM-DD'，如果为None则使用最近日期
        """
        if self.current_data is None:
            logger.error("没有可用数据，请先运行数据加载步骤")
            return
        
        logger.info("生成今日交易信号")
        
        # 获取最新日期的数据
        if date_str is None:
            latest_date = self.current_data['date'].max()
            date_str = latest_date.strftime('%Y-%m-%d')
        
        latest_data = self.current_data[self.current_data['date'] == date_str]
        
        if latest_data.empty:
            logger.warning(f"没有找到 {date_str} 的数据")
            return
        
        # 应用所有策略
        signals = {}
        for strategy_name, strategy_info in self.factor_system.strategies.items():
            condition = strategy_info['condition']
            selected_stocks = latest_data[condition]
            
            if not selected_stocks.empty:
                signals[strategy_name] = selected_stocks['code'].tolist()
                logger.info(f"{strategy_name}: 选中 {len(selected_stocks)} 只股票")
            else:
                logger.info(f"{strategy_name}: 没有选中股票")
        
        # 保存信号
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        signals_file = os.path.join(self.output_path, f"today_signals_{timestamp}.txt")
        
        try:
            with open(signals_file, 'w', encoding='utf-8') as f:
                f.write(f"交易信号生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"数据日期: {date_str}\n\n")
                
                for strategy, stock_list in signals.items():
                    f.write(f"{strategy}:\n")
                    for stock in stock_list:
                        f.write(f"  {stock}\n")
                    f.write(f"  共 {len(stock_list)} 只股票\n\n")
            
            logger.info(f"今日交易信号已保存到: {signals_file}")
            
        except Exception as e:
            logger.error(f"保存交易信号时出错: {e}")
    
    def run_backtest_analysis(self, strategy_name: str = None, holding_period: int = 5):
        """运行回测分析
        
        Args:
            strategy_name: 策略名称，如果为None则回测所有策略
            holding_period: 持仓周期（天）
        """
        if self.current_data is None:
            logger.error("没有可用数据，请先运行数据加载步骤")
            return
        
        logger.info(f"开始回测分析: {strategy_name if strategy_name else '所有策略'}")
        
        if strategy_name:
            # 回测指定策略
            if strategy_name in self.factor_system.strategies:
                strategy_info = self.factor_system.strategies[strategy_name]
                result = self.factor_system.backtest_strategy(
                    self.current_data, 
                    strategy_name, 
                    strategy_info['condition'],
                    holding_period
                )
                
                if result:
                    logger.info(f"{strategy_name} 回测结果:")
                    logger.info(f"  信号数量: {result['total_signals']}")
                    logger.info(f"  平均收益: {result['avg_return']:.2f}%")
                    logger.info(f"  胜率: {result['win_rate']:.2f}%")
                    logger.info(f"  夏普比率: {result['sharpe_ratio']:.2f}")
                    logger.info(f"  最大回撤: {result['max_drawdown']:.2f}%")
            else:
                logger.error(f"策略 {strategy_name} 不存在")
        else:
            # 回测所有策略
            results = self.factor_system.run_all_backtests(self.current_data)
            logger.info("所有策略回测完成")
            
            # 打印摘要
            for name, result in results.items():
                logger.info(f"{name}: 信号={result['total_signals']}, 收益={result['avg_return']:.2f}%, 胜率={result['win_rate']:.2f}%")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='量化交易系统主控制器')
    parser.add_argument('--mode', choices=['full', 'factors', 'strategies', 'backtest', 'optimize', 'signals'],
                        default='full', help='运行模式')
    parser.add_argument('--data-path', default='data/price_volume', help='数据路径')
    parser.add_argument('--output-path', default='output', help='输出路径')
    parser.add_argument('--strategy', help='指定策略名称')
    parser.add_argument('--date', help='指定日期 (YYYY-MM-DD)')
    parser.add_argument('--holding-period', type=int, default=5, help='持仓周期（天）')
    
    args = parser.parse_args()
    
    # 创建控制器
    controller = QuantMainController(args.data_path, args.output_path)
    
    # 根据模式运行
    if args.mode == 'full':
        controller.run_full_pipeline()
    elif args.mode == 'factors':
        controller.run_full_pipeline(['load_data', 'calculate_factors', 'evaluate_factors'])
    elif args.mode == 'strategies':
        controller.run_full_pipeline(['load_data', 'calculate_factors', 'generate_strategies'])
    elif args.mode == 'backtest':
        controller.run_full_pipeline(['load_data', 'calculate_factors', 'generate_strategies'])
        controller.run_backtest_analysis(args.strategy, args.holding_period)
    elif args.mode == 'optimize':
        controller.run_full_pipeline(['load_data', 'calculate_factors', 'evaluate_factors'])
        controller.run_factor_optimization()
    elif args.mode == 'signals':
        controller.run_full_pipeline(['load_data', 'calculate_factors', 'generate_strategies'])
        controller.generate_today_signals(args.date)

if __name__ == "__main__":
    main()