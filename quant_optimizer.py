#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量化因子和策略优化器

功能包括：
1. 因子参数优化
2. 策略参数优化
3. 多因子权重优化
4. 遗传算法优化
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Callable
from scipy.optimize import minimize
import random
import warnings
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuantOptimizer:
    """量化因子和策略优化器"""
    
    def __init__(self, output_path: str = "output"):
        self.output_path = output_path
        self.optimization_results = {}
        
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
    
    def optimize_factor_parameters(self, df: pd.DataFrame, factor_func: Callable, 
                                 param_ranges: Dict[str, Tuple], 
                                 target_metric: str = 'ic_return_open_5d',
                                 maximize: bool = True) -> Dict[str, Any]:
        """优化因子参数
        
        Args:
            df: 包含因子和收益率的数据框
            factor_func: 计算因子的函数
            param_ranges: 参数范围字典
            target_metric: 目标优化指标
            maximize: 是否最大化目标指标
            
        Returns:
            优化结果字典
        """
        logger.info(f"开始优化因子参数，目标指标: {target_metric}")
        
        # 定义目标函数
        def objective(params):
            # 将参数转换为字典
            param_dict = {}
            for i, (param_name, param_range) in enumerate(param_ranges.items()):
                param_dict[param_name] = params[i]
            
            # 计算因子
            try:
                df_with_factor = factor_func(df.copy(), **param_dict)
                
                # 计算目标指标
                valid_data = df_with_factor.dropna(subset=[target_metric])
                if len(valid_data) < 100:
                    return -np.inf if maximize else np.inf
                
                metric_value = valid_data['factor'].corr(valid_data[target_metric])
                
                # 返回负值以便最大化
                return -metric_value if maximize else metric_value
            except Exception as e:
                logger.error(f"计算因子时出错: {e}")
                return -np.inf if maximize else np.inf
        
        # 初始参数值
        initial_params = []
        bounds = []
        for param_name, (min_val, max_val) in param_ranges.items():
            initial_params.append((min_val + max_val) / 2)
            bounds.append((min_val, max_val))
        
        # 执行优化
        try:
            result = minimize(
                objective, 
                initial_params, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 50}
            )
            
            # 获取最优参数
            optimal_params = {}
            for i, param_name in enumerate(param_ranges.keys()):
                optimal_params[param_name] = result.x[i]
            
            # 计算最优因子
            df_optimal = factor_func(df.copy(), **optimal_params)
            
            # 计算最终指标
            valid_data = df_optimal.dropna(subset=[target_metric])
            final_metric = valid_data['factor'].corr(valid_data[target_metric])
            
            optimization_result = {
                'optimal_params': optimal_params,
                'optimal_value': final_metric,
                'success': result.success,
                'message': result.message
            }
            
            logger.info(f"因子参数优化完成: {optimal_params}, 最优值: {final_metric:.4f}")
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"优化过程中出错: {e}")
            return None
    
    def optimize_strategy_weights(self, df: pd.DataFrame, factors: List[str], 
                               target_metric: str = 'return_open_5d',
                               max_weight: float = 1.0) -> Dict[str, Any]:
        """优化多因子策略权重
        
        Args:
            df: 包含因子和收益率的数据框
            factors: 因子列表
            target_metric: 目标收益率指标
            max_weight: 最大权重限制
            
        Returns:
            优化结果字典
        """
        logger.info(f"开始优化多因子策略权重，因子: {factors}")
        
        # 确保所有因子和收益率存在
        available_factors = [f for f in factors if f in df.columns]
        if not available_factors:
            logger.error("没有可用的因子")
            return None
        
        if target_metric not in df.columns:
            logger.error(f"目标收益率指标不存在: {target_metric}")
            return None
        
        # 清理数据
        valid_data = df[available_factors + [target_metric]].dropna()
        if len(valid_data) < 100:
            logger.error("有效数据量不足")
            return None
        
        # 定义目标函数
        def objective(weights):
            # 计算组合因子
            composite_factor = np.zeros(len(valid_data))
            for i, factor in enumerate(available_factors):
                composite_factor += weights[i] * valid_data[factor]
            
            # 计算与收益率的相关系数
            correlation = np.corrcoef(composite_factor, valid_data[target_metric])[0, 1]
            
            # 添加正则化项防止过拟合
            regularization = 0.01 * np.sum(np.abs(weights))
            
            # 返回负值以便最大化
            return -(abs(correlation) - regularization)
        
        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # 权重和为1
        ]
        
        # 边界条件
        bounds = [(0, max_weight) for _ in available_factors]
        
        # 初始权重
        initial_weights = np.ones(len(available_factors)) / len(available_factors)
        
        # 执行优化
        try:
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100}
            )
            
            # 获取最优权重
            optimal_weights = dict(zip(available_factors, result.x))
            
            # 计算最优组合因子
            composite_factor = np.zeros(len(valid_data))
            for factor, weight in optimal_weights.items():
                composite_factor += weight * valid_data[factor]
            
            # 计算最终相关系数
            final_correlation = np.corrcoef(composite_factor, valid_data[target_metric])[0, 1]
            
            optimization_result = {
                'optimal_weights': optimal_weights,
                'optimal_correlation': final_correlation,
                'success': result.success,
                'message': result.message,
                'factors_used': available_factors
            }
            
            logger.info(f"策略权重优化完成: {optimal_weights}, 最优相关系数: {final_correlation:.4f}")
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"权重优化过程中出错: {e}")
            return None
    
    def genetic_algorithm_optimization(self, df: pd.DataFrame, 
                                     objective_func: Callable,
                                     param_ranges: Dict[str, Tuple],
                                     population_size: int = 50,
                                     generations: int = 30,
                                     mutation_rate: float = 0.1,
                                     maximize: bool = True) -> Dict[str, Any]:
        """遗传算法优化
        
        Args:
            df: 数据框
            objective_func: 目标函数
            param_ranges: 参数范围
            population_size: 种群大小
            generations: 迭代代数
            mutation_rate: 变异率
            maximize: 是否最大化目标
            
        Returns:
            优化结果字典
        """
        logger.info(f"开始遗传算法优化，种群大小: {population_size}, 代数: {generations}")
        
        # 初始化种群
        population = []
        for _ in range(population_size):
            individual = {}
            for param_name, (min_val, max_val) in param_ranges.items():
                individual[param_name] = random.uniform(min_val, max_val)
            population.append(individual)
        
        # 进化过程
        best_individual = None
        best_fitness = -np.inf if maximize else np.inf
        
        for generation in range(generations):
            # 计算适应度
            fitness_scores = []
            for individual in population:
                try:
                    fitness = objective_func(df, individual)
                    fitness_scores.append(fitness)
                    
                    # 更新最优个体
                    if (maximize and fitness > best_fitness) or (not maximize and fitness < best_fitness):
                        best_fitness = fitness
                        best_individual = individual.copy()
                except Exception as e:
                    logger.warning(f"计算适应度时出错: {e}")
                    fitness_scores.append(-np.inf if maximize else np.inf)
            
            # 选择
            if maximize:
                selected_indices = np.argsort(fitness_scores)[-population_size//2:]
            else:
                selected_indices = np.argsort(fitness_scores)[:population_size//2]
            
            selected_population = [population[i] for i in selected_indices]
            
            # 交叉和变异
            new_population = selected_population.copy()
            while len(new_population) < population_size:
                # 选择两个父代
                parent1 = random.choice(selected_population)
                parent2 = random.choice(selected_population)
                
                # 交叉
                child = {}
                for param_name in param_ranges.keys():
                    if random.random() < 0.5:
                        child[param_name] = parent1[param_name]
                    else:
                        child[param_name] = parent2[param_name]
                
                # 变异
                for param_name in param_ranges.keys():
                    if random.random() < mutation_rate:
                        min_val, max_val = param_ranges[param_name]
                        child[param_name] = random.uniform(min_val, max_val)
                
                new_population.append(child)
            
            population = new_population
            
            if generation % 5 == 0:
                logger.info(f"第 {generation} 代，最优适应度: {best_fitness:.4f}")
        
        optimization_result = {
            'optimal_params': best_individual,
            'optimal_fitness': best_fitness,
            'generations': generations,
            'population_size': population_size
        }
        
        logger.info(f"遗传算法优化完成: {best_individual}, 最优适应度: {best_fitness:.4f}")
        
        return optimization_result
    
    def optimize_multiple_factors(self, df: pd.DataFrame, 
                                factor_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量优化多个因子
        
        Args:
            df: 数据框
            factor_configs: 因子配置列表，每个配置包含因子函数、参数范围和目标指标
            
        Returns:
            批量优化结果
        """
        logger.info(f"开始批量优化 {len(factor_configs)} 个因子")
        
        results = {}
        
        for i, config in enumerate(factor_configs):
            factor_name = config.get('name', f'factor_{i}')
            factor_func = config['function']
            param_ranges = config['param_ranges']
            target_metric = config.get('target_metric', 'ic_return_open_5d')
            maximize = config.get('maximize', True)
            
            logger.info(f"优化因子 {factor_name} ({i+1}/{len(factor_configs)})")
            
            result = self.optimize_factor_parameters(
                df, factor_func, param_ranges, target_metric, maximize
            )
            
            if result:
                results[factor_name] = result
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.output_path, f"factor_optimization_results_{timestamp}.json")
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"因子优化结果已保存到: {results_file}")
        except Exception as e:
            logger.error(f"保存结果时出错: {e}")
        
        return results
    
    def create_composite_strategy(self, df: pd.DataFrame, 
                                factor_weights: Dict[str, float],
                                threshold: float = 0.7) -> pd.Series:
        """创建基于权重的复合策略
        
        Args:
            df: 数据框
            factor_weights: 因子权重字典
            threshold: 策略触发阈值
            
        Returns:
            策略条件序列
        """
        logger.info(f"创建复合策略，使用因子: {list(factor_weights.keys())}")
        
        # 确保所有因子存在
        available_factors = [f for f in factor_weights.keys() if f in df.columns]
        if not available_factors:
            logger.error("没有可用的因子")
            return pd.Series([False] * len(df))
        
        # 计算复合因子
        composite_factor = pd.Series([0.0] * len(df))
        
        for factor, weight in factor_weights.items():
            if factor in df.columns:
                # 标准化因子
                factor_values = df[factor]
                standardized_factor = (factor_values - factor_values.mean()) / factor_values.std()
                composite_factor += weight * standardized_factor
        
        # 创建策略条件
        strategy_condition = composite_factor > composite_factor.quantile(threshold)
        
        return strategy_condition
    
    def save_optimization_results(self, results: Dict[str, Any], prefix: str = "optimization"):
        """保存优化结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_results_{timestamp}.json"
        filepath = os.path.join(self.output_path, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"优化结果已保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存优化结果时出错: {e}")

# 示例因子函数
def example_momentum_factor(df: pd.DataFrame, window: int = 20, threshold: float = 0.02) -> pd.DataFrame:
    """示例动量因子"""
    df = df.sort_values(['code', 'date'])
    df['momentum'] = df.groupby('code')['close'].pct_change(window)
    df['factor'] = (df['momentum'] > threshold).astype(int)
    return df

def example_volatility_factor(df: pd.DataFrame, window: int = 20, threshold: float = 0.05) -> pd.DataFrame:
    """示例波动率因子"""
    df = df.sort_values(['code', 'date'])
    returns = df.groupby('code')['close'].pct_change()
    df['volatility'] = returns.groupby(df['code']).rolling(window=window).std().reset_index(0, drop=True)
    df['factor'] = (df['volatility'] < threshold).astype(int)
    return df

def main():
    """主函数 - 示例用法"""
    # 创建优化器
    optimizer = QuantOptimizer()
    
    # 示例优化配置
    factor_configs = [
        {
            'name': 'momentum_factor',
            'function': example_momentum_factor,
            'param_ranges': {
                'window': (5, 60),
                'threshold': (0.01, 0.1)
            },
            'target_metric': 'return_open_5d',
            'maximize': True
        },
        {
            'name': 'volatility_factor',
            'function': example_volatility_factor,
            'param_ranges': {
                'window': (10, 50),
                'threshold': (0.02, 0.1)
            },
            'target_metric': 'return_open_5d',
            'maximize': True
        }
    ]
    
    print("量化因子优化器已创建，包含以下功能:")
    print("1. 因子参数优化")
    print("2. 策略权重优化")
    print("3. 遗传算法优化")
    print("4. 批量因子优化")
    print("5. 复合策略创建")
    
    # 注意：实际使用时需要提供数据框df
    # results = optimizer.optimize_multiple_factors(df, factor_configs)

if __name__ == "__main__":
    main()