# 量化交易系统使用说明

## 系统概述

这是一个完整的量化因子挖掘与评估系统，包括多因子策略生成、效果回测和自主迭代优化功能。系统采用模块化设计，支持并行计算，能够处理大量股票数据。

## 系统架构

系统由以下主要模块组成：

1. **quant_factor_system.py** - 核心因子挖掘与评估系统
2. **quant_optimizer.py** - 因子和策略优化器
3. **quant_main_controller.py** - 主控制器，整合所有功能
4. **first_rise_factor_analysis.py** - 首次拉升因子分析（现有）
5. **secondary_rise_factor_analysis.py** - 二次拉升因子分析（现有）

## 功能特性

### 1. 数据加载
- 并行加载所有个股数据（不包括指数和ETF）
- 自动过滤无效数据
- 支持增量更新

### 2. 因子计算
- 技术因子：收益率、成交量、波动率、动量等
- 技术指标：RSI、布林带、MACD
- 自定义因子：首次拉升因子、二次拉升因子
- 全部使用向量化计算，支持并行处理

### 3. 因子评估
- 信息系数（IC）分析
- 分组收益分析
- 胜率计算
- 夏普比率计算

### 4. 策略生成
- 多因子策略生成
- 动量策略
- 价值反转策略
- 成交量突破策略
- 复合策略

### 5. 策略回测
- 多周期回测（1天、3天、5天、10天、20天）
- 收益率计算
- 胜率统计
- 夏普比率计算
- 最大回撤分析

### 6. 优化功能
- 因子参数优化
- 策略权重优化
- 遗传算法优化
- 复合策略创建

## 使用方法

### 1. 运行完整流程

```bash
# 运行完整分析流程
uv run quant_main_controller.py --mode full

# 指定数据路径和输出路径
uv run quant_main_controller.py --mode full --data-path data/price_volume --output-path output
```

### 2. 运行特定步骤

```bash
# 仅运行因子计算和评估
uv run quant_main_controller.py --mode factors

# 仅运行策略生成
uv run quant_main_controller.py --mode strategies

# 运行回测分析
uv run quant_main_controller.py --mode backtest --strategy momentum_strategy --holding-period 5

# 生成今日交易信号
uv run quant_main_controller.py --mode signals --date 2026-01-29

# 运行优化
uv run quant_main_controller.py --mode optimize
```

### 3. 使用核心系统

```bash
# 直接运行因子系统
uv run quant_factor_system.py

# 运行优化器
uv run quant_optimizer.py
```

## 输出结果

系统会在`output`目录下生成以下结果文件：

1. **factor_performance.csv** - 因子表现评估结果
2. **strategy_summary.csv** - 策略摘要信息
3. **today_signals_YYYYMMDD_HHMMSS.txt** - 今日交易信号
4. **factor_optimization_results_YYYYMMDD_HHMMSS.json** - 因子优化结果
5. **optimization_results_YYYYMMDD_HHMMSS.json** - 其他优化结果

## 策略说明

### 1. 动量策略 (momentum_strategy)
- 条件：短期和中期动量为正，且RSI不过热
- 适用场景：趋势跟踪

### 2. 价值反转策略 (value_reversal_strategy)
- 条件：超卖且波动率较低的股票
- 适用场景：均值回归

### 3. 成交量突破策略 (volume_breakout_strategy)
- 条件：成交量放大且价格上涨的突破股票
- 适用场景：突破交易

### 4. 二次拉升策略 (secondary_rise_strategy)
- 条件：二次拉升因子触发且中期动量不过差
- 适用场景：强势股回调后的再次上涨

### 5. 复合策略 (composite_strategy)
- 条件：基于多个因子综合评分
- 适用场景：多因子组合

## 自定义扩展

### 1. 添加新因子

在`quant_factor_system.py`中的`calculate_technical_factors`方法中添加新的因子计算逻辑：

```python
# 示例：添加新的技术因子
df['new_factor'] = df.groupby('code')['close'].rolling(window=10).mean() / df['close']
```

### 2. 添加新策略

在`quant_factor_system.py`中的`generate_strategies`方法中添加新的策略：

```python
# 示例：添加新的策略
new_condition = (df['rsi_14'] < 20) & (df['volume_ratio'] > 2)
strategies['new_strategy'] = {
    'condition': new_condition,
    'description': '新策略描述'
}
```

### 3. 因子优化

使用`quant_optimizer.py`中的优化器进行因子参数优化：

```python
from quant_optimizer import QuantOptimizer

optimizer = QuantOptimizer()
result = optimizer.optimize_factor_parameters(
    df, 
    factor_function, 
    param_ranges={'window': (5, 50), 'threshold': (0.01, 0.1)}
)
```

## 性能优化

### 1. 并行计算
系统使用`multiprocessing`库进行并行数据加载，充分利用多核CPU。

### 2. 向量化操作
所有因子计算都使用pandas的向量化操作，避免循环。

### 3. 内存优化
- 及时清理不需要的中间变量
- 使用适当的数据类型减少内存占用

## 注意事项

1. **数据质量**：确保数据完整性和准确性
2. **过拟合风险**：避免过度优化参数
3. **交易成本**：实际交易中需考虑滑点和手续费
4. **市场变化**：策略需定期重新评估和调整

## 故障排除

### 1. 数据加载失败
- 检查数据路径是否正确
- 确认数据文件格式符合要求
- 查看日志文件`quant_system.log`获取详细信息

### 2. 内存不足
- 减少并行进程数
- 分批处理数据
- 增加系统内存

### 3. 计算错误
- 检查数据是否包含NaN值
- 确认所有必需列存在
- 查看日志文件获取错误详情

## 系统要求

- Python 3.7+
- pandas
- numpy
- scipy
- scikit-learn（可选，用于高级机器学习因子）

## 更新日志

- v1.0: 初始版本，包含基本因子挖掘和策略回测功能
- v1.1: 添加优化器模块，支持因子参数优化
- v1.2: 添加主控制器，整合所有功能，支持命令行操作

## 联系方式

如有问题或建议，请通过以下方式联系：
- 邮箱：example@example.com
- GitHub：github.com/example/quant-system