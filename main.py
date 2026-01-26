#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股数据获取脚本
使用baostock库获取2020年至今的股票、指数、ETF数据
包括日粒度量价数据、基本面数据和行业分类数据
"""

import baostock as bs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据保存路径
DATA_DIR = "data"
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

def login_bs():
    """登录baostock"""
    lg = bs.login()
    if lg.error_code != '0':
        logger.error(f"登录失败: {lg.error_msg}")
        return False
    logger.info("baostock登录成功")
    return True

def logout_bs():
    """登出baostock"""
    bs.logout()
    logger.info("baostock登出")

def get_stock_list():
    """获取A股股票列表"""
    logger.info("正在获取A股股票列表...")
    
    # 获取全部A股
    rs = bs.query_stock_basic()
    if rs.error_code != '0':
        logger.error(f"获取股票列表失败: {rs.error_msg}")
        return None
    
    stocks = []
    while rs.next():
        stocks.append(rs.get_row_data())
    
    df_stocks = pd.DataFrame(stocks, columns=rs.fields)
    
    # 过滤A股（剔除科创板和创业板的特殊处理）
    df_stocks = df_stocks[df_stocks['code'].str.startswith(('sh.', 'sz.'))]
    
    logger.info(f"共获取到 {len(df_stocks)} 只A股")
    return df_stocks

def get_index_list():
    """获取指数列表"""
    logger.info("正在获取指数列表...")
    
    # 主要指数代码
    index_codes = [
        'sh.000001',  # 上证指数
        'sh.000002',  # A股指数
        'sh.000003',  # B股指数
        'sh.000016',  # 上证50
        'sh.000300',  # 沪深300
        'sh.000905',  # 中证500
        'sz.399001',  # 深证成指
        'sz.399006',  # 创业板指
        'sz.399300',  # 沪深300(深)
    ]
    
    indices = []
    for code in index_codes:
        rs = bs.query_stock_basic(code=code)
        if rs.error_code == '0' and rs.next():
            indices.append(rs.get_row_data())
    
    df_indices = pd.DataFrame(indices, columns=['code', 'code_name', 'ipoDate', 'outDate', 'type', 'status'])
    logger.info(f"共获取到 {len(df_indices)} 个指数")
    return df_indices

def get_etf_list():
    """获取ETF列表"""
    logger.info("正在获取ETF列表...")
    
    # ETF通常以510、159、515开头
    rs = bs.query_stock_basic()
    if rs.error_code != '0':
        logger.error(f"获取ETF列表失败: {rs.error_msg}")
        return None
    
    etfs = []
    while rs.next():
        row = rs.get_row_data()
        code = row[0]
        # 过滤ETF代码
        if (code.startswith('sh.510') or code.startswith('sh.515') or 
            code.startswith('sz.159') or 'ETF' in row[1]):
            etfs.append(row)
    
    df_etfs = pd.DataFrame(etfs, columns=rs.fields)
    logger.info(f"共获取到 {len(df_etfs)} 只ETF")
    return df_etfs

def get_daily_data(code, start_date, end_date):
    """获取日K线数据"""
    rs = bs.query_history_k_data_plus(
        code=code,
        fields="date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3"  # 复权类型，3=后复权
    )
    
    if rs.error_code != '0':
        logger.error(f"获取{code}日K数据失败: {rs.error_msg}")
        return None
    
    data = []
    while rs.next():
        data.append(rs.get_row_data())
    
    if not data:
        return None
    
    df = pd.DataFrame(data, columns=rs.fields)
    
    # 数据类型转换
    numeric_cols = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['date'] = pd.to_datetime(df['date'])
    
    return df

def get_financial_data(code, year):
    """获取财务数据"""
    rs = bs.query_profit_data(code=code, year=year, quarter=4)
    
    if rs.error_code != '0':
        logger.error(f"获取{code}财务数据失败: {rs.error_msg}")
        return None
    
    data = []
    while rs.next():
        data.append(rs.get_row_data())
    
    if not data:
        return None
    
    df = pd.DataFrame(data, columns=rs.fields)
    return df

def get_industry_info(code):
    """获取行业分类信息"""
    rs = bs.query_stock_industry(code=code)
    
    if rs.error_code != '0':
        logger.error(f"获取{code}行业信息失败: {rs.error_msg}")
        return None
    
    data = []
    while rs.next():
        data.append(rs.get_row_data())
    
    if not data:
        return None
    
    df = pd.DataFrame(data, columns=rs.fields)
    return df

def save_data(df, filename):
    """保存数据到CSV文件"""
    if df is None or df.empty:
        return
    
    filepath = os.path.join(DATA_DIR, filename)
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    logger.info(f"数据已保存到: {filepath}")

def process_security(code, name, security_type):
    """处理单个证券的数据获取和保存"""
    try:
        # 获取日K数据
        daily_df = get_daily_data(code, START_DATE, END_DATE)
        if daily_df is not None and not daily_df.empty:
            filename = f"{security_type}_{code.replace('.', '_')}_{name.replace(' ', '_').replace('*', '')}.csv"
            save_data(daily_df, filename)
        
        # 获取行业信息（仅股票）
        if security_type == 'stock':
            industry_df = get_industry_info(code)
            if industry_df is not None and not industry_df.empty:
                industry_filename = f"industry_{code.replace('.', '_')}_{name.replace(' ', '_').replace('*', '')}.csv"
                save_data(industry_df, industry_filename)
        
        # 获取财务数据（仅股票）
        if security_type == 'stock':
            current_year = datetime.now().year
            for year in range(2020, current_year):
                financial_df = get_financial_data(code, year)
                if financial_df is not None and not financial_df.empty:
                    financial_filename = f"financial_{code.replace('.', '_')}_{name.replace(' ', '_').replace('*', '')}_{year}.csv"
                    save_data(financial_df, financial_filename)
        
        return True
        
    except Exception as e:
        logger.error(f"处理{code}时发生错误: {str(e)}")
        return False

def main():
    """主函数"""
    logger.info("开始获取A股数据...")
    logger.info(f"数据时间范围: {START_DATE} 至 {END_DATE}")
    
    # 确保数据目录存在
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 登录baostock
    if not login_bs():
        return
    
    try:
        # 获取各类证券列表
        stock_list = get_stock_list()
        index_list = get_index_list()
        etf_list = get_etf_list()
        
        if stock_list is None:
            logger.error("获取股票列表失败")
            return
        
        # 处理股票数据
        logger.info("开始处理股票数据...")
        success_count = 0
        for _, row in tqdm(stock_list.iterrows(), total=len(stock_list), desc="股票进度"):
            code = row['code']
            name = row['code_name']
            if process_security(code, name, 'stock'):
                success_count += 1
            time.sleep(0.1)  # 避免请求过快
        
        logger.info(f"股票数据处理完成，成功: {success_count}/{len(stock_list)}")
        
        # 处理指数数据
        if index_list is not None:
            logger.info("开始处理指数数据...")
            success_count = 0
            for _, row in index_list.iterrows():
                code = row['code']
                name = row['code_name']
                if process_security(code, name, 'index'):
                    success_count += 1
                time.sleep(0.1)
            
            logger.info(f"指数数据处理完成，成功: {success_count}/{len(index_list)}")
        
        # 处理ETF数据
        if etf_list is not None:
            logger.info("开始处理ETF数据...")
            success_count = 0
            for _, row in etf_list.iterrows():
                code = row['code']
                name = row['code_name']
                if process_security(code, name, 'etf'):
                    success_count += 1
                time.sleep(0.1)
            
            logger.info(f"ETF数据处理完成，成功: {success_count}/{len(etf_list)}")
        
        logger.info("所有数据处理完成！")
        
    finally:
        # 登出baostock
        logout_bs()

if __name__ == "__main__":
    main()
