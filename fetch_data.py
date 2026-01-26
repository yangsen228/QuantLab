#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版A股数据获取脚本
获取股票、指数、ETF的量价、基本面、行业等数据
支持增量更新模式
"""

import baostock as bs
import pandas as pd
from datetime import datetime, timedelta
import os
import time
from tqdm import tqdm
from typing import Optional, Dict, List
import argparse

# 配置参数
DATA_DIR = "data"
DEFAULT_START_DATE = "2026-01-26"
END_DATE = datetime.now().strftime("%Y-%m-%d")
REQUEST_DELAY = 0.1  # 请求间隔时间(秒)

# 数据类型配置
DATA_TYPES = {
    'price_volume': {
        'folder': 'price_volume',
        'fields': "date,code,open,high,low,close,preclose,volume,amount,pctChg,adjustflag,turn,tradestatus,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST",
        'frequency': 'd',
        'adjustflag': '1'  # 复权状态（1：后复权，2：前复权，3：不复权）
    }
}

class StockDataFetcher:
    """股票数据获取器"""
    
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.login_status = False
        
    def login(self) -> bool:
        """登录baostock"""
        lg = bs.login()
        if lg.error_code != '0':
            print(f"登录失败: {lg.error_msg}")
            return False
        self.login_status = True
        return True
    
    def logout(self):
        """登出baostock"""
        if self.login_status:
            bs.logout()
            self.login_status = False
    
    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        print("获取股票列表...")
        rs = bs.query_stock_basic()
        stocks = []
        while rs.next():
            stocks.append(rs.get_row_data())
        
        if not stocks:
            print("未获取到股票列表")
            return pd.DataFrame()
            
        df = pd.DataFrame(stocks, columns=rs.fields)
        # 过滤A股（上证和深证）
        df = df[df['code'].str.startswith(('sh.', 'sz.'))]
        print(f"共{len(df)}只股票")
        return df
    
    def get_last_date_from_file(self, filepath: str) -> Optional[str]:
        """从现有文件获取最后日期"""
        if not os.path.exists(filepath):
            return None
        
        try:
            df = pd.read_csv(filepath)
            if 'date' in df.columns and not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                last_date = df['date'].max()
                # 返回下一天作为开始日期
                next_date = last_date + timedelta(days=1)
                return next_date.strftime('%Y-%m-%d')
        except Exception as e:
            print(f"读取文件失败 {filepath}: {e}")
        
        return None
    
    def fetch_data_for_stock(self, code: str, name: str, data_type: str, 
                           start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取单只股票指定类型的数据"""
        config = DATA_TYPES[data_type]
        
        if data_type == 'basic':
            # 基本面数据使用不同的API
            rs = bs.query_stock_basic(code=code)
        elif data_type == 'industry':
            # 行业数据
            rs = bs.query_stock_industry(code=code)
        else:
            # 量价数据
            rs = bs.query_history_k_data_plus(
                code=code,
                fields=config['fields'],
                start_date=start_date,
                end_date=end_date,
                frequency=config['frequency'],
                adjustflag=config['adjustflag']
            )
        
        if rs.error_code != '0':
            print(f"获取{code}的{data_type}数据失败: {rs.error_msg}")
            return None
        
        data = []
        while rs.next():
            data.append(rs.get_row_data())
        
        if not data:
            return None
        
        df = pd.DataFrame(data, columns=rs.fields)
        
        # 数据类型转换
        if data_type == 'price_volume':
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'pctChg', 'turn']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def save_data(self, df: pd.DataFrame, filepath: str):
        """保存数据到文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if os.path.exists(filepath):
            # 增量更新模式
            try:
                existing_df = pd.read_csv(filepath)
                if 'date' in existing_df.columns:
                    existing_df['date'] = pd.to_datetime(existing_df['date'])
                
                # 合并数据
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                
                # 按日期去重，保留最新数据
                if 'date' in combined_df.columns:
                    combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
                    combined_df = combined_df.sort_values('date')
                
                combined_df.to_csv(filepath, index=False, encoding='utf-8-sig')
            except Exception as e:
                print(f"增量更新失败 {filepath}: {e}，将覆盖保存")
                df.to_csv(filepath, index=False, encoding='utf-8-sig')
        else:
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
    
    def fetch_all_data(self, start_date: str = DEFAULT_START_DATE, end_date: str = END_DATE):
        """获取所有股票的所有类型数据"""
        print(f"开始获取A股数据，时间范围: {start_date} 至 {end_date}")
        
        stock_df = self.get_stock_list()
        if stock_df.empty:
            return
        
        total_stats = {}
        
        for data_type in DATA_TYPES.keys():
            print(f"\n获取{data_type}数据...")
            folder = DATA_TYPES[data_type]['folder']
            stats = {'success': 0, 'updated': 0, 'new': 0, 'failed': 0}
            
            for _, row in tqdm(stock_df.iterrows(), total=len(stock_df), desc=f"获取{data_type}"):
                code = row['code']
                name = row['code_name'].replace(' ', '_').replace('*', '')
                
                # 构建文件名
                filename = f"{code.replace('.', '_')}_{name}.csv"
                filepath = os.path.join(self.data_dir, folder, filename)
                print(filepath)
                
                # 确定起始日期
                file_start_date = self.get_last_date_from_file(filepath)
                if file_start_date:
                    # 检查是否已是最新数据
                    if file_start_date > end_date:
                        continue
                    actual_start_date = file_start_date
                    stats['updated'] += 1
                else:
                    actual_start_date = start_date
                    stats['new'] += 1
                
                # 获取数据
                df = self.fetch_data_for_stock(code, name, data_type, actual_start_date, end_date)
                
                if df is not None and not df.empty:
                    self.save_data(df, filepath)
                    stats['success'] += 1
                else:
                    stats['failed'] += 1
                
                time.sleep(REQUEST_DELAY)
            
            total_stats[data_type] = stats
            print(f"{data_type}数据获取完成: 成功{stats['success']}, 新增{stats['new']}, 更新{stats['updated']}, 失败{stats['failed']}")
        
        return total_stats

def main(start_date):
    """主函数"""
    print(f"使用开始日期: {start_date}")
    
    fetcher = StockDataFetcher()
    
    try:
        # 登录
        if not fetcher.login():
            return
        
        # 获取所有数据
        stats = fetcher.fetch_all_data(start_date=start_date)
        
        print("\n数据获取全部完成！")
        for data_type, stat in stats.items():
            print(f"{data_type}: 成功{stat['success']}, 新增{stat['new']}, 更新{stat['updated']}, 失败{stat['failed']}")
        
    finally:
        fetcher.logout()

if __name__ == "__main__":
    while True:
        now = datetime.now()
        # 检查是否到了下午7点 (19:00)
        if now.hour == 19 and now.minute == 0:
            # 使用当天日期作为start_date
            start_date = now.strftime("%Y-%m-%d")
            print(f"执行数据获取任务，开始日期: {start_date}", flush=True)
            main(start_date)
            # 等待一分钟，避免重复执行
            time.sleep(60)
        else:
            # 计算到下一个19:00的等待时间
            next_run = now.replace(hour=19, minute=0, second=0, microsecond=0)
            if now.hour >= 19:
                # 如果已经过了今天的19:00，等待到明天的19:00
                next_run += timedelta(days=1)
            
            wait_seconds = (next_run - now).total_seconds()
            print(f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}，等待下次执行时间: {next_run.strftime('%Y-%m-%d %H:%M:%S')}, 还需等待 {int(wait_seconds)} 秒", flush=True)
            time.sleep(min(wait_seconds, 600))  # 最多等待60秒，然后重新检查