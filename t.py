#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 14:19:25 2025

@author: wuzhixiang

"""
import nest_asyncio
nest_asyncio.apply()
from collections import defaultdict
import traceback
import asyncio
import enum
from typing import Optional, Dict, Set, Any, Callable, Union
import uuid
import weakref
import multiprocessing as mp
import pickle
import functools
from concurrent.futures import ProcessPoolExecutor, as_completed


class Debugger:
    def __init__(self,f):
        self.func=f
    def __call__(self,*args,**kwargs):
        try:
            return self.func(*args,**kwargs)
        except Exception as e:
            print(f"Error in {self.func.__name__}: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return None




# 多进程工作函数（定义在类外，便于序列化）
def _mp_filter_rows(data_chunk, func_code, func_globals):
    """多进程工作函数：筛选行数据"""
    try:
        # 重建函数对象
        exec(func_code, func_globals)
        func = func_globals['__temp_func__']
        
        result = []
        for row_data in data_chunk:
            if func(row_data):
                result.append(row_data)
        return result
    except Exception as e:
        # 返回错误信息而不是抛出异常
        return {'error': str(e)}

def _mp_filter_cols(data_chunk, func_code, func_globals, keys):
    """多进程工作函数：筛选列数据"""
    try:
        # 重建函数对象
        exec(func_code, func_globals)
        func = func_globals['__temp_func__']
        
        result = []
        for key in keys:
            if key in data_chunk:
                if func({key: data_chunk[key]}):
                    result.append({key: data_chunk[key]})
        return result
    except Exception as e:
        return {'error': str(e)}


class Table:
    """
    用于管理结构化表格数据的类，支持行和列的增删改查，以及数据查询、排序、分组和聚合等操作。
    """
    # __init__特殊方法，别包装
    def __init__(self,*keys):
        """
        初始化表格，指定列名
        
        Parameters
        ----------
        *keys : str
            表格的列名列表，可变参数
        
        Returns
        -------
        None
        """
        self.rows=[]
        self.keys=list(keys)
        parent=self
        class Row:
            def __init__(self,key):
                self.key=key
                self.row=dict(zip(parent.keys,[None for i in parent.keys]))
                
            def __setitem__(self, name, value):
                try:self.row[name]=value
                except KeyError as e: raise e
                
            def __getitem__(self,name):
                try:return self.row[name]
                except KeyError as e: raise e
            
            def __delitem__(self,name):
                self.row[name]=None

            def __repr__(self):
                return f'Row({self.row})'
        self.rowclass=Row
    
    def _serialize_function(self, func):
        """序列化函数以便在多进程间传递"""
        try:
            # 获取函数源码
            import inspect
            func_code = inspect.getsource(func)
            
            # 创建临时的全局命名空间
            func_globals = {
                '__temp_func__': func,
                '__builtins__': __builtins__,
                # 添加可能需要的模块
                'defaultdict': defaultdict,
            }
            
            # 将函数重新命名为临时名称
            func_code = func_code.replace(func.__name__, '__temp_func__', 1)
            
            return func_code, func_globals
        except Exception:
            # 如果无法获取源码，尝试使用pickle
            try:
                return pickle.dumps(func), None
            except Exception:
                raise ValueError("Function cannot be serialized for multiprocessing")
    
    def _chunk_data(self, data, num_chunks):
        """将数据分割成多个块"""
        chunk_size = max(1, len(data) // num_chunks)
        chunks = []
        for i in range(0, len(data), chunk_size):
            chunks.append(data[i:i + chunk_size])
        return chunks
    
    def _multiprocess_filter(self, func, by='row', max_workers=None):
        """使用多进程进行筛选"""
        if max_workers is None:
            max_workers = min(4, mp.cpu_count())
        
        # 序列化函数
        try:
            func_code, func_globals = self._serialize_function(func)
        except ValueError as e:
            print(f"Multiprocessing fallback to single process: {e}")
            return self._single_process_filter(func, by)
        
        # 准备数据
        if by == 'row':
            data = [row.row for row in self.rows]
            chunks = self._chunk_data(data, max_workers)
            
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    for chunk in chunks:
                        future = executor.submit(_mp_filter_rows, chunk, func_code, func_globals)
                        futures.append(future)
                    
                    results = []
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            if isinstance(result, dict) and 'error' in result:
                                print(f"Process error: {result['error']}")
                                # 降级到单进程处理
                                return self._single_process_filter(func, by)
                            results.extend(result)
                        except Exception as e:
                            print(f"Process execution error: {e}")
                            return self._single_process_filter(func, by)
                    
                    return results
            except Exception as e:
                print(f"ProcessPoolExecutor error: {e}")
                return self._single_process_filter(func, by)
                
        elif by == 'col':
            # 准备列数据
            col = dict(zip(self.keys, [[] for _ in self.keys]))
            for row in self.rows:
                for k, v in row.row.items():
                    col[k].append(v)
            
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    future = executor.submit(_mp_filter_cols, col, func_code, func_globals, self.keys)
                    result = future.result()
                    
                    if isinstance(result, dict) and 'error' in result:
                        print(f"Process error: {result['error']}")
                        return self._single_process_filter(func, by)
                    
                    return result
            except Exception as e:
                print(f"ProcessPoolExecutor error: {e}")
                return self._single_process_filter(func, by)
        
        return []
    
    def _single_process_filter(self, func, by='row'):
        """单进程筛选（原始逻辑）"""
        r = []
        
        if by == 'row':
            for row in self.rows:
                if func(row.row):
                    r.append(row.row)
        elif by == 'col':
            col = dict(zip(self.keys, [[] for _ in self.keys]))
            for row in self.rows:
                for k, v in row.row.items():
                    col[k].append(v)
            for key in self.keys:
                if func({key: col[key]}):
                    r.append({key: col[key]})
        
        return r
    
    def row(self,ik):
        """
        通过索引或行键获取行对象
        
        Parameters
        ----------
        ik : int or str
            若为int，按索引获取行；若为str，按行键匹配获取行
        
        Returns
        -------
        Row or None
            匹配的行对象，无匹配则返回None
        """
        if isinstance(ik,str):
            for i in self.rows:
                if i.key==ik:
                    return i
            return None
        elif isinstance(ik,int):
            try:return self.rows[ik]
            except KeyError:return None
        
    def col(self,k):
        """
        获取指定列的所有值
        
        Parameters
        ----------
        k : str
            列名
        
        Returns
        -------
        list
            该列所有行的值组成的列表
        """
        return list(map(lambda x:x.row[k],self.rows))

    def addrow(self,key,row=None,ik=-1):
        """
        向表格添加行
        
        Parameters
        ----------
        key : str
            行的唯一标识键
        row : dict, optional
            行数据字典，键为列名，值为对应数据，默认为None（空行）
        ik : int or str, optional
            插入位置，int为索引，str为行键（插入到该键对应行前），默认为-1（末尾）
        
        Returns
        -------
        None
        """
        rowc=self.rowclass(key)
        if row:rowc.row=row 
        if isinstance(ik,int):
            self.rows.insert(ik,rowc)
        elif isinstance(ik,str):
            for i,row in enumerate(self.rows):
                if row.key==ik:
                    self.rows.insert(i,rowc)

    def set(self,ik,col,value):
        """
        设置指定单元格的值
        
        Parameters
        ----------
        ik : int or str
            行索引（int）或行键（str）
        col : str
            列名
        value : any
            要设置的单元格值
        
        Returns
        -------
        None
        """
        self.row(ik)[col]=value
    
    def addcol(self,colname,col=None):
        """
        向表格添加列
        
        Parameters
        ----------
        colname : str
            新列的列名
        col : list, optional
            新列各单元格的值列表，长度应与行数一致，默认为None（全为None）
        
        Returns
        -------
        None
        """
        if col==None:
            col=[None for i in range(len(self.rows))]
        for i,row in enumerate(self.rows):
            row[colname]=col[i]
        self.keys.append(colname)
    
    def delcol(self,colname):
        """
        删除指定列
        
        Parameters
        ----------
        colname : str
            要删除的列名
        
        Returns
        -------
        None
        """
        self.keys.remove(colname)
        for i,row in enumerate(self.rows):
            row.row.pop(colname)
        
    def delrow(self,ik):
        """
        删除指定行
        
        Parameters
        ----------
        ik : int or str
            若为int，按索引删除行；若为str，按行键匹配删除行
        
        Returns
        -------
        None
        """
        if isinstance(ik,int):
            self.rows.pop(ik)
        elif isinstance(ik,str):
            for i,row in enumerate(self.rows):
                if row.key==ik:
                    self.rows.pop(i)

    def _option_aggregate(self,r,aggregate):
        """
        对数据进行聚合操作
        
        Parameters
        ----------
        r : list
            待聚合的数据列表
        aggregate : str or callable
            聚合方式，内置支持'sum'/'avg'/'max'/'min'/'count'/'count_distinct'，或自定义聚合函数
        
        Returns
        -------
        any
            聚合结果
        """
        if callable(aggregate):
                return aggregate(r)
        elif isinstance(aggregate, str):
            if aggregate == 'count':
                    return len(r)
            elif aggregate == 'count_distinct':
                    return len(set(r)) 
            
            # 对于数值聚合，需要提取数值
            if not r:
                return 0
            
            # 如果r包含字典，尝试提取数值
            values = []
            for item in r:
                if isinstance(item, dict):
                    # 如果是字典，提取所有数值
                    for v in item.values():
                        if isinstance(v, (int, float)):
                            values.append(v)
                elif isinstance(item, (int, float)):
                    values.append(item)
            
            # 如果没有找到数值，回退到原始数据
            if not values:
                values = r
            
            if aggregate == 'sum':
                return sum(values)
            elif aggregate == 'avg':
                return sum(values) / len(values) if values else 0
            elif aggregate == 'max':
                return max(values) if values else None
            elif aggregate == 'min':
                return min(values) if values else None
    
    def _option_sort(self,r,sort):
        """
        对数据进行排序操作

        Parameters
        ----------
        r : list
            待排序的数据列表
        sort : str or callable
            排序方式，内置支持'sum'/'avg'/'max'/'min'/'count'/'count_distinct'，或自定义排序函数

        Returns
        -------
        list
            排序后的数据列表
        """
        if callable(sort):
            return sorted(r,key=sort)
        else:
            return sorted(r)
    
    def _option_offset(self,r,offset):
        """
        对数据进行偏移和限制处理
        
        Parameters
        ----------
        r : list
            原始数据
        offset : dict
            包含'offset'（起始位置）和'limit'（数量限制）的字典
        
        Returns
        -------
        list
            处理后的数据列表
        """
        limit = offset.get('limit', None)
        offset_val = offset.get('offset', 0)
        
        if limit is None:
            # 如果没有限制，取到列表末尾
            return r[offset_val:]
        else:
            # 有限制时，取指定范围
            return r[offset_val:offset_val + limit]

    def _option_group_by(self,r,group_by):
        """
        对数据进行分组操作

        Parameters
        ----------
        r : list
            原始数据
        group_by : str
            分组依据的列名

        Returns
        -------
        dict
            分组后的数据字典，键为分组依据的值，值为分组后的数据列表

        """
        group=defaultdict(list)
        for i, row in enumerate(r):
            if isinstance(row, dict):
                # 如果是dict对象（来自query方法）
                group_key = row.get(group_by)
                group[(f'row_{i}', group_key)].append(row)
            else:
                # 如果是Row对象（原始实现）
                tpl=(row.key, row.row[group_by])
                group[tpl].append(row.row)
        return group

    def _option_select(self, r, select):
        """
        选择要返回的列（投影操作）
        
        Parameters
        ----------
        r : list
            原始数据
        select : list or str
            要选择的列名列表，或'*'表示所有列
            
        Returns
        -------
        list
            投影后的数据列表
        """
        if select == '*':
            return r
        if isinstance(select, str):
            select = [select]
        
        result = []
        for row in r:
            if isinstance(row, dict):
                projected_row = {col: row.get(col) for col in select if col in row}
                result.append(projected_row)
        return result
    
    def _option_distinct(self, r, distinct=True):
        """
        去除重复记录
        
        Parameters
        ----------
        r : list
            原始数据
        distinct : bool or list
            True表示对所有列去重，list表示对指定列去重
            
        Returns
        -------
        list
            去重后的数据列表
        """
        if not distinct:
            return r
            
        seen = set()
        result = []
        
        for row in r:
            if isinstance(row, dict):
                if isinstance(distinct, list):
                    # 对指定列去重
                    key = tuple(row.get(col) for col in distinct if col in row)
                else:
                    # 对所有列去重
                    key = tuple(sorted(row.items()))
                
                if key not in seen:
                    seen.add(key)
                    result.append(row)
        return result
    
    def _option_order_by(self, r, order_by):
        """
        按指定列排序
        
        Parameters
        ----------
        r : list
            原始数据
        order_by : dict or list
            排序规则，格式：{'column': 'asc/desc'} 或 [('column', 'asc')]
            
        Returns
        -------
        list
            排序后的数据列表
        """
        if isinstance(order_by, dict):
            order_by = list(order_by.items())
        
        for col, direction in reversed(order_by):
            reverse = direction.lower() == 'desc'
            r = sorted(r, key=lambda x: x.get(col, ''), reverse=reverse)
        
        return r
    
    def _option_having(self, r, having_func):
        """
        对分组后的数据进行筛选
        
        Parameters
        ----------
        r : dict or list
            分组后的数据
        having_func : callable
            筛选函数
            
        Returns
        -------
        dict or list
            筛选后的数据
        """
        if isinstance(r, dict):
            # 对分组数据进行筛选
            return {k: v for k, v in r.items() if having_func(v)}
        else:
            # 对普通列表数据进行筛选
            return [row for row in r if having_func(row)]
    
    def _option_join(self, r, join_config):
        """
        与其他表进行连接操作
        
        Parameters
        ----------
        r : list
            当前表数据
        join_config : dict
            连接配置: {
                'table': other_table,
                'type': 'inner/left/right/full',
                'on': ('key1', 'key2') or lambda
            }
            
        Returns
        -------
        list
            连接后的数据列表
        """
        other_table = join_config['table']
        join_type = join_config.get('type', 'inner')
        join_condition = join_config['on']
        
        result = []
        
        if callable(join_condition):
            # 使用函数进行连接
            for row1 in r:
                matched = False
                for row2 in other_table.rows:
                    if join_condition(row1, row2.row):
                        merged = {**row1, **row2.row}
                        result.append(merged)
                        matched = True
                
                if not matched and join_type in ['left', 'full']:
                    # 左连接或全连接时，保留未匹配的左表数据
                    merged = {**row1}
                    for key in other_table.keys:
                        merged[key] = None
                    result.append(merged)
        
        elif isinstance(join_condition, tuple) and len(join_condition) == 2:
            # 使用键进行连接
            left_key, right_key = join_condition
            
            # 创建右表索引
            right_index = {}
            for row in other_table.rows:
                key_val = row.row.get(right_key)
                if key_val not in right_index:
                    right_index[key_val] = []
                right_index[key_val].append(row.row)
            
            for row1 in r:
                left_val = row1.get(left_key)
                matched = False
                
                if left_val in right_index:
                    for row2 in right_index[left_val]:
                        merged = {**row1, **row2}
                        result.append(merged)
                        matched = True
                
                if not matched and join_type in ['left', 'full']:
                    merged = {**row1}
                    for key in other_table.keys:
                        merged[key] = None
                    result.append(merged)
        
        return result
    
    def _option_pivot(self, r, pivot_config):
        """
        数据透视操作
        
        Parameters
        ----------
        r : list
            原始数据
        pivot_config : dict
            透视配置: {
                'index': 'column_name',      # 行索引列
                'columns': 'column_name',    # 列索引列  
                'values': 'column_name',     # 值列
                'aggfunc': 'sum/avg/count'   # 聚合函数
            }
            
        Returns
        -------
        dict
            透视后的数据
        """
        index_col = pivot_config['index']
        column_col = pivot_config['columns']
        value_col = pivot_config['values']
        aggfunc = pivot_config.get('aggfunc', 'sum')
        
        # 收集所有唯一的列值
        unique_columns = set()
        for row in r:
            unique_columns.add(row.get(column_col))
        
        # 构建透视表
        pivot_data = {}
        for row in r:
            idx = row.get(index_col)
            col = row.get(column_col)
            val = row.get(value_col)
            
            if idx not in pivot_data:
                pivot_data[idx] = {c: [] for c in unique_columns}
            
            pivot_data[idx][col].append(val)
        
        # 聚合数据
        result = {}
        for idx, cols in pivot_data.items():
            result[idx] = {}
            for col, values in cols.items():
                if values:
                    if aggfunc == 'sum':
                        result[idx][col] = sum(values)
                    elif aggfunc == 'avg':
                        result[idx][col] = sum(values) / len(values)
                    elif aggfunc == 'count':
                        result[idx][col] = len(values)
                    else:
                        result[idx][col] = values[0]
                else:
                    result[idx][col] = None
        
        return result
    
    def _option_window_function(self, r, window_config):
        """
        窗口函数操作
        
        Parameters
        ----------
        r : list
            原始数据
        window_config : dict
            窗口配置: {
                'func': 'row_number/rank/dense_rank/lag/lead',
                'partition_by': 'column_name',  # 分区列
                'order_by': 'column_name',      # 排序列
                'lag_lead_offset': int          # lag/lead偏移量
            }
            
        Returns
        -------
        list
            添加窗口函数结果的数据列表
        """
        func = window_config['func']
        partition_by = window_config.get('partition_by')
        order_by = window_config.get('order_by')
        offset = window_config.get('lag_lead_offset', 1)
        
        # 按分区分组
        if partition_by:
            partitions = defaultdict(list)
            for row in r:
                key = row.get(partition_by)
                partitions[key].append(row)
        else:
            partitions = {'all': r}
        
        result = []
        
        for partition_key, partition_data in partitions.items():
            # 在分区内排序
            if order_by:
                partition_data = sorted(partition_data, key=lambda x: x.get(order_by, ''))
            
            # 应用窗口函数
            for i, row in enumerate(partition_data):
                new_row = row.copy()
                
                if func == 'row_number':
                    new_row[f'{func}'] = i + 1
                elif func == 'rank':
                    # 简化的rank实现
                    new_row[f'{func}'] = i + 1
                elif func == 'dense_rank':
                    # 简化的dense_rank实现
                    new_row[f'{func}'] = i + 1
                elif func == 'lag':
                    if i >= offset:
                        new_row[f'{func}'] = partition_data[i - offset].get(order_by)
                    else:
                        new_row[f'{func}'] = None
                elif func == 'lead':
                    if i < len(partition_data) - offset:
                        new_row[f'{func}'] = partition_data[i + offset].get(order_by)
                    else:
                        new_row[f'{func}'] = None
                
                result.append(new_row)
        
        return result

    def query(self,func,**options):
        """
        对表格数据进行查询筛选（增强版）
        
        Parameters
        ----------
        func : callable
            筛选函数，接收行/列数据并返回bool，决定是否保留
        **options : dict
            查询选项，包括：
            - 'limit': 结果数量限制
            - 'by': 查询对象（'row'或'col'）
            - 'aggregate': 聚合方式
            - 'sort': 排序规则（已废弃，使用order_by）
            - 'offset': 偏移设置
            - 'group_by': 分组列
            - 'select': 选择列（投影）
            - 'distinct': 去重（True或列名列表）
            - 'order_by': 排序规则字典或列表
            - 'having': 分组后筛选函数
            - 'join': 表连接配置
            - 'pivot': 数据透视配置
            - 'window': 窗口函数配置
            - 'case_sensitive': 字符串比较是否区分大小写
            - 'task': 启用多进程处理，可以是True（使用默认设置）或字典配置
                     字典格式: {'enabled': True, 'max_workers': 4, 'chunk_size': 1000}
        
        Returns
        -------
        list or dict or any
            查询处理后的结果
        """
        if not callable(func):
            return
        r=[]

        # 现有选项
        limit=options.get('limit',float('inf')) 
        by=options.get('by','row')
        aggregate=options.get('aggregate',False)
        sort=options.get('sort',False)  # 保持向后兼容
        offset=options.get('offset',{"offset":0})
        group_by=options.get('group_by')

        # 新增选项
        select=options.get('select', '*')
        distinct=options.get('distinct', False)
        order_by=options.get('order_by', False)
        having=options.get('having', False)
        join=options.get('join', False)
        pivot=options.get('pivot', False)
        window=options.get('window', False)
        case_sensitive=options.get('case_sensitive', True)
        task=options.get('task', False)  # 多进程选项

        # 基础查询 - 支持多进程
        use_multiprocessing = False
        max_workers = None
        
        # 解析task参数
        if task:
            if isinstance(task, bool):
                use_multiprocessing = True
                max_workers = None  # 使用默认
            elif isinstance(task, dict):
                use_multiprocessing = task.get('enabled', True)
                max_workers = task.get('max_workers', None)
            elif isinstance(task, int):
                use_multiprocessing = True
                max_workers = task
        
        # 决定使用单进程还是多进程
        if use_multiprocessing and len(self.rows) > 100:  # 只有数据量大时才使用多进程
            try:
                r = self._multiprocess_filter(func, by, max_workers)
            except Exception as e:
                print(f"Multiprocessing failed, fallback to single process: {e}")
                r = self._single_process_filter(func, by)
        else:
            # 原始单进程逻辑
            if by=='row':
                for row in self.rows:
                    if func(row.row):
                        r.append(row.row)
                        if len(r)>=limit:break
                
            elif by=='col':
                col=dict(zip(self.keys,[[] for i in self.keys]))
                for row in self.rows:
                    for k,v in row.row.items():
                        col[k].append(v)
                for key in self.keys:
                    if func({key:col[key]}):
                        r.append({key:col[key]})
                        if len(r)>=limit:break
        
        # 应用limit（如果使用了多进程，需要在这里应用limit）
        if use_multiprocessing and len(r) > limit:
            r = r[:int(limit)]
        
        # 应用各种选项（按执行顺序）
        
        # 1. 表连接
        if join:
            r = self._option_join(r, join)
        
        # 2. 投影（选择列）
        if select != '*':
            r = self._option_select(r, select)
        
        # 3. 去重
        if distinct:
            r = self._option_distinct(r, distinct)
        
        # 4. 排序（优先使用order_by，然后是sort）
        if order_by:
            r = self._option_order_by(r, order_by)
        elif sort:
            r = self._option_sort(r, sort)
        
        # 5. 分组
        if group_by:
            r = self._option_group_by(r, group_by)
        
        # 6. 分组后筛选
        if having:
            r = self._option_having(r, having)
        
        # 7. 窗口函数
        if window:
            r = self._option_window_function(r, window)
        
        # 8. 数据透视
        if pivot:
            r = self._option_pivot(r, pivot)
        
        # 9. 偏移和限制
        if offset and not group_by and not pivot:  # 分组和透视后不适用offset
            r = self._option_offset(r, offset)
        
        # 10. 聚合（最后执行）
        if aggregate:
            r = self._option_aggregate(r, aggregate)
        
        return r
    
    def __repr__(self):
        """以可视化表格形式返回字符串表示"""
        if not self.rows:
            return "Empty Table(columns: {})".format(self.keys)
        
        # 计算每列最大宽度
        col_widths = {key: len(str(key)) for key in self.keys}
        for row in self.rows:
            for key, value in row.row.items():
                col_widths[key] = max(col_widths[key], len(str(value)))
        
        # 构建分隔线
        separator = "+".join(["-" * (width + 2) for width in col_widths.values()])
        separator = f"+{separator}+"
        
        # 构建表头
        header = "|".join([f" {key:^{col_widths[key]}} " for key in self.keys])
        header = f"|{header}|"
        
        # 构建行数据
        rows_str = []
        for row in self.rows:
            row_data = "|".join([f" {str(row.row[key]):{col_widths[key]}} " for key in self.keys])
            rows_str.append(f"|{row_data}|")
        
        # 组合所有部分
        return "\n".join([separator, header, separator] + rows_str + [separator])

            


# 信我这是b+树
class Node:
    def __init__(self, is_leaf=False, val_keys=[], max_size=4):
        self.is_leaf = is_leaf
        self.max_size = max_size
        self.parent = None
        
        if is_leaf:
            self.val = Table(*val_keys)
            self.next = None  # 叶子节点的链表指针
        else:
            self.keys = []  # 内部节点的键
            self.children = []  # 内部节点的子节点列表
    
    def is_full(self):
        """检查节点是否已满"""
        if self.is_leaf:
            return len(self.val.rows) >= self.max_size
        else:
            return len(self.keys) >= self.max_size
    
    def split_leaf(self):
        """分裂叶子节点"""
        if not self.is_leaf:
            raise ValueError("只能分裂叶子节点")
            
        # 创建新的叶子节点
        new_node = Node(is_leaf=True, val_keys=self.val.keys, max_size=self.max_size)
        
        # 分割数据：前一半留在当前节点，后一半移到新节点
        mid = len(self.val.rows) // 2
        
        # 将后一半数据移到新节点
        for i in range(mid, len(self.val.rows)):
            row = self.val.rows[i]
            new_node.val.addrow(row.key, row.row, -1)
        
        # 从当前节点删除后一半数据
        self.val.rows = self.val.rows[:mid]
        
        # 更新链表指针
        new_node.next = self.next
        self.next = new_node
        new_node.parent = self.parent
        
        # 返回新节点的第一个键（用于父节点）
        return new_node.val.rows[0].key if new_node.val.rows else None, new_node
    
    def split_internal(self):
        """分裂内部节点"""
        if self.is_leaf:
            raise ValueError("只能分裂内部节点")
            
        # 创建新的内部节点
        new_node = Node(is_leaf=False, max_size=self.max_size)
        
        mid = len(self.keys) // 2
        
        # 分割键和子节点
        promote_key = self.keys[mid]
        new_node.keys = self.keys[mid + 1:]
        new_node.children = self.children[mid + 1:]
        
        self.keys = self.keys[:mid]
        self.children = self.children[:mid + 1]
        
        # 更新子节点的父指针
        for child in new_node.children:
            child.parent = new_node
        
        new_node.parent = self.parent
        
        return promote_key, new_node

class TaskStatus(enum.Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class AsyncTask:
    """异步任务封装类"""
    
    def __init__(self, coro, name: Optional[str] = None, team: Optional['AsyncTaskTeam'] = None, auto_start: bool = True):
        self.id = uuid.uuid4()
        self.name = name or f"Task-{self.id.hex[:8]}"
        self.coro = coro
        self.team = weakref.ref(team) if team else None
        self.status = TaskStatus.PENDING
        self.created_at = None  # 将在start时设置
        self.started_at = None
        self.finished_at = None
        self.result_value = None
        self.exception_value = None
        self.task = None  # 延迟创建
        self.auto_start = auto_start
        
        if auto_start:
            self.start()
    
    def start(self) -> bool:
        """启动任务"""
        if self.task is not None:
            # 任务已经启动
            return False
        
        try:
            # 检查事件循环是否存在
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 如果没有运行中的事件循环，尝试获取或创建一个
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        
        self.task = asyncio.create_task(self.coro, name=self.name)
        self.status = TaskStatus.RUNNING
        self.created_at = loop.time()
        self.started_at = loop.time()
        
        # 添加任务完成回调
        self.task.add_done_callback(self._on_task_done)
        return True
    
    def _on_task_done(self, task: asyncio.Task):
        """任务完成时的回调"""
        try:
            loop = asyncio.get_running_loop()
            self.finished_at = loop.time()
        except RuntimeError:
            pass
            
        try:
            if task.cancelled():
                self.status = TaskStatus.CANCELLED
            elif task.exception():
                self.status = TaskStatus.FAILED
                self.exception_value = task.exception()
            else:
                self.status = TaskStatus.COMPLETED
                self.result_value = task.result()
        except Exception as e:
            self.status = TaskStatus.FAILED
            self.exception_value = e
        
        # 通知team任务完成
        if self.team and self.team():
            self.team()._on_task_done(self.id)
    
    def cancel(self, msg: str = "Task cancelled") -> bool:
        """取消任务"""
        if self.task and not self.task.done():
            self.status = TaskStatus.CANCELLED
            return self.task.cancel()
        elif not self.task:
            # 如果任务还没有启动，直接标记为取消
            self.status = TaskStatus.CANCELLED
            return True
        return False
    
    def cancelled(self) -> bool:
        """检查任务是否被取消"""
        return self.status == TaskStatus.CANCELLED or (self.task and self.task.cancelled())
    
    def done(self) -> bool:
        """检查任务是否完成"""
        if not self.task:
            return self.status in [TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.FAILED]
        return self.task.done()
    
    def started(self) -> bool:
        """检查任务是否已启动"""
        return self.task is not None
    
    def result(self, timeout: Optional[float] = None):
        """获取任务结果（阻塞）"""
        if not self.task:
            if self.status == TaskStatus.PENDING:
                raise RuntimeError("Task not started yet, call start() first")
            elif self.status == TaskStatus.CANCELLED:
                raise asyncio.CancelledError("Task was cancelled")
            elif self.status == TaskStatus.FAILED:
                raise self.exception_value
            else:
                return self.result_value
        
        if timeout:
            try:
                return asyncio.wait_for(self.task, timeout=timeout)
            except asyncio.TimeoutError:
                raise asyncio.TimeoutError(f"Task {self.name} timed out after {timeout}s")
        else:
            return self.task.result()
    
    async def wait(self):
        """异步等待任务完成"""
        if not self.task:
            if self.status == TaskStatus.PENDING:
                raise RuntimeError("Task not started yet, call start() first")
            elif self.status == TaskStatus.CANCELLED:
                raise asyncio.CancelledError("Task was cancelled")
            elif self.status == TaskStatus.FAILED:
                raise self.exception_value
            else:
                return self.result_value
        
        return await self.task
    
    def exception(self):
        """获取任务异常"""
        return self.task.exception() if self.task else self.exception_value
    
    @property 
    def elapsed_time(self) -> float:
        """获取任务运行时间"""
        if not self.started_at:
            return 0.0
        
        try:
            current_time = asyncio.get_running_loop().time()
        except RuntimeError:
            import time
            current_time = time.time()
            
        if self.finished_at:
            return self.finished_at - self.started_at
        else:
            return current_time - self.started_at
    
    @property
    def execution_time(self) -> Optional[float]:
        """获取任务执行时间（从开始到完成）"""
        if self.started_at and self.finished_at:
            return self.finished_at - self.started_at
        return None
    
    def __repr__(self):
        status_info = f"status={self.status.value}"
        if self.started():
            elapsed = f", elapsed={self.elapsed_time:.2f}s"
        else:
            elapsed = ", not_started"
        
        return f"AsyncTask(id={self.id.hex[:8]}, name={self.name}, {status_info}{elapsed})"
    
    def __await__(self):
        """支持await语法"""
        if not self.task:
            # 如果任务没有启动，自动启动它
            if self.status == TaskStatus.PENDING:
                self.start()
            if not self.task:
                raise RuntimeError("Failed to start task")
        
        return self.task.__await__()


class AsyncTaskTeam:
    """异步任务团队管理类"""
    
    def __init__(self, name: str, max_concurrent: Optional[int] = None):
        self.name = name
        self.max_concurrent = max_concurrent
        self.tasks: Dict[uuid.UUID, AsyncTask] = {}
        self.completed_tasks: Dict[uuid.UUID, AsyncTask] = {}
        self.semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent else None
        self.created_at = None
        self.is_running = False
        
        try:
            loop = asyncio.get_running_loop()
            self.created_at = loop.time()
        except RuntimeError:
            pass
    
    def add_task(self, coro, name: Optional[str] = None, auto_start: bool = False) -> AsyncTask:
        """添加任务到团队"""
        if self.max_concurrent and len(self.tasks) >= self.max_concurrent:
            raise RuntimeError(f"Team {self.name} has reached max concurrent limit: {self.max_concurrent}")
        
        task = AsyncTask(coro, name, team=self, auto_start=auto_start)
        self.tasks[task.id] = task
        
        if auto_start:
            task.status = TaskStatus.RUNNING
        
        return task
    
    def add_task_lazy(self, coro, name: Optional[str] = None) -> AsyncTask:
        """添加任务到团队（延迟执行）"""
        return self.add_task(coro, name, auto_start=False)
    
    def start_task(self, task_id: uuid.UUID) -> bool:
        """启动指定任务"""
        if task_id in self.tasks:
            return self.tasks[task_id].start()
        return False
    
    def start_all(self) -> int:
        """启动所有待执行的任务"""
        started_count = 0
        self.is_running = True
        
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING:
                if task.start():
                    started_count += 1
        
        return started_count
    
    def pause_all(self) -> int:
        """暂停所有运行中的任务（实际上是取消，因为asyncio不支持真正的暂停）"""
        paused_count = 0
        self.is_running = False
        
        for task in self.tasks.values():
            if task.status == TaskStatus.RUNNING:
                if task.cancel("Paused by team"):
                    paused_count += 1
        
        return paused_count
    
    def restart_all(self, recreate_cancelled: bool = True) -> int:
        """重启所有任务"""
        restarted_count = 0
        
        if recreate_cancelled:
            # 重新创建被取消的任务
            cancelled_tasks = []
            for task_id, task in self.tasks.items():
                if task.status == TaskStatus.CANCELLED:
                    cancelled_tasks.append((task_id, task))
            
            for task_id, old_task in cancelled_tasks:
                # 创建新的任务实例
                new_task = AsyncTask(old_task.coro, old_task.name, team=self, auto_start=False)
                self.tasks[new_task.id] = new_task
                # 移除旧任务
                del self.tasks[task_id]
                restarted_count += 1
        
        # 启动所有待执行的任务
        started_count = self.start_all()
        return restarted_count + started_count
    
    async def run_all_concurrent(self, max_workers: Optional[int] = None) -> Dict[uuid.UUID, Any]:
        """并发运行所有任务"""
        if not self.tasks:
            return {}
        
        # 确定实际的并发数
        actual_max = max_workers or self.max_concurrent or len(self.tasks)
        
        # 启动所有任务
        self.start_all()
        
        # 如果有并发限制，使用semaphore控制
        if actual_max < len(self.tasks):
            return await self._run_with_semaphore(actual_max)
        else:
            # 直接等待所有任务
            return await self.wait_all()
    
    async def _run_with_semaphore(self, max_concurrent: int) -> Dict[uuid.UUID, Any]:
        """使用信号量控制并发执行"""
        semaphore = asyncio.Semaphore(max_concurrent)
        results = {}
        
        async def limited_task(task: AsyncTask):
            async with semaphore:
                try:
                    result = await task.wait()
                    return task.id, result
                except Exception as e:
                    return task.id, e
        
        # 创建受限制的任务列表
        limited_tasks = [limited_task(task) for task in self.tasks.values() if task.started()]
        
        # 等待所有受限任务完成
        completed = await asyncio.gather(*limited_tasks, return_exceptions=True)
        
        for item in completed:
            if isinstance(item, tuple):
                task_id, result = item
                results[task_id] = result
            else:
                # 处理异常情况
                results[uuid.uuid4()] = item
        
        return results
    
    async def run_batch(self, batch_size: int = 5, delay_between_batches: float = 0.1) -> Dict[uuid.UUID, Any]:
        """分批运行任务"""
        if not self.tasks:
            return {}
        
        all_results = {}
        task_list = list(self.tasks.values())
        
        # 分批处理
        for i in range(0, len(task_list), batch_size):
            batch = task_list[i:i + batch_size]
            
            # 启动这一批任务
            for task in batch:
                if task.status == TaskStatus.PENDING:
                    task.start()
            
            # 等待这一批完成
            batch_results = {}
            for task in batch:
                try:
                    result = await task.wait()
                    batch_results[task.id] = result
                except Exception as e:
                    batch_results[task.id] = e
            
            all_results.update(batch_results)
            
            # 批次间延迟
            if i + batch_size < len(task_list) and delay_between_batches > 0:
                await asyncio.sleep(delay_between_batches)
        
        return all_results

    async def add_task_with_semaphore(self, coro_func, *args, name: Optional[str] = None, **kwargs) -> AsyncTask:
        """使用信号量控制并发添加任务"""
        if not self.semaphore:
            return self.add_task(coro_func(*args, **kwargs), name)
        
        async def wrapped_coro():
            async with self.semaphore:
                return await coro_func(*args, **kwargs)
        
        return self.add_task(wrapped_coro(), name)
    
    def cancel_task(self, task_id: uuid.UUID, msg: str = "Cancelled by team") -> bool: 
        """取消指定任务"""
        if task_id in self.tasks:
            return self.tasks[task_id].cancel(msg)
        return False
    
    def cancel_all(self, msg: str = "All tasks cancelled by team"):
        """取消所有任务"""
        cancelled_count = 0
        for task in self.tasks.values():
            if task.cancel(msg):
                cancelled_count += 1
        return cancelled_count
    
    def _on_task_done(self, task_id: uuid.UUID):
        """任务完成时的内部回调"""
        if task_id in self.tasks:
            task = self.tasks.pop(task_id)
            self.completed_tasks[task_id] = task
    
    async def wait_all(self, timeout: Optional[float] = None) -> Dict[uuid.UUID, Any]:
        """等待所有任务完成"""
        if not self.tasks:
            return {}
        
        # 启动所有未启动的任务
        for task in self.tasks.values():
            if not task.started():
                task.start()
        
        tasks = list(self.tasks.values())
        if timeout:
            await asyncio.wait_for(
                asyncio.gather(*[task.wait() for task in tasks], return_exceptions=True),
                timeout=timeout
            )
        else:
            await asyncio.gather(*[task.wait() for task in tasks], return_exceptions=True)
        
        # 返回所有结果
        results = {}
        for task in tasks:
            try:
                results[task.id] = task.result_value
            except Exception as e:
                results[task.id] = e
        
        return results
    
    def get_status_summary(self) -> Dict[str, int]:
        """获取任务状态统计"""
        summary = {status.value: 0 for status in TaskStatus}
        
        for task in list(self.tasks.values()) + list(self.completed_tasks.values()):
            summary[task.status.value] += 1
        
        return summary
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        all_tasks = list(self.tasks.values()) + list(self.completed_tasks.values())
        
        if not all_tasks:
            return {"total_tasks": 0}
        
        completed_tasks = [t for t in all_tasks if t.execution_time is not None]
        
        stats = {
            "total_tasks": len(all_tasks),
            "completed_tasks": len(completed_tasks),
            "pending_tasks": len([t for t in all_tasks if t.status == TaskStatus.PENDING]),
            "running_tasks": len([t for t in all_tasks if t.status == TaskStatus.RUNNING]),
            "failed_tasks": len([t for t in all_tasks if t.status == TaskStatus.FAILED]),
            "cancelled_tasks": len([t for t in all_tasks if t.status == TaskStatus.CANCELLED]),
        }
        
        if completed_tasks:
            execution_times = [t.execution_time for t in completed_tasks]
            stats.update({
                "avg_execution_time": sum(execution_times) / len(execution_times),
                "min_execution_time": min(execution_times),
                "max_execution_time": max(execution_times),
            })
        
        return stats
    
    @property
    def active_count(self) -> int:
        """活跃任务数量"""
        return len(self.tasks)
    
    @property
    def completed_count(self) -> int:
        """已完成任务数量"""
        return len(self.completed_tasks)
    
    @property
    def pending_count(self) -> int:
        """待执行任务数量"""
        return len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING])
    
    @property
    def running_count(self) -> int:
        """运行中任务数量"""
        return len([t for t in self.tasks.values() if t.status == TaskStatus.RUNNING])
    
    def __repr__(self):
        return f"AsyncTaskTeam(name={self.name}, active={self.active_count}, completed={self.completed_count}, pending={self.pending_count}, running={self.running_count})"


class AsyncTaskPoolMixin:
    """异步任务池混入类"""
    
    def init_async_task(self):
        """初始化异步任务管理器"""
        self.async_tasks: Dict[uuid.UUID, AsyncTask] = {}
        self.teams: Dict[str, AsyncTaskTeam] = {}
        self.default_team = self.create_team("default")
    
    def task(self, name: Optional[str] = None, team: Optional[str] = None, auto_start: bool = True):
        """任务装饰器"""
        def decorator(func: Callable):
            if asyncio.iscoroutinefunction(func):
                # 如果已经是协程函数，直接使用
                async def async_wrapper(*args, **kwargs):
                    return await func(*args, **kwargs)
            else:
                # 如果是同步函数，包装成协程
                async def async_wrapper(*args, **kwargs):
                    return func(*args, **kwargs)
            
            def task_creator(*args, **kwargs) -> AsyncTask:
                coro = async_wrapper(*args, **kwargs)
                target_team = self.get_team(team) if team else self.default_team
                task = target_team.add_task(coro, name or func.__name__, auto_start=auto_start)
                self.async_tasks[task.id] = task
                return task
            
            # 保留原函数的元数据
            task_creator.__name__ = func.__name__
            task_creator.__doc__ = func.__doc__
            task_creator.original_func = func
            
            return task_creator
        return decorator
    
    def lazy_task(self, name: Optional[str] = None, team: Optional[str] = None):
        """延迟执行任务装饰器"""
        return self.task(name=name, team=team, auto_start=False)
    
    def create_task(self, coro, name: Optional[str] = None, team: Optional[str] = None, auto_start: bool = True) -> AsyncTask:
        """创建单个异步任务"""
        target_team = self.get_team(team) if team else self.default_team
        task = target_team.add_task(coro, name, auto_start=auto_start)
        self.async_tasks[task.id] = task
        return task
    
    def create_lazy_task(self, coro, name: Optional[str] = None, team: Optional[str] = None) -> AsyncTask:
        """创建延迟执行的异步任务"""
        return self.create_task(coro, name, team, auto_start=False)
    
    def start_task(self, task_id: uuid.UUID) -> bool:
        """启动指定任务"""
        if task_id in self.async_tasks:
            return self.async_tasks[task_id].start()
        return False
    
    def start_all_tasks(self) -> int:
        """启动所有待执行的任务"""
        started_count = 0
        for task in self.async_tasks.values():
            if task.status == TaskStatus.PENDING:
                if task.start():
                    started_count += 1
        return started_count
    
    async def run_all_teams(self, max_workers_per_team: Optional[int] = None) -> Dict[str, Dict[uuid.UUID, Any]]:
        """运行所有团队的所有任务"""
        results = {}
        for team_name, team in self.teams.items():
            try:
                results[team_name] = await team.run_all_concurrent(max_workers_per_team)
            except Exception as e:
                results[team_name] = {"error": str(e)}
        return results
    
    async def run_team_batch(self, team_name: str, batch_size: int = 5, delay: float = 0.1) -> Dict[uuid.UUID, Any]:
        """分批运行指定团队的任务"""
        team = self.get_team(team_name)
        if not team:
            raise ValueError(f"Team '{team_name}' not found")
        return await team.run_batch(batch_size, delay)
    
    def cancel_task(self, task_id: uuid.UUID, msg: str = "Cancelled by pool") -> bool:
        """取消指定任务"""
        if task_id in self.async_tasks:
            return self.async_tasks[task_id].cancel(msg)
        return False
    
    def get_task(self, task_id: uuid.UUID) -> Optional[AsyncTask]:
        """获取任务"""
        return self.async_tasks.get(task_id)
    
    def create_team(self, name: str, max_concurrent: Optional[int] = None) -> AsyncTaskTeam:
        """创建任务团队"""
        if name in self.teams:
            raise ValueError(f"Team '{name}' already exists")
        
        team = AsyncTaskTeam(name, max_concurrent)
        self.teams[name] = team
        return team
    
    def get_team(self, name: str) -> Optional[AsyncTaskTeam]:
        """获取任务团队"""
        return self.teams.get(name)
    
    def delete_team(self, name: str, cancel_tasks: bool = True) -> bool:
        """删除任务团队"""
        if name not in self.teams:
            return False
        
        team = self.teams[name]
        if cancel_tasks:
            team.cancel_all("Team deleted")
        
        del self.teams[name]
        return True
    
    async def wait_all_tasks(self, timeout: Optional[float] = None) -> Dict[str, Dict[uuid.UUID, Any]]:
        """等待所有团队的所有任务完成"""
        results = {}
        for team_name, team in self.teams.items():
            try:
                results[team_name] = await team.wait_all(timeout)
            except asyncio.TimeoutError:
                results[team_name] = {"error": "timeout"}
            except Exception as e:
                results[team_name] = {"error": str(e)}
        
        return results
    
    def get_global_status(self) -> Dict[str, Any]:
        """获取全局任务状态"""
        total_tasks = len(self.async_tasks)
        team_stats = {}
        
        for team_name, team in self.teams.items():
            team_stats[team_name] = {
                "active": team.active_count,
                "completed": team.completed_count,
                "pending": team.pending_count,
                "running": team.running_count,
                "status_summary": team.get_status_summary(),
                "performance": team.get_performance_stats()
            }
        
        # 全局统计
        global_stats = {
            "total_tasks": total_tasks,
            "pending_tasks": len([t for t in self.async_tasks.values() if t.status == TaskStatus.PENDING]),
            "running_tasks": len([t for t in self.async_tasks.values() if t.status == TaskStatus.RUNNING]),
            "completed_tasks": len([t for t in self.async_tasks.values() if t.status == TaskStatus.COMPLETED]),
            "failed_tasks": len([t for t in self.async_tasks.values() if t.status == TaskStatus.FAILED]),
            "cancelled_tasks": len([t for t in self.async_tasks.values() if t.status == TaskStatus.CANCELLED]),
        }
        
        return {
            "global_stats": global_stats,
            "teams": team_stats,
            "team_count": len(self.teams)
        }
    
    def cleanup_completed_tasks(self):
        """清理已完成的任务"""
        completed_ids = []
        for task_id, task in self.async_tasks.items():
            if task.done():
                completed_ids.append(task_id)
        
        for task_id in completed_ids:
            del self.async_tasks[task_id]
        
        return len(completed_ids)


class Tree(AsyncTaskPoolMixin):
    def __init__(self, val_keys=[], max_size=4):
        self.val_keys = val_keys
        self.max_size = max_size
        self.root = Node(is_leaf=True, val_keys=val_keys, max_size=max_size)
        self.leaf_head = self.root  # 叶子节点链表的头节点
        self.init_async_task()
    def search(self, key):
        """搜索指定键的值"""
        node = self._search_leaf(key)
        return node.val.row(key) if node else None
    
    def _search_leaf(self, key):
        """搜索到包含指定键的叶子节点"""
        current = self.root
        
        while not current.is_leaf:
            # 在内部节点中找到合适的子节点
            i = 0
            while i < len(current.keys) and key >= current.keys[i]:
                i += 1
            current = current.children[i]
        
        return current
    
    def insert(self, key, row_data):
        """插入键值对"""
        leaf = self._search_leaf(key)
        
        # 检查键是否已存在
        existing_row = leaf.val.row(key)
        if existing_row:
            # 更新现有行
            existing_row.row.update(row_data)
            return
        
        # 插入新行 - 修复addrow调用
        leaf.val.addrow(key, row_data, -1)
        
        # 检查是否需要分裂
        if leaf.is_full():
            self._split_leaf_and_promote(leaf)
    
    def _split_leaf_and_promote(self, leaf):
        """分裂叶子节点并向上传播"""
        promote_key, new_leaf = leaf.split_leaf()
        
        if leaf.parent is None:
            # 根节点分裂：创建新的根节点
            new_root = Node(is_leaf=False, max_size=self.max_size)
            new_root.keys = [promote_key]
            new_root.children = [leaf, new_leaf]
            leaf.parent = new_root
            new_leaf.parent = new_root
            self.root = new_root
        else:
            # 向父节点插入新键
            self._insert_into_parent(leaf.parent, promote_key, new_leaf)
    
    def _insert_into_parent(self, parent, key, new_node):
        """向父节点插入键和子节点"""
        # 找到插入位置
        i = 0
        while i < len(parent.keys) and key >= parent.keys[i]:
            i += 1
        
        # 插入键和子节点
        parent.keys.insert(i, key)
        parent.children.insert(i + 1, new_node)
        new_node.parent = parent
        
        # 检查父节点是否需要分裂
        if parent.is_full():
            self._split_internal_and_promote(parent)
    
    def _split_internal_and_promote(self, internal):
        """分裂内部节点并向上传播"""
        promote_key, new_internal = internal.split_internal()
        
        if internal.parent is None:
            # 根节点分裂：创建新的根节点
            new_root = Node(is_leaf=False, max_size=self.max_size)
            new_root.keys = [promote_key]
            new_root.children = [internal, new_internal]
            internal.parent = new_root
            new_internal.parent = new_root
            self.root = new_root
        else:
            # 向父节点插入新键
            self._insert_into_parent(internal.parent, promote_key, new_internal)
    
    def delete(self, key):
        """删除指定键"""
        leaf = self._search_leaf(key)
        existing_row = leaf.val.row(key) if leaf else None
        if not leaf or not existing_row:
            return False
        
        leaf.val.delrow(key)
        return True
    
    def range_query(self, start_key=None, end_key=None):
        """范围查询：返回指定范围内的所有键值对"""
        result = []
        current = self.leaf_head
        
        while current:
            for row in current.val.rows:
                if start_key is not None and row.key < start_key:
                    continue
                if end_key is not None and row.key > end_key:
                    return result
                result.append((row.key, row.row))
            current = current.next
        
        return result
    
    def query(self,*args,**kwargs):
        """
        对整个B+树中的所有表数据进行全局查询
        
        Parameters
        ----------
        func : callable
            筛选函数，接收行/列数据并返回bool，决定是否保留
        **options : dict
            查询选项，会传递给每个节点的val.query方法，包括：
            - 'limit': 全局结果数量限制（在合并后应用）
            - 'by': 查询对象（'row'或'col'） 
            - 'aggregate': 聚合方式
            - 'sort'/'order_by': 排序规则
            - 'offset': 偏移设置
            - 'group_by': 分组列
            - 'select': 选择列（投影）
            - 'distinct': 去重
            - 'having': 分组后筛选函数
            - 'join': 表连接配置
            - 'pivot': 数据透视配置
            - 'window': 窗口函数配置
            - 'merge_results': 是否合并所有节点的结果（默认True）
            - 'atask': 如果提供任务名，则创建异步任务执行查询
            - 'atask_auto_start': 异步任务是否自动启动（默认True）
            
        Returns
        -------
        list or dict or any or AsyncTask
            如果使用atask选项，返回AsyncTask对象；否则返回查询结果
        """
        
        # 检查是否需要创建异步任务
        if kwargs.get('atask') is not None:
            task_name = kwargs.pop('atask')
            auto_start = kwargs.pop('atask_auto_start', True)
            
            # 创建异步包装函数
            async def async_query():
                return self._base_query(*args, **kwargs)
            
            # 创建并返回异步任务
            return self.create_task(async_query(), name=task_name, auto_start=auto_start)
        
        return self._base_query(*args,**kwargs)
    
    def _base_query(self, func, **options):
        """
        对整个B+树中的所有表数据进行全局查询
        
        Parameters
        ----------
        func : callable
            筛选函数，接收行/列数据并返回bool，决定是否保留
        **options : dict
            查询选项，会传递给每个节点的val.query方法，包括：
            - 'limit': 全局结果数量限制（在合并后应用）
            - 'by': 查询对象（'row'或'col'） 
            - 'aggregate': 聚合方式
            - 'sort'/'order_by': 排序规则
            - 'offset': 偏移设置
            - 'group_by': 分组列
            - 'select': 选择列（投影）
            - 'distinct': 去重
            - 'having': 分组后筛选函数
            - 'join': 表连接配置
            - 'pivot': 数据透视配置
            - 'window': 窗口函数配置
            - 'merge_results': 是否合并所有节点的结果（默认True）
            
        Returns
        -------
        list or dict or any
            全局查询结果
        """
        if not callable(func):
            return []
        
        # 提取全局选项
        global_limit = options.pop('limit', None)
        merge_results = options.pop('merge_results', True)
        aggregate = options.get('aggregate', False)
        
        # 收集所有叶子节点的查询结果
        all_results = []
        aggregate_values = []  # 用于收集聚合前的原始数据
        current_leaf = self.leaf_head
        
        while current_leaf:
            if current_leaf.val.rows:  # 只查询非空的叶子节点
                # 为每个叶子节点创建选项副本（避免修改原始选项）
                node_options = options.copy()
                
                # 如果有聚合操作，先收集原始数据，稍后统一聚合
                if aggregate:
                    # 临时移除aggregate选项，获取原始数据
                    temp_aggregate = node_options.pop('aggregate', None)
                    node_result = current_leaf.val.query(func, **node_options)
                    # 收集原始数据用于全局聚合
                    if isinstance(node_result, list):
                        aggregate_values.extend(node_result)
                else:
                    # 对单个节点执行查询
                    node_result = current_leaf.val.query(func, **node_options)
                
                    # 根据结果类型进行处理
                    if isinstance(node_result, list):
                        all_results.extend(node_result)
                    elif isinstance(node_result, dict):
                        # 如果是字典结果（如分组、pivot），需要特殊处理
                        if merge_results:
                            # 合并字典结果
                            if not hasattr(self, '_merged_dict_result'):
                                self._merged_dict_result = {}
                            for k, v in node_result.items():
                                if k in self._merged_dict_result:
                                    if isinstance(v, list):
                                        self._merged_dict_result[k].extend(v)
                                    else:
                                        self._merged_dict_result[k] = v
                                else:
                                    self._merged_dict_result[k] = v
                        else:
                            all_results.append(node_result)
                    elif node_result is not None:
                        all_results.append(node_result)
            
            current_leaf = current_leaf.next
        
        # 处理聚合操作
        if aggregate and aggregate_values:
            # 使用原始的_option_aggregate方法进行全局聚合
            dummy_table = type('DummyTable', (), {})()
            dummy_table._option_aggregate = self.leaf_head.val._option_aggregate.__get__(dummy_table)
            return dummy_table._option_aggregate(aggregate_values, aggregate)
        
        # 处理字典类型的合并结果
        if hasattr(self, '_merged_dict_result'):
            result = self._merged_dict_result
            delattr(self, '_merged_dict_result')
            return result
        
        # 如果不合并结果，直接返回
        if not merge_results:
            return all_results
        
        # 应用全局限制
        if global_limit and isinstance(all_results, list):
            all_results = all_results[:global_limit]
        
        return all_results
    
    def query_summary(self, func, **options):
        """
        对整个B+树进行查询并返回统计摘要
        
        Parameters
        ----------
        func : callable
            筛选函数
        **options : dict
            查询选项，传给query
            
        Returns
        -------
        dict
            查询摘要，包含匹配记录数、节点数等信息
        """
        matched_count = 0
        node_count = 0
        empty_nodes = 0
        
        current_leaf = self.leaf_head
        while current_leaf:
            node_count += 1
            if not current_leaf.val.rows:
                empty_nodes += 1
            else:
                # 统计匹配的记录数
                node_options = options.copy()
                node_result = current_leaf.val.query(func, **node_options)
                if isinstance(node_result, list):
                    matched_count += len(node_result)
                elif isinstance(node_result, dict):
                    matched_count += len(node_result)
                elif node_result is not None:
                    matched_count += 1
            
            current_leaf = current_leaf.next
        
        return {
            'total_nodes': node_count,
            'empty_nodes': empty_nodes,
            'active_nodes': node_count - empty_nodes,
            'matched_records': matched_count,
            'query_function': func.__name__ if hasattr(func, '__name__') else str(func)
        }

    def display(self):
        """显示树的结构"""
        def _display_node(node, level=0):
            indent = "  " * level
            if node.is_leaf:
                print(f"{indent}Leaf: {[row.key for row in node.val.rows]}")
            else:
                print(f"{indent}Internal: {node.keys}")
                for child in node.children:
                    _display_node(child, level + 1)
        
        print("B+ Tree Structure:")
        _display_node(self.root)
    
    def __repr__(self):
        """返回树的字符串表示"""
        return f"B+Tree(max_size={self.max_size}, root={'Leaf' if self.root.is_leaf else 'Internal'})"


# 测试代码
if __name__ == "__main__":
    # ============ B+树测试 ============
    # 创建一个B+树
    tree = Tree(val_keys=['name', 'age', 'city'], max_size=3)
    
    print("=== B+树测试 ===")
    
    # 插入数据
    test_data = [
        ('alice', {'name': 'Alice', 'age': 25, 'city': 'New York'}),
        ('bob', {'name': 'Bob', 'age': 30, 'city': 'London'}),
        ('charlie', {'name': 'Charlie', 'age': 35, 'city': 'Paris'}),
        ('david', {'name': 'David', 'age': 28, 'city': 'Tokyo'}),
        ('eve', {'name': 'Eve', 'age': 32, 'city': 'Berlin'}),
        ('frank', {'name': 'Frank', 'age': 27, 'city': 'Sydney'}),
    ]
    
    print("\n插入数据...")
    for key, data in test_data:
        tree.insert(key, data)
        print(f"插入: {key} -> {data}")
    
    print(f"\n树结构:")
    tree.display()
    
    print(f"\n搜索测试:")
    search_keys = ['alice', 'david', 'nonexistent']
    for key in search_keys:
        result = tree.search(key)
        print(f"搜索 '{key}': {result.row if result else 'Not found'}")
    
    print(f"\n范围查询测试 (b-e):")
    range_result = tree.range_query('b', 'e')
    for key, data in range_result:
        print(f"  {key}: {data}")
    
    print(f"\n删除测试:")
    tree.delete('charlie')
    print("删除 'charlie' 后的搜索结果:", tree.search('charlie'))
    
    print(f"\n最终树结构:")
    tree.display()
    
    print("\n" + "="*60)
    
    # ============ Tree 全局查询功能测试 ============
    print("=== Tree 全局查询功能测试 ===")
    
    # 首先为B+树插入更多数据以便测试全局查询
    additional_data = [
        ('grace', {'name': 'Grace', 'age': 29, 'city': 'Toronto'}),
        ('henry', {'name': 'Henry', 'age': 33, 'city': 'Berlin'}),
        ('iris', {'name': 'Iris', 'age': 26, 'city': 'Madrid'}),
        ('jack', {'name': 'Jack', 'age': 31, 'city': 'Rome'}),
    ]
    
    print("\n插入更多测试数据...")
    for key, data in additional_data:
        tree.insert(key, data)
        print(f"插入: {key} -> {data}")
    
    print(f"\n扩展后的树结构:")
    tree.display()
    
    # 1. 全局基础查询测试
    print("\n1. 全局基础查询 - 查找年龄大于30的所有人:")
    result = tree.query(
        lambda x: x.get('age', 0) > 30,
        select=['name', 'age', 'city']
    )
    for person in result:
        print(f"  {person}")
    
    # 2. 全局排序查询
    print("\n2. 全局排序查询 - 按年龄升序排列所有人:")
    result = tree.query(
        lambda x: True,  # 选择所有记录
        select=['name', 'age'],
        order_by=[('age', 'asc')],
        limit=5
    )
    for person in result:
        print(f"  {person}")
    
    # 3. 全局分组查询
    print("\n3. 全局分组查询 - 按城市分组:")
    result = tree.query(
        lambda x: True,
        group_by='city'
    )
    for city_key, people in list(result.items())[:3]:  # 显示前3个城市
        city_name = city_key[1]  # 获取城市名称
        print(f"  {city_name}: {len(people)} 人")
        for person in people[:2]:  # 每城市显示最多2人
            print(f"    - {person['name']}, {person['age']}岁")
    
    # 4. 全局去重查询
    print("\n4. 全局去重查询 - 获取所有不重复的城市:")
    result = tree.query(
        lambda x: True,
        select=['city'],
        distinct=['city']
    )
    cities = [r['city'] for r in result]
    print(f"  所有城市: {', '.join(cities)}")
    
    # 5. 全局统计查询
    print("\n5. 全局统计查询 - 平均年龄:")
    result = tree.query(
        lambda x: x.get('age') is not None,
        select=['age'],
        aggregate='avg'
    )
    print(f"  全树平均年龄: {result:.1f}岁")
    
    # 6. 全局查询摘要
    print("\n6. 全局查询摘要 - 年龄>28的记录统计:")
    summary = tree.query_summary(
        lambda x: x.get('age', 0) > 28
    )
    print(f"  总节点数: {summary['total_nodes']}")
    print(f"  活跃节点数: {summary['active_nodes']}")
    print(f"  空节点数: {summary['empty_nodes']}")
    print(f"  匹配记录数: {summary['matched_records']}")
    
    # 7. 复杂全局查询组合
    print("\n7. 复杂全局查询 - 欧洲城市的人，按年龄降序:")
    european_cities = ['London', 'Paris', 'Berlin', 'Madrid', 'Rome']
    result = tree.query(
        lambda x: x.get('city') in european_cities,
        select=['name', 'age', 'city'],
        order_by=[('age', 'desc')],
        limit=4
    )
    for person in result:
        print(f"  {person}")
    
    # 8. 不合并结果的查询（按节点分别返回）
    print("\n8. 按节点分别查询 - 每个节点的年轻人(age<30):")
    result = tree.query(
        lambda x: x.get('age', 0) < 30,
        select=['name', 'age'],
        merge_results=False
    )
    for i, node_result in enumerate(result):
        if node_result and isinstance(node_result, list):  # 确保是非空列表
            print(f"  节点 {i+1}: {len(node_result)} 人")
            for person in node_result[:2]:  # 每节点最多显示2人
                print(f"    - {person}")
        elif node_result and isinstance(node_result, dict):
            print(f"  节点 {i+1}: 字典结果 - {node_result}")
        elif node_result:
            print(f"  节点 {i+1}: 其他结果 - {node_result}")
    
    print("\n" + "="*60)
    
    # ============ Table 增强查询功能测试 ============
    print("=== Table 增强查询功能测试 ===")
    
    # 创建测试表
    table = Table('id', 'name', 'age', 'department', 'salary', 'city')
    
    # 添加测试数据
    test_employees = [
        {'id': 1, 'name': 'Alice', 'age': 25, 'department': 'IT', 'salary': 60000, 'city': 'New York'},
        {'id': 2, 'name': 'Bob', 'age': 30, 'department': 'HR', 'salary': 55000, 'city': 'London'},
        {'id': 3, 'name': 'Charlie', 'age': 35, 'department': 'IT', 'salary': 70000, 'city': 'Paris'},
        {'id': 4, 'name': 'David', 'age': 28, 'department': 'Finance', 'salary': 65000, 'city': 'Tokyo'},
        {'id': 5, 'name': 'Eve', 'age': 32, 'department': 'IT', 'salary': 68000, 'city': 'Berlin'},
        {'id': 6, 'name': 'Frank', 'age': 27, 'department': 'HR', 'salary': 52000, 'city': 'Sydney'},
        {'id': 7, 'name': 'Grace', 'age': 29, 'department': 'Finance', 'salary': 63000, 'city': 'Toronto'},
        {'id': 8, 'name': 'Henry', 'age': 33, 'department': 'IT', 'salary': 72000, 'city': 'Berlin'},
        {'id': 9, 'name': 'Iris', 'age': 26, 'department': 'HR', 'salary': 54000, 'city': 'New York'},
        {'id': 10, 'name': 'Jack', 'age': 31, 'department': 'Finance', 'salary': 66000, 'city': 'London'},
    ]
    
    for i, emp in enumerate(test_employees, 1):
        table.addrow(f'emp_{i}', emp, -1)
    
    print("\n原始数据表:")
    print(table)

    # 1. SELECT (投影) 测试
    print("\n1. SELECT 投影测试 - 只选择 name 和 salary 列:")
    result = table.query(
        lambda x: True,  # 选择所有行
        select=['name', 'salary']
    )
    for row in result[:3]:  # 显示前3条
        print(f"  {row}")
    
    # 2. DISTINCT 去重测试
    print("\n2. DISTINCT 去重测试 - 按 department 去重:")
    result = table.query(
        lambda x: True,
        select=['department'],
        distinct=['department']
    )
    for row in result:
        print(f"  {row}")
    
    # 3. ORDER BY 排序测试
    print("\n3. ORDER BY 排序测试 - 按 age 升序，salary 降序:")
    result = table.query(
        lambda x: True,
        select=['name', 'age', 'salary'],
        order_by=[('age', 'asc'), ('salary', 'desc')],
        limit=5
    )
    for row in result:
        print(f"  {row}")
    
    # 4. GROUP BY + HAVING 测试
    print("\n4. GROUP BY + HAVING 测试 - 按部门分组，平均工资>60000的部门:")
    result = table.query(
        lambda x: True,
        group_by='department',
        having=lambda group: sum(row['salary'] for row in group) / len(group) > 60000
    )
    for dept, employees in result.items():
        avg_salary = sum(emp['salary'] for emp in employees) / len(employees)
        print(f"  {dept[1]}: 平均工资 {avg_salary:.0f}, 人数 {len(employees)}")
    
    # 5. 窗口函数测试
    print("\n5. 窗口函数测试 - 按部门分区，按工资排序，添加行号:")
    result = table.query(
        lambda x: True,
        select=['name', 'department', 'salary'],
        window={
            'func': 'row_number',
            'partition_by': 'department',
            'order_by': 'salary'
        }
    )
    for row in result[:6]:  # 显示前6条
        print(f"  {row}")
    
    # 6. 复杂查询组合测试
    print("\n6. 复杂查询组合 - IT部门，工资>65000，按工资降序，只显示姓名和工资:")
    result = table.query(
        lambda x: x['department'] == 'IT' and x['salary'] > 65000,
        select=['name', 'salary'],
        order_by=[('salary', 'desc')]
    )
    for row in result:
        print(f"  {row}")
    
    # 7. PIVOT 数据透视测试
    print("\n7. PIVOT 数据透视测试 - 按城市和部门统计人数:")
    result = table.query(
        lambda x: True,
        pivot={
            'index': 'city',
            'columns': 'department', 
            'values': 'id',
            'aggfunc': 'count'
        }
    )
    print("  城市-部门人数透视表:")
    for city, depts in list(result.items())[:3]:  # 显示前3个城市
        print(f"    {city}: {dict(depts)}")
    
    # 8. 表连接测试
    print("\n8. 表连接测试:")
    # 创建部门信息表
    dept_table = Table('dept_name', 'manager', 'budget')
    dept_data = [
        {'dept_name': 'IT', 'manager': 'John', 'budget': 500000},
        {'dept_name': 'HR', 'manager': 'Sarah', 'budget': 200000},
        {'dept_name': 'Finance', 'manager': 'Mike', 'budget': 300000},
    ]
    
    for i, dept in enumerate(dept_data, 1):
        dept_table.addrow(f'dept_{i}', dept, -1)
    
    result = table.query(
        lambda x: True,
        select=['name', 'department', 'salary', 'manager', 'budget'],
        join={
            'table': dept_table,
            'type': 'inner',
            'on': ('department', 'dept_name')
        },
        limit=5
    )
    
    for row in result:
        print(f"  {row}")
    
    print("\n测试完成！")
    
    print("\n" + "="*60)
    
    # ============ AsyncTask 系统测试 ============
    print("=== AsyncTask 异步任务系统测试 ===")
    
    async def async_task_demo():
        """异步任务系统演示"""
        
        # 1. 基础任务创建和管理
        print("\n1. 基础任务创建测试:")
        
        async def simple_task(name: str, delay: float):
            await asyncio.sleep(delay)
            return f"Task {name} completed after {delay}s"
        
        # 创建任务
        task1 = tree.create_task(simple_task("A", 1.0), name="SimpleTask-A")
        task2 = tree.create_task(simple_task("B", 0.5), name="SimpleTask-B")
        
        print(f"创建任务: {task1}")
        print(f"创建任务: {task2}")
        
        # 等待任务完成
        result1 = await task1
        result2 = await task2
        
        print(f"任务A结果: {result1}")
        print(f"任务B结果: {result2}")
        
        # 2. 任务装饰器测试
        print("\n2. 任务装饰器测试:")
        
        @tree.task(name="DecoratedTask")
        async def decorated_async_task(x: int, y: int):
            await asyncio.sleep(0.2)
            return x + y
        
        @tree.task(name="SyncTask", team="math_team")
        def decorated_sync_task(x: int, y: int):
            return x * y
        
        # 创建数学运算团队
        math_team = tree.create_team("math_team", max_concurrent=3)
        
        # 使用装饰器创建任务
        add_task = decorated_async_task(10, 20)
        mul_task = decorated_sync_task(5, 6)
        
        print(f"装饰器任务: {add_task}")
        print(f"装饰器任务: {mul_task}")
        
        add_result = await add_task
        mul_result = await mul_task
        
        print(f"加法结果: {add_result}")
        print(f"乘法结果: {mul_result}")
        
        # 3. 团队管理测试
        print("\n3. 团队管理测试:")
        
        # 创建数据处理团队（不设置并发限制，在任务级别控制）
        data_team = tree.create_team("data_processing")
        
        async def process_data(data_id: int, processing_time: float):
            await asyncio.sleep(processing_time)
            return f"Data {data_id} processed"
        
        # 批量添加任务到团队
        data_tasks = []
        for i in range(5):
            task = data_team.add_task(
                process_data(i, 0.2 + i * 0.05), 
                name=f"DataProcess-{i}"
            )
            tree.async_tasks[task.id] = task
            data_tasks.append(task)
        
        print(f"数据处理团队: {data_team}")
        print(f"数学团队: {math_team}")
        
        # 等待数据处理团队的所有任务完成
        data_results = await data_team.wait_all(timeout=5.0)
        print(f"数据处理结果: {len(data_results)} 个任务完成")
        
        # 4. 并发控制测试
        print("\n4. 并发控制测试:")
        
        # 创建有并发限制的团队
        limited_team = tree.create_team("limited_concurrent", max_concurrent=2)
        
        async def concurrent_task(task_id: int):
            print(f"  开始处理任务 {task_id}")
            await asyncio.sleep(0.3)
            print(f"  完成处理任务 {task_id}")
            return f"Task {task_id} done"
        
        # 手动添加任务并处理并发限制
        limited_tasks = []
        for i in range(4):
            try:
                # 检查是否可以添加任务
                if len(limited_team.tasks) < limited_team.max_concurrent:
                    # 只有在确定可以添加时才创建协程
                    task = limited_team.add_task(concurrent_task(i), name=f"Limited-{i}")
                    tree.async_tasks[task.id] = task
                    limited_tasks.append(task)
                    print(f"  添加任务 {i} 到受限团队")
                else:
                    print(f"  任务 {i} 等待队列位置...")
                    # 等待一段时间让现有任务完成
                    await asyncio.sleep(0.4)  # 增加等待时间确保任务完成
                    
                    # 重试添加
                    if len(limited_team.tasks) < limited_team.max_concurrent:
                        task = limited_team.add_task(concurrent_task(i), name=f"Limited-{i}")
                        tree.async_tasks[task.id] = task
                        limited_tasks.append(task)
                        print(f"  延迟添加任务 {i}")
                    else:
                        print(f"  任务 {i} 队列仍满，跳过添加")
            except RuntimeError as e:
                print(f"  任务 {i} 添加失败: {e}")
        
        # 等待有限并发任务完成
        if limited_tasks:
            limited_results = await limited_team.wait_all()
            print(f"并发控制测试完成: {len(limited_results)} 个任务")
        
        # 5. 任务状态监控
        print("\n5. 任务状态监控:")
        
        async def long_running_task(task_id: str):
            for i in range(5):
                await asyncio.sleep(0.2)
                if i == 2:  # 模拟中途检查
                    print(f"  任务 {task_id} 进度: {(i+1)/5*100:.0f}%")
            return f"LongTask {task_id} finished"
        
        # 创建监控团队
        monitor_team = tree.create_team("monitoring")
        
        # 添加长时间运行的任务
        long_tasks = []
        for i in range(3):
            task = monitor_team.add_task(
                long_running_task(f"L{i}"), 
                name=f"LongTask-{i}"
            )
            tree.async_tasks[task.id] = task
            long_tasks.append(task)
        
        # 等待一段时间后检查状态
        await asyncio.sleep(0.5)
        status = tree.get_global_status()
        print(f"全局状态: {status}")
        
        # 等待剩余任务完成
        await monitor_team.wait_all()
        print("所有长任务完成")
        
        # 6. 错误处理测试
        print("\n6. 错误处理测试:")
        
        async def error_task(should_fail: bool):
            await asyncio.sleep(0.1)
            if should_fail:
                raise ValueError("Intentional error for testing")
            return "Success"
        
        error_team = tree.create_team("error_handling")
        
        # 创建成功和失败的任务
        success_task = error_team.add_task(error_task(False), name="SuccessTask")
        failure_task = error_team.add_task(error_task(True), name="FailureTask")
        tree.async_tasks[success_task.id] = success_task
        tree.async_tasks[failure_task.id] = failure_task
        
        # 等待任务完成并检查结果
        error_results = await error_team.wait_all()
        
        for task_id, result in error_results.items():
            if isinstance(result, Exception):
                print(f"  任务失败: {result}")
            else:
                print(f"  任务成功: {result}")
        
        # 7. 任务取消测试
        print("\n7. 任务取消测试:")
        
        async def cancellable_task(task_name: str):
            try:
                for i in range(10):
                    await asyncio.sleep(0.1)
                return f"Task {task_name} completed"
            except asyncio.CancelledError:
                print(f"  任务 {task_name} 被取消")
                raise
        
        cancel_team = tree.create_team("cancellation")
        
        # 创建可取消的任务
        cancel_task1 = cancel_team.add_task(cancellable_task("C1"), name="CancelTest-1")
        cancel_task2 = cancel_team.add_task(cancellable_task("C2"), name="CancelTest-2")
        tree.async_tasks[cancel_task1.id] = cancel_task1
        tree.async_tasks[cancel_task2.id] = cancel_task2
        
        # 等待一段时间后取消一个任务
        await asyncio.sleep(0.3)
        cancel_team.cancel_task(cancel_task1.id, "Manual cancellation")
        
        # 等待剩余任务
        cancel_results = await cancel_team.wait_all()
        print(f"取消测试结果: {len(cancel_results)} 个任务处理完毕")
        
        # 8. 最终状态报告
        print("\n8. 最终状态报告:")
        final_status = tree.get_global_status()
        print(f"最终全局状态: {final_status}")
        
        # 清理已完成的任务
        cleaned_count = tree.cleanup_completed_tasks()
        print(f"清理了 {cleaned_count} 个已完成的任务")
        
        print("\nAsyncTask系统测试完成!")
    
    # 运行异步任务演示
    try:
        asyncio.run(async_task_demo())

    except Exception as e:
        print(f"异步任务演示出错: {e}")
        import traceback
        traceback.print_exc()