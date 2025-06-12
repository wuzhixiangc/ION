#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IntegratedObjectNetwork (ION) - 结合OND和ICombinedDataStructure的高级数据结构
"""
import hashlib
import pickle
import threading
import concurrent.futures
import json
import base64
import os
import re
import difflib
import heapq  # 确保在文件顶部导入了heapq模块
import time
import uuid
import logging
import weakref
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from collections import deque, Counter, defaultdict
import numpy as np
import pandas as pd
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available. ML features will be disabled.")
    SKLEARN_AVAILABLE = False

try:
    import warnings
    warnings.filterwarnings('ignore')
except ImportError:
    pass
import functools
import logging
import time
# 导入bsd模块中的ICombinedDataStructure
from bsd import ICombinedDataStructure, HashTable, NestedBidirectionalMap, RelationshipChain

# 导入ond模块的核心类
from ond import RNode, obj_to_number
class MLInterceptorMeta(type):
    """元类：自动为所有方法添加ML拦截器"""
    
    def __new__(mcs, name, bases, namespace, **kwargs):
        # 定义需要记录的方法模式
        ml_tracked_methods = {
            'get_': 'read',
            'find_': 'search', 
            'search_': 'search',
            'query_': 'search',
            'create_': 'create',
            'add_': 'create',
            'update_': 'update',
            'modify_': 'update',
            'remove_': 'delete',
            'delete_': 'delete',
            'astar_': 'pathfinding',
            'fuzzy_': 'fuzzy_search',
            'batch_': 'batch_operation',
            'analyze_': 'analysis',
            'optimize_': 'optimization',
            'calculate_': 'calculation'
        }
        
        # 不应该被拦截的方法（避免递归）
        excluded_methods = {
            '__init__', '__getattr__', '__setattr__', '__getattribute__',
            '__getstate__', '__setstate__', '__str__', '__repr__',
            '_extract_input_features', '_extract_output_features', 
            '_record_io_for_training', '_analyze_method_result',
            'get_current_load_factor', 'hf', '_check_resize',
            '_trigger_io_training', '_record_batch_statistics',
            # ML引擎相关方法
            'record_access', 'record_query_performance', 
            '_extract_node_features', 'predict_cache_candidates',
            # 内部辅助方法
            '_get_node_from_input', '_normalize_row_id', '_index_row',
            '_remove_row_index', '_index_metadata', '_remove_metadata_index',
            '_index_tag', '_remove_tag_index', '_index_value_type',
            # 属性访问相关
            'keys', 'values', 'items', '__len__', '__contains__',
            '__getitem__', '__setitem__', '__delitem__'
        }
        
        # 包装符合条件的方法
        for attr_name, attr_value in list(namespace.items()):
            if (callable(attr_value) and 
                not attr_name.startswith('_') and 
                attr_name not in excluded_methods):
                
                # 检查是否匹配需要记录的方法模式
                access_type = 'unknown'
                for prefix, atype in ml_tracked_methods.items():
                    if attr_name.startswith(prefix):
                        access_type = atype
                        break
                
                # 只包装匹配的方法
                if access_type != 'unknown':
                    namespace[attr_name] = mcs._wrap_method(attr_value, attr_name, access_type)
        
        return super().__new__(mcs, name, bases, namespace)
    
    @staticmethod
    def _wrap_method(original_method, method_name, access_type):
        """包装方法以添加ML记录"""
        def ml_wrapped_method(self, *args, **kwargs):
            # 检查是否启用ML
            if not hasattr(self, 'ml_engine') or not self.ml_engine.enabled:
                return original_method(self, *args, **kwargs)
            
            start_time = time.time()
            
            try:
                # 执行原始方法
                result = original_method(self, *args, **kwargs)
                execution_time = time.time() - start_time
                
                # 安全地记录ML数据（避免递归）
                try:
                    # 记录访问模式
                    if args and hasattr(self, '_extract_node_key_from_arg'):
                        node_key = self._extract_node_key_from_arg(args[0])
                        if node_key:
                            self.ml_engine.record_access(node_key, access_type)
                    
                    # 记录查询性能
                    result_count = 0
                    if isinstance(result, (list, tuple, set)):
                        result_count = len(result)
                    elif result is not None:
                        result_count = 1
                    
                    self.ml_engine.record_query_performance(
                        method_name, execution_time, result_count
                    )
                    
                    # 简化的IO记录（避免复杂的特征提取）
                    if hasattr(self.ml_engine, 'io_training_data'):
                        self.ml_engine.io_training_data.append({
                            'method_name': method_name,
                            'args_count': len(args),
                            'kwargs_count': len(kwargs),
                            'execution_time': execution_time,
                            'result_size': result_count,
                            'access_type': access_type,
                            'timestamp': time.time()
                        })
                        
                        # 限制训练数据大小
                        if len(self.ml_engine.io_training_data) > 10000:
                            self.ml_engine.io_training_data = self.ml_engine.io_training_data[-5000:]
                
                except Exception as ml_error:
                    # ML记录失败不应该影响原始方法
                    pass
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                # 记录失败的操作
                try:
                    self.ml_engine.record_query_performance(
                        f"{method_name}_failed", execution_time, 0
                    )
                except:
                    pass
                raise e
        
        return ml_wrapped_method
# 机器学习智能引擎
class MLEngine:
    """ION数据结构的机器学习智能引擎"""
    
    def __init__(self, ion_instance):
        self.ion = ion_instance
        self.enabled = SKLEARN_AVAILABLE
        
        # 访问模式数据
        self.access_patterns = []
        self.access_history = defaultdict(list)
        self.query_performance = []
        
        # ML模型
        self.cache_predictor = None
        self.partition_optimizer = None
        self.path_optimizer = None
        self.anomaly_detector = None
        self.performance_predictor = None
        
        # 特征数据
        self.io_training_data = []  # 添加这一行
        self.node_features = {}
        self.query_features = []
        
        # 配置
        self.learning_enabled = True
        self.prediction_threshold = 0.7
        self.anomaly_threshold = -0.5
        
        if self.enabled:
            self._initialize_models()
    
    def _initialize_models(self):
        """初始化机器学习模型"""
        if not self.enabled:
            return
            
        try:
            # 缓存预测模型
            self.cache_predictor = MLPRegressor(
                hidden_layer_sizes=(50, 30),
                max_iter=500,
                random_state=42
            )
            
            # 分区优化模型
            self.partition_optimizer = KMeans(
                n_clusters=5,
                random_state=42
            )
            
            # 异常检测模型
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            # 性能预测模型
            self.performance_predictor = LinearRegression()
            
            #print("ML引擎初始化成功")
            
        except Exception as e:
            print(f"ML模型初始化失败: {e}")
            self.enabled = False
    
    def record_access(self, node_key, access_type='read', context=None):
        """记录节点访问模式"""
        if not self.enabled or not self.learning_enabled:
            return
            
        timestamp = time.time()
        access_record = {
            'node_key': node_key,
            'access_type': access_type,
            'timestamp': timestamp,
            'context': context or {}
        }
        
        self.access_patterns.append(access_record)
        self.access_history[node_key].append(access_record)
        
        # 限制历史记录大小
        if len(self.access_patterns) > 10000:
            self.access_patterns = self.access_patterns[-5000:]
    
    def record_query_performance(self, query_type, execution_time, result_count, context=None):
        """记录查询性能数据"""
        if not self.enabled or not self.learning_enabled:
            return
            
        performance_record = {
            'query_type': query_type,
            'execution_time': execution_time,
            'result_count': result_count,
            'timestamp': time.time(),
            'context': context or {}
        }
        
        self.query_performance.append(performance_record)
        
        # 限制记录大小
        if len(self.query_performance) > 5000:
            self.query_performance = self.query_performance[-2500:]
    
    def predict_cache_candidates(self, top_k=10):
        """预测应该缓存的节点"""
        if not self.enabled or len(self.access_patterns) < 100:
            return []
        
        try:
            # 计算节点访问特征
            node_stats = defaultdict(lambda: {
                'access_count': 0,
                'recent_accesses': 0,
                'access_frequency': 0.0,
                'last_access': 0
            })
            
            current_time = time.time()
            recent_threshold = current_time - 3600  # 1小时内
            
            for record in self.access_patterns:
                key = record['node_key']
                timestamp = record['timestamp']
                
                node_stats[key]['access_count'] += 1
                node_stats[key]['last_access'] = max(node_stats[key]['last_access'], timestamp)
                
                if timestamp > recent_threshold:
                    node_stats[key]['recent_accesses'] += 1
            
            # 计算访问频率
            for key, stats in node_stats.items():
                if stats['access_count'] > 0:
                    time_span = current_time - (current_time - 86400)  # 24小时
                    stats['access_frequency'] = stats['access_count'] / time_span
            
            # 按综合得分排序
            candidates = []
            for key, stats in node_stats.items():
                score = (
                    stats['access_frequency'] * 0.4 +
                    stats['recent_accesses'] * 0.3 +
                    (1.0 / (current_time - stats['last_access'] + 1)) * 0.3
                )
                candidates.append((key, score))
            
            candidates.sort(key=lambda x: x[1], reverse=True)
            return [key for key, score in candidates[:top_k]]
            
        except Exception as e:
            print(f"缓存预测失败: {e}")
            return []
    
    def optimize_partitions(self):
        """基于访问模式优化分区"""
        if not self.enabled or len(self.access_patterns) < 200:
            return {}
        
        try:
            # 构建节点关系矩阵
            node_keys = list(set(record['node_key'] for record in self.access_patterns))
            if len(node_keys) < 10:
                return {}
            
            # 创建特征矩阵
            features = []
            for key in node_keys:
                node = self.ion.get_node_by_key(key)
                if node:
                    feature_vector = self._extract_node_features(node)
                    features.append(feature_vector)
                else:
                    features.append([0] * 10)  # 默认特征
            
            features = np.array(features)
            
            # 使用KMeans聚类
            n_clusters = min(5, len(node_keys) // 10)
            if n_clusters < 2:
                return {}
            
            self.partition_optimizer.n_clusters = n_clusters
            cluster_labels = self.partition_optimizer.fit_predict(features)
            
            # 生成分区建议
            partition_suggestions = defaultdict(list)
            for i, key in enumerate(node_keys):
                partition_name = f"ml_partition_{cluster_labels[i]}"
                partition_suggestions[partition_name].append(key)
            
            return dict(partition_suggestions)
            
        except Exception as e:
            print(f"分区优化失败: {e}")
            return {}
    
    def _extract_node_features(self, node):
        """提取节点特征"""
        features = []
        
        # 基本特征
        features.append(len(node.r) if node.r else 0)  # 关系数量
        features.append(node.weight if hasattr(node, 'weight') else 1.0)  # 权重
        features.append(len(node.tags) if node.tags else 0)  # 标签数量
        features.append(len(node.metadata) if node.metadata else 0)  # 元数据数量
        
        # 访问特征
        access_count = len(self.access_history.get(node.key, []))
        features.append(access_count)
        
        # 最近访问时间
        recent_access = 0
        if node.key in self.access_history:
            recent_records = self.access_history[node.key]
            if recent_records:
                recent_access = time.time() - recent_records[-1]['timestamp']
        features.append(recent_access)
        
        # 值类型特征
        val_type = type(node.val).__name__
        type_encoding = hash(val_type) % 1000 / 1000.0
        features.append(type_encoding)
        
        # 补充特征到固定长度
        while len(features) < 10:
            features.append(0.0)
        
        return features[:10]
    
    def detect_anomalies(self):
        """检测异常访问模式"""
        if not self.enabled or len(self.access_patterns) < 100:
            return []
        
        try:
            # 构建访问特征
            features = []
            current_time = time.time()
            
            # 按时间窗口聚合访问数据
            time_windows = defaultdict(lambda: defaultdict(int))
            window_size = 300  # 5分钟窗口
            
            for record in self.access_patterns:
                window = int(record['timestamp'] // window_size)
                time_windows[window][record['node_key']] += 1
            
            # 构建特征向量
            for window, accesses in time_windows.items():
                feature_vector = [
                    len(accesses),  # 访问的不同节点数
                    sum(accesses.values()),  # 总访问次数
                    max(accesses.values()) if accesses else 0,  # 最大单节点访问次数
                    len([k for k, v in accesses.items() if v > 10])  # 高频访问节点数
                ]
                features.append(feature_vector)
            
            if len(features) < 10:
                return []
            
            features = np.array(features)
            
            # 训练异常检测模型
            self.anomaly_detector.fit(features)
            anomaly_scores = self.anomaly_detector.decision_function(features)
            
            # 识别异常
            anomalies = []
            for i, score in enumerate(anomaly_scores):
                if score < self.anomaly_threshold:
                    window_start = list(time_windows.keys())[i] * window_size
                    anomalies.append({
                        'window_start': window_start,
                        'anomaly_score': score,
                        'description': '检测到异常访问模式'
                    })
            
            return anomalies
            
        except Exception as e:
            print(f"异常检测失败: {e}")
            return []
    
    def predict_query_performance(self, query_type, context=None):
        """预测查询性能"""
        if not self.enabled or len(self.query_performance) < 50:
            return None
        
        try:
            # 准备训练数据
            X = []
            y = []
            
            for record in self.query_performance:
                features = [
                    hash(record['query_type']) % 1000,
                    record['result_count'],
                    len(record.get('context', {}))
                ]
                X.append(features)
                y.append(record['execution_time'])
            
            X = np.array(X)
            y = np.array(y)
            
            # 训练模型
            self.performance_predictor.fit(X, y)
            
            # 预测当前查询
            query_features = [
                hash(query_type) % 1000,
                0,  # 结果数量未知
                len(context or {})
            ]
            
            predicted_time = self.performance_predictor.predict([query_features])[0]
            return max(0, predicted_time)
            
        except Exception as e:
            print(f"性能预测失败: {e}")
            return None
    
    def optimize_path_finding(self, start_key, goal_key):
        """优化路径查找的启发式函数"""
        if not self.enabled:
            return None
        
        try:
            # 基于历史路径查找数据优化启发式
            start_node = self.ion.get_node_by_key(start_key)
            goal_node = self.ion.get_node_by_key(goal_key)
            
            if not start_node or not goal_node:
                return None
            
            # 计算智能启发式值
            def smart_heuristic(current, goal):
                # 基于节点特征的启发式
                current_features = self._extract_node_features(current)
                goal_features = self._extract_node_features(goal)
                
                # 计算特征距离
                feature_distance = np.linalg.norm(
                    np.array(current_features) - np.array(goal_features)
                )
                
                # 结合传统启发式
                traditional_h = 1.0  # 默认值
                
                return feature_distance * 0.7 + traditional_h * 0.3
            
            return smart_heuristic
            
        except Exception as e:
            print(f"路径优化失败: {e}")
            return None
    
    def get_intelligence_stats(self):
        """获取智能统计信息"""
        stats = {
            'enabled': self.enabled,
            'access_patterns_count': len(self.access_patterns),
            'query_performance_records': len(self.query_performance),
            'models_trained': {
                'cache_predictor': hasattr(self, 'cache_predictor') and self.cache_predictor is not None,
                'partition_optimizer': hasattr(self, 'partition_optimizer') and self.partition_optimizer is not None,
                'anomaly_detector': hasattr(self, 'anomaly_detector') and self.anomaly_detector is not None,
                'performance_predictor': hasattr(self, 'performance_predictor') and self.performance_predictor is not None,
                'io_predictor': hasattr(self, 'io_predictor') and self.io_predictor is not None
            },
            'learning_enabled': self.learning_enabled,
            'io_training_data_count': len(getattr(self, 'io_training_data', [])),
            'batch_statistics_count': len(getattr(self, 'batch_statistics', []))
        }
        
        return stats
    def incremental_learning(self, new_data_batch=None):
        """增量学习 - 使用新数据更新模型而不重新训练"""
        if not self.enabled or not self.learning_enabled:
            return False
        
        try:
            # 缓存预测的增量学习
            if len(self.access_patterns) >= 200:
                # 准备增量训练数据
                recent_patterns = self.access_patterns[-100:]  # 最近100个访问模式
                
                # 构建特征和标签
                X_new = []
                y_new = []
                
                for pattern in recent_patterns:
                    node = self.ion.get_node_by_key(pattern['node_key'])
                    if node:
                        features = self._extract_node_features(node)
                        # 预测目标：访问频率
                        access_freq = len(self.access_history[pattern['node_key']]) / (time.time() - pattern['timestamp'] + 1)
                        X_new.append(features)
                        y_new.append(access_freq)
                
                if len(X_new) >= 10:
                    X_new = np.array(X_new)
                    y_new = np.array(y_new)
                    # 设置模型训练标志
                    if hasattr(self, 'cache_predictor') and self.cache_predictor is not None:
                        self.cache_model = self.cache_predictor  # 为了兼容性
                    if hasattr(self, 'mini_batch_kmeans'):
                        self.partition_optimizer = self.mini_batch_kmeans
                        self.partition_model = self.mini_batch_kmeans  # 为了兼容性
                    # 使用partial_fit进行增量学习（如果模型支持）
                    if hasattr(self.cache_predictor, 'partial_fit'):
                        self.cache_predictor.partial_fit(X_new, y_new)
                    else:
                        # 重新训练但只使用最近数据
                        self.cache_predictor.fit(X_new, y_new)
            
            # 分区优化的增量学习
            if len(self.access_patterns) >= 300:
                # 使用在线K-means更新聚类中心
                recent_keys = list(set(p['node_key'] for p in self.access_patterns[-150:]))
                if len(recent_keys) >= 10:
                    features = []
                    for key in recent_keys:
                        node = self.ion.get_node_by_key(key)
                        if node:
                            features.append(self._extract_node_features(node))
                    
                    if len(features) >= 5:
                        features = np.array(features)
                        # 使用mini-batch K-means进行增量更新
                        from sklearn.cluster import MiniBatchKMeans
                        if not hasattr(self, 'mini_batch_kmeans'):
                            self.mini_batch_kmeans = MiniBatchKMeans(
                                n_clusters=min(5, len(features)//2),
                                random_state=42,
                                batch_size=min(10, len(features))
                            )
                        self.mini_batch_kmeans.partial_fit(features)
            
            return True
            
        except Exception as e:
            print(f"增量学习失败: {e}")
            return False
    def train_io_predictor(self):
        """训练输入输出预测器"""
        if not hasattr(self, 'io_training_data') or len(self.io_training_data) < 10:
            return False
        
        try:
            import pandas as pd
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import LabelEncoder
            
            # 准备训练数据
            data = []
            for record in self.io_training_data:
                # 处理ML拦截器记录的简化格式
                if 'method_name' in record:
                    # 新的简化格式
                    row = {
                        'method': record.get('method_name', 'unknown'),
                        'access_type': record.get('access_type', 'unknown'),
                        'args_count': record.get('args_count', 0),
                        'kwargs_count': record.get('kwargs_count', 0),
                        'execution_time': record.get('execution_time', 0.0),
                        'result_count': record.get('result_size', 0),
                        'timestamp': record.get('timestamp', 0)
                    }
                else:
                    # 原有的复杂格式（向后兼容）
                    input_features = record.get('input_features', {})
                    output_features = record.get('output_features', {})
                    
                    row = {
                        'method': record.get('method', 'unknown'),
                        'access_type': record.get('access_type', 'unknown'),
                        'args_count': input_features.get('args_count', 0),
                        'kwargs_count': input_features.get('kwargs_count', 0),
                        'execution_time': record.get('execution_time', 0.0),
                        'result_count': output_features.get('result_count', 0),
                        'timestamp': record.get('timestamp', 0)
                    }
                
                data.append(row)
            
            if not data:
                return False
            
            # 创建DataFrame
            df = pd.DataFrame(data)
            
            # 编码分类特征
            le_method = LabelEncoder()
            le_access = LabelEncoder()
            
            df['method_encoded'] = le_method.fit_transform(df['method'])
            df['access_type_encoded'] = le_access.fit_transform(df['access_type'])
            
            # 选择特征和目标
            feature_columns = ['method_encoded', 'access_type_encoded', 'args_count', 'kwargs_count']
            available_features = [col for col in feature_columns if col in df.columns]
            
            if len(available_features) < 2:
                print(f"特征列不足: {available_features}")
                return False
            
            X = df[available_features]
            y = df['execution_time']
            
            # 训练模型
            self.io_predictor = RandomForestRegressor(n_estimators=50, random_state=42)
            self.io_predictor.fit(X, y)
            
            # 保存编码器
            self.method_encoder = le_method
            self.access_encoder = le_access
            self.io_feature_columns = available_features
            
            return True
            
        except Exception as e:
            print(f"IO预测器训练失败: {e}")
            return False
    
    def predict_method_performance(self, method_name, args_count, kwargs_count, total_input_size, access_type):
        """预测方法性能"""
        if not hasattr(self, 'io_predictor'):
            return None
        
        try:
            # 编码输入
            method_encoded = self.method_encoder.transform([method_name])[0] if method_name in self.method_encoder.classes_ else 0
            access_encoded = self.access_type_encoder.transform([access_type])[0] if access_type in self.access_type_encoder.classes_ else 0
            
            # 预测
            # 构建特征向量，只使用训练时可用的特征
            feature_values = []
            if 'args_count' in self.io_feature_cols:
                feature_values.append(args_count)
            if 'kwargs_count' in self.io_feature_cols:
                feature_values.append(kwargs_count)
            if 'total_input_size' in self.io_feature_cols:
                feature_values.append(total_input_size)
            if 'method_encoded' in self.io_feature_cols:
                feature_values.append(method_encoded)
            if 'access_type_encoded' in self.io_feature_cols:
                feature_values.append(access_encoded)

            X = [feature_values]
            prediction = self.io_predictor.predict(X)[0]
            
            return {
                'predicted_execution_time': prediction[0],
                'predicted_result_count': prediction[1]
            }
        except Exception as e:
            print(f"性能预测失败: {e}")
            return None
    def auto_hyperparameter_tuning(self):
        """自动超参数调优"""
        if not self.enabled or len(self.query_performance) < 100:
            return {}
        
        try:
            from sklearn.model_selection import GridSearchCV
            from sklearn.metrics import make_scorer
            
            # 准备数据
            X = []
            y = []
            for record in self.query_performance[-200:]:  # 使用最近200条记录
                features = [
                    hash(record['query_type']) % 1000,
                    record['result_count'],
                    len(record.get('context', {}))
                ]
                X.append(features)
                y.append(record['execution_time'])
            
            if len(X) < 50:
                return {}
            
            X = np.array(X)
            y = np.array(y)
            
            # 为MLPRegressor调优超参数
            param_grid = {
                'hidden_layer_sizes': [(30, 20), (50, 30), (100, 50)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'max_iter': [300, 500, 1000]
            }
            
            # 使用网格搜索
            grid_search = GridSearchCV(
                MLPRegressor(random_state=42),
                param_grid,
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            grid_search.fit(X, y)
            
            # 更新模型
            self.cache_predictor = grid_search.best_estimator_
            
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            print(f"超参数调优完成 - 最佳参数: {best_params}, 最佳得分: {best_score}")
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'tuning_completed': True
            }
            
        except Exception as e:
            print(f"超参数调优失败: {e}")
            return {'tuning_completed': False, 'error': str(e)}
    
    def adaptive_learning_schedule(self):
        """自适应学习调度 - 根据数据量和性能自动调整学习频率"""
        current_time = time.time()
        
        # 根据访问模式数量决定学习频率
        if len(self.access_patterns) < 100:
            return False  # 数据太少，不进行学习
        elif len(self.access_patterns) < 500:
            # 每收集50个新模式就进行一次增量学习
            if len(self.access_patterns) % 50 == 0:
                return self.incremental_learning()
        elif len(self.access_patterns) < 2000:
            # 每收集100个新模式就进行一次增量学习
            if len(self.access_patterns) % 100 == 0:
                return self.incremental_learning()
        else:
            # 数据量大时，每收集200个新模式进行一次学习
            if len(self.access_patterns) % 200 == 0:
                return self.incremental_learning()
        
        # 每1000个查询性能记录进行一次超参数调优
        if len(self.query_performance) % 1000 == 0 and len(self.query_performance) > 0:
            return self.auto_hyperparameter_tuning()
        
        return False
        def enable_learning(self, enabled=True):
            """启用/禁用学习功能"""
            self.learning_enabled = enabled
        
        def clear_learning_data(self):
            """清除学习数据"""
            self.access_patterns.clear()
            self.access_history.clear()
            self.query_performance.clear()
            self.node_features.clear()
            self.query_features.clear()
# 同步系统基础类和接口
class SyncOperation(Enum):
    """同步操作类型枚举"""
    CREATE = 'create'           # 创建节点
    UPDATE = 'update'           # 更新节点
    DELETE = 'delete'           # 删除节点
    ADD_RELATION = 'add_relation'       # 添加关系
    REMOVE_RELATION = 'remove_relation' # 移除关系
    BATCH_CREATE = 'batch_create'       # 批量创建
    BATCH_UPDATE = 'batch_update'       # 批量更新
    BATCH_DELETE = 'batch_delete'       # 批量删除

class SyncEvent:
    """同步事件类，包含操作信息"""
    def __init__(self, operation, node=None, old_value=None, new_value=None, 
                 metadata=None, relations=None, context=None):
        self.operation = operation      # 操作类型
        self.node = node               # 相关节点
        self.old_value = old_value     # 旧值
        self.new_value = new_value     # 新值
        self.metadata = metadata       # 元数据
        self.relations = relations     # 关系信息
        self.context = context         # 上下文信息
        self.timestamp = time.time()   # 时间戳
        self.event_id = str(uuid.uuid4())  # 事件ID

class ISyncHandler:
    """同步处理器接口，用户需要实现这个接口"""
    
    def on_node_create(self, event: SyncEvent):
        """节点创建时调用"""
        pass
    
    def on_node_update(self, event: SyncEvent):
        """节点更新时调用"""
        pass
    
    def on_node_delete(self, event: SyncEvent):
        """节点删除时调用"""
        pass
    
    def on_relation_add(self, event: SyncEvent):
        """关系添加时调用"""
        pass
    
    def on_relation_remove(self, event: SyncEvent):
        """关系移除时调用"""
        pass
    
    def on_batch_operation(self, events: List[SyncEvent]):
        """批量操作时调用"""
        pass
    
    def on_transaction_commit(self, transaction_id: str, events: List[SyncEvent]):
        """事务提交时调用"""
        pass
    
    def on_transaction_rollback(self, transaction_id: str, events: List[SyncEvent]):
        """事务回滚时调用"""
        pass

class LinkedListSyncHandler(ISyncHandler):
    """链表同步处理器示例实现"""
    
    def __init__(self):
        self.linked_list = []
        self.node_positions = {}  # 节点键 -> 位置映射
        self.lock = threading.RLock()
    
    def on_node_create(self, event: SyncEvent):
        """将新节点添加到链表末尾"""
        with self.lock:
            node = event.node
            self.linked_list.append(node)
            self.node_positions[node.key] = len(self.linked_list) - 1
            print(f"LinkedList: 添加节点 {node.key} 到位置 {len(self.linked_list) - 1}")
    
    def on_node_update(self, event: SyncEvent):
        """更新链表中的节点"""
        with self.lock:
            node = event.node
            if node.key in self.node_positions:
                pos = self.node_positions[node.key]
                self.linked_list[pos] = node
                print(f"LinkedList: 更新位置 {pos} 的节点 {node.key}")
    
    def on_node_delete(self, event: SyncEvent):
        """从链表中删除节点"""
        with self.lock:
            node = event.node
            if node.key in self.node_positions:
                pos = self.node_positions[node.key]
                del self.linked_list[pos]
                del self.node_positions[node.key]
                
                # 更新后续节点的位置映射
                for i in range(pos, len(self.linked_list)):
                    node_at_pos = self.linked_list[i]
                    self.node_positions[node_at_pos.key] = i
                
                print(f"LinkedList: 删除节点 {node.key}")
    
    def get_ordered_nodes(self):
        """获取按插入顺序排列的节点"""
        with self.lock:
            return self.linked_list.copy()
    
    def find_node_position(self, key):
        """查找节点在链表中的位置"""
        with self.lock:
            return self.node_positions.get(key, -1)

class TreeSyncHandler(ISyncHandler):
    """树结构同步处理器示例实现"""
    
    def __init__(self):
        self.tree = {}  # 父节点 -> 子节点列表
        self.parent_map = {}  # 子节点 -> 父节点
        self.lock = threading.RLock()
    
    def on_node_create(self, event: SyncEvent):
        """将节点添加到树结构"""
        with self.lock:
            node = event.node
            parent_key = event.context.get('parent_key') if event.context else None
            
            if parent_key:
                # 添加为子节点
                if parent_key not in self.tree:
                    self.tree[parent_key] = []
                self.tree[parent_key].append(node.key)
                self.parent_map[node.key] = parent_key
                print(f"Tree: 添加节点 {node.key} 作为 {parent_key} 的子节点")
            else:
                # 添加为根节点
                if 'roots' not in self.tree:
                    self.tree['roots'] = []
                self.tree['roots'].append(node.key)
                print(f"Tree: 添加根节点 {node.key}")
    
    def on_node_delete(self, event: SyncEvent):
        """从树结构中删除节点"""
        with self.lock:
            node = event.node
            key = node.key
            
            # 删除父子关系
            if key in self.parent_map:
                parent_key = self.parent_map[key]
                if parent_key in self.tree:
                    self.tree[parent_key].remove(key)
                del self.parent_map[key]
            elif 'roots' in self.tree and key in self.tree['roots']:
                self.tree['roots'].remove(key)
            
            # 删除子节点关系
            if key in self.tree:
                del self.tree[key]
            
            print(f"Tree: 删除节点 {key}")
    
    def get_children(self, parent_key):
        """获取指定节点的子节点"""
        with self.lock:
            return self.tree.get(parent_key, []).copy()
    
    def get_parent(self, child_key):
        """获取指定节点的父节点"""
        with self.lock:
            return self.parent_map.get(child_key)
    
    def get_roots(self):
        """获取所有根节点"""
        with self.lock:
            return self.tree.get('roots', []).copy()

class GraphSyncHandler(ISyncHandler):
    """图结构同步处理器示例实现"""
    
    def __init__(self):
        self.adjacency_list = {}  # 邻接表表示
        self.edge_weights = {}    # 边权重
        self.lock = threading.RLock()
    
    def on_node_create(self, event: SyncEvent):
        """添加节点到图"""
        with self.lock:
            node = event.node
            if node.key not in self.adjacency_list:
                self.adjacency_list[node.key] = set()
                print(f"Graph: 添加节点 {node.key}")
    
    def on_node_delete(self, event: SyncEvent):
        """从图中删除节点"""
        with self.lock:
            node = event.node
            key = node.key
            
            # 删除所有相关的边
            if key in self.adjacency_list:
                # 删除出边
                for neighbor in self.adjacency_list[key]:
                    if neighbor in self.adjacency_list:
                        self.adjacency_list[neighbor].discard(key)
                    # 删除边权重
                    edge_key = f"{key}-{neighbor}"
                    self.edge_weights.pop(edge_key, None)
                    edge_key = f"{neighbor}-{key}"
                    self.edge_weights.pop(edge_key, None)
                
                del self.adjacency_list[key]
                print(f"Graph: 删除节点 {key}")
    
    def on_relation_add(self, event: SyncEvent):
        """添加边到图"""
        with self.lock:
            relations = event.relations
            if relations:
                source_key = relations.get('source')
                target_key = relations.get('target')
                weight = relations.get('weight', 1.0)
                
                if source_key and target_key:
                    # 确保节点存在
                    if source_key not in self.adjacency_list:
                        self.adjacency_list[source_key] = set()
                    if target_key not in self.adjacency_list:
                        self.adjacency_list[target_key] = set()
                    
                    # 添加边
                    self.adjacency_list[source_key].add(target_key)
                    edge_key = f"{source_key}-{target_key}"
                    self.edge_weights[edge_key] = weight
                    
                    print(f"Graph: 添加边 {source_key} -> {target_key} (权重: {weight})")
    
    def on_relation_remove(self, event: SyncEvent):
        """从图中删除边"""
        with self.lock:
            relations = event.relations
            if relations:
                source_key = relations.get('source')
                target_key = relations.get('target')
                
                if source_key and target_key:
                    if source_key in self.adjacency_list:
                        self.adjacency_list[source_key].discard(target_key)
                    
                    edge_key = f"{source_key}-{target_key}"
                    self.edge_weights.pop(edge_key, None)
                    
                    print(f"Graph: 删除边 {source_key} -> {target_key}")
    
    def get_neighbors(self, node_key):
        """获取节点的邻居"""
        with self.lock:
            return list(self.adjacency_list.get(node_key, set()))
    
    def get_edge_weight(self, source, target):
        """获取边的权重"""
        with self.lock:
            edge_key = f"{source}-{target}"
            return self.edge_weights.get(edge_key, None)
    
    def get_all_edges(self):
        """获取所有边"""
        with self.lock:
            edges = []
            for source, targets in self.adjacency_list.items():
                for target in targets:
                    edge_key = f"{source}-{target}"
                    weight = self.edge_weights.get(edge_key, 1.0)
                    edges.append((source, target, weight))
            return edges

class SyncManager:
    """同步管理器，管理所有同步处理器"""
    
    def __init__(self):
        self.handlers = {}  # 处理器名称 -> 处理器实例
        self.event_queue = deque()  # 事件队列
        self.async_mode = False  # 是否异步处理
        self.worker_thread = None
        self.running = False
        self.lock = threading.RLock()
        self.batch_size = 100  # 批处理大小
        self.batch_timeout = 1.0  # 批处理超时时间（秒）
    
    def register_handler(self, name: str, handler: ISyncHandler):
        """注册同步处理器"""
        with self.lock:
            self.handlers[name] = handler
            print(f"SyncManager: 注册处理器 '{name}'")
    
    def unregister_handler(self, name: str):
        """注销同步处理器"""
        with self.lock:
            if name in self.handlers:
                del self.handlers[name]
                print(f"SyncManager: 注销处理器 '{name}'")
    
    def get_handler(self, name: str) -> ISyncHandler:
        """获取指定的同步处理器"""
        with self.lock:
            return self.handlers.get(name)
    
    def list_handlers(self) -> List[str]:
        """列出所有注册的处理器名称"""
        with self.lock:
            return list(self.handlers.keys())
    
    def set_async_mode(self, async_mode: bool):
        """设置是否异步处理事件"""
        with self.lock:
            self.async_mode = async_mode
            if async_mode and not self.running:
                self.start_async_processing()
            elif not async_mode and self.running:
                self.stop_async_processing()
    
    def start_async_processing(self):
        """启动异步事件处理"""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._process_events_async, daemon=True)
            self.worker_thread.start()
            print("SyncManager: 启动异步事件处理")
    
    def stop_async_processing(self):
        """停止异步事件处理"""
        if self.running:
            self.running = False
            if self.worker_thread:
                self.worker_thread.join(timeout=5.0)
            print("SyncManager: 停止异步事件处理")
    
    def trigger_event(self, event: SyncEvent):
        """触发同步事件"""
        if self.async_mode:
            with self.lock:
                self.event_queue.append(event)
        else:
            self._process_event_sync(event)
    
    def trigger_batch_events(self, events: List[SyncEvent]):
        """触发批量同步事件"""
        if self.async_mode:
            with self.lock:
                self.event_queue.extend(events)
        else:
            self._process_batch_events_sync(events)
    
    def _process_event_sync(self, event: SyncEvent):
        """同步处理单个事件"""
        with self.lock:
            for name, handler in self.handlers.items():
                try:
                    if event.operation == SyncOperation.CREATE:
                        handler.on_node_create(event)
                    elif event.operation == SyncOperation.UPDATE:
                        handler.on_node_update(event)
                    elif event.operation == SyncOperation.DELETE:
                        handler.on_node_delete(event)
                    elif event.operation == SyncOperation.ADD_RELATION:
                        handler.on_relation_add(event)
                    elif event.operation == SyncOperation.REMOVE_RELATION:
                        handler.on_relation_remove(event)
                except Exception as e:
                    print(f"SyncManager: 处理器 '{name}' 处理事件时出错: {e}")
    
    def _process_batch_events_sync(self, events: List[SyncEvent]):
        """同步处理批量事件"""
        with self.lock:
            for name, handler in self.handlers.items():
                try:
                    handler.on_batch_operation(events)
                    # 也逐个处理事件
                    for event in events:
                        self._process_single_handler_event(handler, event)
                except Exception as e:
                    print(f"SyncManager: 处理器 '{name}' 处理批量事件时出错: {e}")
    
    def _process_single_handler_event(self, handler: ISyncHandler, event: SyncEvent):
        """为单个处理器处理单个事件"""
        try:
            if event.operation == SyncOperation.CREATE:
                handler.on_node_create(event)
            elif event.operation == SyncOperation.UPDATE:
                handler.on_node_update(event)
            elif event.operation == SyncOperation.DELETE:
                handler.on_node_delete(event)
            elif event.operation == SyncOperation.ADD_RELATION:
                handler.on_relation_add(event)
            elif event.operation == SyncOperation.REMOVE_RELATION:
                handler.on_relation_remove(event)
        except Exception as e:
            print(f"SyncManager: 处理事件时出错: {e}")
    
    def _process_events_async(self):
        """异步事件处理工作线程"""
        batch = []
        last_batch_time = time.time()
        
        while self.running:
            try:
                # 收集事件到批次
                current_time = time.time()
                
                with self.lock:
                    while self.event_queue and len(batch) < self.batch_size:
                        batch.append(self.event_queue.popleft())
                
                # 检查是否需要处理批次
                should_process = (
                    len(batch) >= self.batch_size or
                    (batch and current_time - last_batch_time >= self.batch_timeout)
                )
                
                if should_process and batch:
                    self._process_batch_events_sync(batch)
                    batch.clear()
                    last_batch_time = current_time
                
                # 短暂休眠
                time.sleep(0.01)
                
            except Exception as e:
                print(f"SyncManager: 异步处理线程出错: {e}")
                time.sleep(0.1)
    
    def get_queue_size(self) -> int:
        """获取事件队列大小"""
        with self.lock:
            return len(self.event_queue)
    
    def clear_queue(self):
        """清空事件队列"""
        with self.lock:
            self.event_queue.clear()
            print("SyncManager: 清空事件队列")

# 事务状态枚举类
class TransactionState(Enum):
    """事务状态枚举类"""
    ACTIVE = 'active'       # 活跃状态
    COMMITTED = 'committed' # 已提交
    ABORTED = 'aborted'     # 已中止
    PREPARED = 'prepared'   # 准备好提交（用于两阶段提交）

# 锁类型枚举类
class LockType(Enum):
    """锁类型枚举类"""
    SHARED = 'shared'           # 共享锁（读锁）
    EXCLUSIVE = 'exclusive'     # 排他锁（写锁）
    INTENT_SHARED = 'IS'        # 意向共享锁
    INTENT_EXCLUSIVE = 'IX'     # 意向排他锁
    
# 事务隔离级别枚举类
class IsolationLevel(Enum):
    """事务隔离级别枚举类"""
    READ_UNCOMMITTED = 'read_uncommitted'  # 读未提交
    READ_COMMITTED = 'read_committed'      # 读已提交
    REPEATABLE_READ = 'repeatable_read'    # 可重复读
    SERIALIZABLE = 'serializable'          # 串行化

# 事务日志条目类
class TransactionLogEntry:
    """事务日志条目，记录事务操作以便回滚或重做"""
    def __init__(self, operation, node_key=None, old_value=None, new_value=None, 
                 metadata=None, tags=None, relationships=None):
        self.operation = operation        # 操作类型
        self.node_key = node_key          # 节点键
        self.old_value = old_value        # 旧值
        self.new_value = new_value        # 新值
        self.metadata = metadata          # 元数据
        self.tags = tags                  # 标签
        self.relationships = relationships # 关系
        self.timestamp = time.time()      # 时间戳
        
    def __str__(self):
        return f"TransactionLogEntry({self.operation}, {self.node_key}, ts={self.timestamp})"

# 锁管理器类
class LockManager:
    """锁管理器，负责管理节点和关系的锁"""
    def __init__(self):
        self.locks = {}  # 资源ID -> {事务ID -> 锁类型}
        self.lock = threading.RLock()  # 保护锁表的锁
        self.wait_for_graph = defaultdict(set)  # 事务等待图，用于死锁检测
        self.timeout = 10  # 默认超时时间（秒）
        
    def acquire_lock(self, txn_id, resource_id, lock_type, timeout=None):
        """
        为事务获取指定资源的锁
        
        参数:
            txn_id: 事务ID
            resource_id: 资源ID（如节点键或关系ID）
            lock_type: 锁类型
            timeout: 超时时间，None表示使用默认值
            
        返回:
            bool: 是否成功获取锁
            
        异常:
            TimeoutError: 如果在规定时间内无法获取锁
            DeadlockError: 如果检测到死锁
        """
        if timeout is None:
            timeout = self.timeout
            
        start_time = time.time()
        
        while True:
            with self.lock:
                # 如果资源没有被锁定，直接获取锁
                if resource_id not in self.locks:
                    self.locks[resource_id] = {txn_id: lock_type}
                    return True
                
                # 检查是否兼容
                current_locks = self.locks[resource_id]
                
                # 如果当前事务已经持有该资源的锁
                if txn_id in current_locks:
                    current_lock = current_locks[txn_id]
                    
                    # 如果已经持有更强的锁，直接返回成功
                    if self._is_stronger_lock(current_lock, lock_type):
                        return True
                    
                    # 如果需要升级锁，检查是否可以升级
                    if self._can_upgrade(resource_id, txn_id, lock_type):
                        current_locks[txn_id] = lock_type
                        return True
                    
                # 检查是否与其他事务的锁兼容
                if self._is_compatible(resource_id, txn_id, lock_type):
                    self.locks[resource_id][txn_id] = lock_type
                    return True
                
                # 更新等待图
                for other_txn_id in current_locks:
                    if other_txn_id != txn_id:
                        self.wait_for_graph[txn_id].add(other_txn_id)
                
                # 检测死锁
                if self._detect_deadlock(txn_id):
                    # 清理等待关系
                    if txn_id in self.wait_for_graph:
                        del self.wait_for_graph[txn_id]
                    raise DeadlockError(f"检测到死锁: 事务 {txn_id}")
            
            # 检查是否超时
            if time.time() - start_time > timeout:
                raise TimeoutError(f"获取锁超时: 资源 {resource_id}, 事务 {txn_id}")
            
            # 短暂睡眠后重试
            time.sleep(0.01)
    
    def release_lock(self, txn_id, resource_id=None):
        """
        释放事务持有的锁
        
        参数:
            txn_id: 事务ID
            resource_id: 资源ID，None表示释放该事务的所有锁
            
        返回:
            bool: 是否成功释放锁
        """
        with self.lock:
            # 释放特定资源的锁
            if resource_id is not None:
                if resource_id in self.locks and txn_id in self.locks[resource_id]:
                    del self.locks[resource_id][txn_id]
                    if not self.locks[resource_id]:  # 如果没有其他事务持有该资源的锁
                        del self.locks[resource_id]
                    
                    # 清理等待图
                    if txn_id in self.wait_for_graph:
                        del self.wait_for_graph[txn_id]
                    return True
                return False
            
            # 释放事务的所有锁
            resources_to_check = list(self.locks.keys())
            for res_id in resources_to_check:
                if txn_id in self.locks[res_id]:
                    del self.locks[res_id][txn_id]
                    if not self.locks[res_id]:
                        del self.locks[res_id]
            
            # 清理等待图
            if txn_id in self.wait_for_graph:
                del self.wait_for_graph[txn_id]
                
            # 清除其他事务对该事务的等待
            for waiting_txn in list(self.wait_for_graph.keys()):
                if txn_id in self.wait_for_graph[waiting_txn]:
                    self.wait_for_graph[waiting_txn].remove(txn_id)
                    if not self.wait_for_graph[waiting_txn]:
                        del self.wait_for_graph[waiting_txn]
            
            return True
    
    def _is_compatible(self, resource_id, txn_id, requested_lock):
        """检查请求的锁是否与当前持有的锁兼容"""
        current_locks = self.locks[resource_id]
        
        # 检查每个已存在的锁
        for other_txn_id, other_lock in current_locks.items():
            if other_txn_id == txn_id:
                continue  # 跳过自己持有的锁
                
            # 判断锁兼容性
            if not self._are_locks_compatible(requested_lock, other_lock):
                return False
                
        return True
    
    def _are_locks_compatible(self, lock1, lock2):
        """判断两种锁类型是否兼容"""
        # 兼容性矩阵
        compatibility = {
            LockType.SHARED: {
                LockType.SHARED: True,
                LockType.EXCLUSIVE: False,
                LockType.INTENT_SHARED: True,
                LockType.INTENT_EXCLUSIVE: False
            },
            LockType.EXCLUSIVE: {
                LockType.SHARED: False,
                LockType.EXCLUSIVE: False,
                LockType.INTENT_SHARED: False,
                LockType.INTENT_EXCLUSIVE: False
            },
            LockType.INTENT_SHARED: {
                LockType.SHARED: True,
                LockType.EXCLUSIVE: False,
                LockType.INTENT_SHARED: True,
                LockType.INTENT_EXCLUSIVE: True
            },
            LockType.INTENT_EXCLUSIVE: {
                LockType.SHARED: False,
                LockType.EXCLUSIVE: False,
                LockType.INTENT_SHARED: True,
                LockType.INTENT_EXCLUSIVE: True
            }
        }
        
        return compatibility[lock1][lock2]
    
    def _is_stronger_lock(self, current_lock, requested_lock):
        """判断当前持有的锁是否比请求的锁更强"""
        strength = {
            LockType.SHARED: 1,
            LockType.INTENT_SHARED: 2,
            LockType.INTENT_EXCLUSIVE: 3,
            LockType.EXCLUSIVE: 4
        }
        
        return strength[current_lock] >= strength[requested_lock]
    
    def _can_upgrade(self, resource_id, txn_id, new_lock_type):
        """检查是否可以升级锁"""
        current_locks = self.locks[resource_id]
        
        # 如果只有当前事务持有锁，则可以升级
        if len(current_locks) == 1 and txn_id in current_locks:
            current_locks[txn_id] = new_lock_type
            return True
            
        # 检查锁升级是否与其他事务的锁兼容
        for other_txn_id, other_lock in current_locks.items():
            if other_txn_id == txn_id:
                continue
                
            if not self._are_locks_compatible(new_lock_type, other_lock):
                return False
                
        current_locks[txn_id] = new_lock_type
        return True
    
    def _detect_deadlock(self, txn_id):
        """使用等待图检测死锁"""
        visited = set()
        
        def dfs(current_txn):
            visited.add(current_txn)
            
            # 检查是否形成环
            if current_txn in self.wait_for_graph:
                for waiting_for_txn in self.wait_for_graph[current_txn]:
                    if waiting_for_txn == txn_id:
                        return True  # 找到环，即死锁
                    if waiting_for_txn not in visited:
                        if dfs(waiting_for_txn):
                            return True
            return False
            
        return dfs(txn_id)
    
    def get_locks_by_transaction(self, txn_id):
        """获取事务持有的所有锁"""
        result = {}
        
        with self.lock:
            for resource_id, locks in self.locks.items():
                if txn_id in locks:
                    result[resource_id] = locks[txn_id]
                    
        return result
    
    def get_transactions_by_resource(self, resource_id):
        """获取持有特定资源锁的所有事务"""
        with self.lock:
            if resource_id in self.locks:
                return dict(self.locks[resource_id])
            return {}
    
    def detect_all_deadlocks(self):
        """
        检测系统中所有的死锁
        
        返回:
            list: 死锁事务循环列表
        """
        deadlocks = []
        
        with self.lock:
            # 复制等待图以避免在检测过程中修改
            wait_graph = {txn_id: set(waiting_for) 
                        for txn_id, waiting_for in self.wait_for_graph.items()}
            
            # 检查所有事务
            for txn_id in wait_graph:
                cycle = self._find_deadlock_cycle(txn_id, wait_graph)
                if cycle and cycle not in deadlocks:
                    deadlocks.append(cycle)
                    
        return deadlocks
    
    def _find_deadlock_cycle(self, start_txn, wait_graph=None):
        """
        查找从指定事务开始的死锁循环
        
        参数:
            start_txn: 起始事务ID
            wait_graph: 等待图，如果为None则使用当前等待图
            
        返回:
            list: 死锁循环的事务ID列表，如果没有死锁则返回None
        """
        if wait_graph is None:
            wait_graph = self.wait_for_graph
            
        if start_txn not in wait_graph:
            return None
            
        # 使用DFS查找循环
        visited = set()
        path = []
        
        def dfs(txn_id):
            visited.add(txn_id)
            path.append(txn_id)
            
            if txn_id in wait_graph:
                for waiting_for in wait_graph[txn_id]:
                    if waiting_for == start_txn:
                        # 找到循环，返回完整路径
                        return path + [start_txn]
                    
                    if waiting_for not in visited:
                        result = dfs(waiting_for)
                        if result:
                            return result
            
            # 回溯
            path.pop()
            return None
            
        return dfs(start_txn)


# 自定义异常类
class TransactionError(Exception):
    """事务相关异常的基类"""
    pass

class DeadlockError(TransactionError):
    """死锁异常"""
    pass

class TimeoutError(TransactionError):
    """超时异常"""
    pass

class ConcurrencyError(TransactionError):
    """并发控制异常"""
    pass

class TransactionAbortError(TransactionError):
    """事务中止异常"""
    pass

# 事务类
class Transaction:
    """表示数据库事务的类"""
    def __init__(self, ion, isolation_level=IsolationLevel.SERIALIZABLE, timeout=10):
        self.id = str(uuid.uuid4())  # 唯一事务ID
        self.ion = weakref.proxy(ion)  # 避免循环引用
        self.state = TransactionState.ACTIVE  # 初始状态为活跃
        self.isolation_level = isolation_level  # 隔离级别
        self.log = []  # 事务日志
        self.timeout = timeout  # 锁超时时间
        self.start_time = time.time()  # 开始时间
        self.commit_time = None  # 提交时间
        self.modified_nodes = set()  # 修改过的节点
        self.read_set = set()  # 读取过的节点
        
    def add_log_entry(self, operation, node_key=None, old_value=None, new_value=None, 
                      metadata=None, tags=None, relationships=None):
        """添加日志条目"""
        entry = TransactionLogEntry(
            operation, node_key, old_value, new_value, metadata, tags, relationships
        )
        self.log.append(entry)
        
        # 根据操作类型记录修改或读取
        if operation in ('create', 'update', 'delete'):
            if node_key is not None:
                self.modified_nodes.add(node_key)
        elif operation == 'read':
            if node_key is not None:
                self.read_set.add(node_key)
                
        return entry
        
    def commit(self):
        """提交事务"""
        if self.state != TransactionState.ACTIVE:
            raise TransactionError(f"无法提交非活跃事务: 当前状态={self.state}")
            
        try:
            # 在提交之前执行任何必要的检查
            self.ion._check_transaction_conflicts(self)
            
            # 提交更改
            self.state = TransactionState.COMMITTED
            self.commit_time = time.time()
            
            # 释放所有锁
            self.ion._release_transaction_locks(self.id)
            
            # 触发回调
            self.ion._trigger_event('transaction_committed', transaction=self)
            
            return True
        except Exception as e:
            # 提交失败，回滚事务
            self.abort(reason=str(e))
            raise
            
    def abort(self, reason=None):
        """中止事务并回滚更改"""
        if self.state == TransactionState.COMMITTED:
            raise TransactionError("无法中止已提交的事务")
            
        try:
            # 回滚所有更改
            self._rollback()
            
            # 更新状态
            self.state = TransactionState.ABORTED
            
            # 释放所有锁
            self.ion._release_transaction_locks(self.id)
            
            # 触发回调
            self.ion._trigger_event('transaction_aborted', transaction=self, reason=reason)
            
            return True
        except Exception as e:
            logging.error(f"事务回滚失败: {e}")
            # 即使回滚失败，仍然需要释放锁
            try:
                self.ion._release_transaction_locks(self.id)
            except:
                pass
            raise
            
    def _rollback(self):
        """回滚事务中的更改"""
        # 倒序遍历日志条目进行回滚
        for entry in reversed(self.log):
            try:
                if entry.operation == 'create':
                    # 删除创建的节点
                    if entry.node_key is not None:
                        self.ion._rollback_create_node(entry.node_key)
                        
                elif entry.operation == 'update':
                    # 恢复更新前的值
                    if entry.node_key is not None and entry.old_value is not None:
                        self.ion._rollback_update_node(entry.node_key, entry.old_value, 
                                                      entry.metadata, entry.tags, entry.relationships)
                        
                elif entry.operation == 'delete':
                    # 恢复删除的节点
                    if entry.node_key is not None and entry.old_value is not None:
                        self.ion._rollback_delete_node(entry.node_key, entry.old_value, 
                                                      entry.metadata, entry.tags, entry.relationships)
                        
                elif entry.operation == 'add_relationship':
                    # 删除添加的关系
                    if entry.relationships is not None:
                        self.ion._rollback_add_relationship(entry.relationships)
                        
                elif entry.operation == 'remove_relationship':
                    # 恢复删除的关系
                    if entry.relationships is not None:
                        self.ion._rollback_remove_relationship(entry.relationships)
            except Exception as e:
                logging.error(f"回滚操作 {entry.operation} 失败: {e}")
                
    def __enter__(self):
        """支持使用with语句"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出with块时自动提交或回滚事务"""
        if exc_type is None:
            # 如果没有异常，提交事务
            self.commit()
        else:
            # 如果有异常，回滚事务
            self.abort(reason=str(exc_val))
            
    def __str__(self):
        return f"Transaction(id={self.id}, state={self.state}, modified={len(self.modified_nodes)})"

class ION(metaclass=MLInterceptorMeta):
    """
    IntegratedObjectNetwork类 - 整合OND和ICombinedDataStructure的功能
    提供更高级的对象网络结构和操作能力
    """
    
    class IONNode:
        """节点类，继承和扩展RNode的功能"""
        def __init__(self, key, val, metadata=None, tags=None, weight=1.0, row=None):
            self._key = key
            self._val = val
            self._r = []  # 关系列表
            self._metadata = metadata or {}
            self._weight = float(weight)  # 节点权重
            self._visited = False  # 用于遍历
            self._created_at = time.time()  # 创建时间
            self._updated_at = time.time()  # 更新时间
            self._ion_reference = None  # 引用所属的ION实例
            self._partition = None  # 数据分区标识
            self._row = row  # 表行标识，用于表查找功能
            
            # 确保标签被正确处理
            if tags is not None:
                if isinstance(tags, str):
                    # 如果是字符串，视为单个标签
                    self._tags = {tags}
                else:
                    # 否则尝试转换为集合
                    self._tags = set(tags)
            else:
                self._tags = set()  # 空集合
        
        @property
        def key(self):
            """键属性访问器"""
            return self._key
            
        @key.setter
        def key(self, new_key):
            """键属性设置器"""
            if self._ion_reference:
                # 如果有ION引用，通过ION更新键
                self._ion_reference.update_node_key(self, new_key)
            else:
                self._key = new_key
                
        @property
        def val(self):
            """值属性访问器"""
            return self._val
            
        @val.setter
        def val(self, new_val):
            """值属性设置器"""
            if self._ion_reference:
                # 如果有ION引用，通过ION更新值
                self._ion_reference.update_node_value(self, new_val)
            else:
                self._val = new_val
                
        @property
        def metadata(self):
            """元数据属性访问器"""
            return self._metadata
            
        @metadata.setter
        def metadata(self, new_metadata):
            """元数据属性设置器"""
            if self._ion_reference:
                # 如果有ION引用，通过ION更新元数据
                self._ion_reference.update_node_metadata(self, new_metadata)
            else:
                self._metadata = new_metadata or {}
                
        @property
        def r(self):
            """关系列表属性访问器"""
            return self._r
            
        @r.setter
        def r(self, new_relations):
            """关系列表属性设置器"""
            if self._ion_reference:
                # 如果有ION引用，通过ION更新关系
                self._ion_reference.update_node_relations(self, new_relations)
            else:
                self._r = new_relations or []
                
        @property
        def weight(self):
            """权重属性访问器"""
            return self._weight
            
        @weight.setter
        def weight(self, new_weight):
            """权重属性设置器"""
            self._weight = float(new_weight)
            
        @property
        def visited(self):
            """访问状态属性访问器"""
            return self._visited
            
        @visited.setter
        def visited(self, value):
            """访问状态属性设置器"""
            self._visited = bool(value)
            
        @property
        def tags(self):
            """标签属性访问器"""
            return self._tags
            
        def add_tag(self, tag):
            """添加标签"""
            self._tags.add(tag)
            if self._ion_reference:
                self._ion_reference._index_tag(tag, self)
                
        def remove_tag(self, tag):
            """移除标签"""
            if tag in self._tags:
                self._tags.remove(tag)
                if self._ion_reference:
                    self._ion_reference._remove_tag_index(tag, self)
            
        def add_relation(self, node, rel_type=None, rel_weight=1.0, metadata=None):
            """添加带类型、权重和元数据的关系"""
            relation = {
                'node': node,
                'type': rel_type,
                'weight': rel_weight,
                'metadata': metadata
            }
            self._r.append(relation)
            
        def get_relations(self, rel_type=None):
            """获取特定类型的关系"""
            if rel_type is None:
                return [r['node'] for r in self._r]
            return [r['node'] for r in self._r if r['type'] == rel_type]
        
        def update_metadata(self, key, value):
            """更新单个元数据项"""
            if self._metadata is None:
                self._metadata = {}
                
            old_value = self._metadata.get(key)
            self._metadata[key] = value
            
            # 如果有ION引用，同步更新索引
            if self._ion_reference and old_value != value:
                if old_value is not None:
                    self._ion_reference._remove_metadata_index(key, old_value, self)
                self._ion_reference._index_metadata(key, value, self)
        
        @property
        def row(self):
            """行标识属性访问器"""
            return self._row
            
        @row.setter
        def row(self, new_row):
            """行标识属性设置器"""
            if self._ion_reference:
                # 如果有ION引用，通过ION更新行标识
                self._ion_reference.update_node_row(self, new_row)
            else:
                self._row = new_row
        
        def __str__(self):
            return f"IONNode(key={self._key}, val={self._val}, relations={len(self._r)})"

    def __init__(self, start=None, size=1024, max_workers=4, load_factor_threshold=0.75, 
                enable_memory_optimization=False, index_type="hash", cache_strategy="none", 
                memory_limit_mb=None, persistence_enabled=False, persistence_path=None,
                auto_optimize=False,
                # 新增优化选项
                partition_config=None,
                parallel_query=False,
                index_compression=False,
                large_dataset_mode=False,
                max_partition_size=10000,
                dynamic_partition=False,
                # 性能优化增强选项
                value_index_type="bidirectional",  # 值索引类型: "bidirectional", "hash", "btree"
                query_cache_enabled=True,          # 启用查询缓存
                query_cache_size=1000,             # 查询缓存大小
                fine_grained_locks=True,           # 细粒度锁
                lock_timeout_ms=1000,              # 锁超时时间(毫秒)
                batch_buffer_size=5000,            # 批处理缓冲区大小
                node_pool_enabled=True,            # 启用节点对象池
                node_reuse_threshold=0.5,          # 节点重用阈值
                value_compression=False,           # 值压缩
                async_indexing=False,              # 异步索引更新
                # 同步系统选项
                enable_sync=True,                  # 启用同步系统
                sync_async_mode=False,             # 同步系统异步模式
                sync_batch_size=100,               # 同步批处理大小
                sync_batch_timeout=1.0,
                #智能选项
                enable_learning=True
                ):           # 同步批处理超时时间
        """初始化ION实例，添加优化参数支持，特别针对大型数据集(10万+)
        
        参数:
            start: 起始节点
            size: 初始桶大小，处理大数据时推荐设置更大的值(如100000)
            max_workers: 并行处理的最大工作线程数
            load_factor_threshold: 触发扩容的负载因子阈值
            enable_memory_optimization: 是否启用内存优化
            index_type: 索引类型，可选值: "hash"(默认), "lsm", "b+tree"
            cache_strategy: 缓存策略，可选值: "none"(不缓存), "lru", "lfu"
            memory_limit_mb: 内存限制(MB)，超过此限制触发GC，None表示不限制
            persistence_enabled: 是否启用自动持久化
            persistence_path: 持久化路径，None表示使用临时目录
            auto_optimize: 是否启用自动优化
            
            # 大数据集优化选项
            partition_config: 数据分区配置，格式: {"strategy": "hash|range|field", "field": "metadata_field"}
            parallel_query: 是否启用并行查询（多线程同时查询不同分区）
            index_compression: 是否启用索引压缩（以空间换时间，但会降低修改性能）
            large_dataset_mode: 启用大数据集优化模式（自动调整多个参数）
            max_partition_size: 每个分区最大节点数，超过将触发分区分裂
            dynamic_partition: 是否启用动态分区（根据访问模式自动调整分区）
            
            # 性能优化增强选项
            value_index_type: 值索引类型，用于优化按值查询性能
            query_cache_enabled: 是否启用查询缓存
            query_cache_size: 查询缓存最大条目数
            fine_grained_locks: 是否使用细粒度锁替代全局锁
            lock_timeout_ms: 锁请求超时时间(毫秒)
            batch_buffer_size: 批处理操作的缓冲区大小
            node_pool_enabled: 是否启用节点对象池(减少内存分配)
            node_reuse_threshold: 节点对象重用阈值
            value_compression: 是否启用值压缩(对大字符串和字典)
            async_indexing: 是否启用异步索引更新
        """
        # 处理大数据集模式
        if large_dataset_mode:
            # 自动调整参数以适应大数据集
            size = max(size, 100000)  # 至少10万桶
            load_factor_threshold = min(load_factor_threshold, 0.6)  # 降低负载因子
            enable_memory_optimization = True
            parallel_query = True
            query_cache_enabled = True
            fine_grained_locks = True
            node_pool_enabled = True
            if not memory_limit_mb:
                memory_limit_mb = 2048  # 默认2GB内存限制
            max_workers = max(max_workers, 8)  # 增加工作线程
        
        # OND风格的哈希表存储
        self.buckets = [[] for _ in range(size)]
        self.size = size
        self.count = 0  # 实际节点数
        self.start = [start] if start else []
        
        # ICombinedDataStructure的功能组合
        self._combined_data = ICombinedDataStructure(initial_capacity=size, load_factor=load_factor_threshold)
        
        # 映射和索引
        self.bimap = NestedBidirectionalMap()  # 双向映射
        self.metadata_index = {}  # 元数据索引
        self.tag_index = {}  # 标签索引
        self.value_type_index = {}  # 值类型索引
        self.relation_type_index = {}  # 关系类型索引
        
        # 性能优化增强: 值索引改进
        self.value_index_type = value_index_type
        self.value_index = {}  # 优化的值索引
        
        # 并发和锁
        self.lock = threading.RLock()  # 全局锁
        
        # 性能优化增强: 细粒度锁
        self.fine_grained_locks = fine_grained_locks
        if fine_grained_locks:
            # 使用更多更小粒度的锁
            self.lock_stripes = 64  # 锁分段数
            self.stripe_locks = [threading.RLock() for _ in range(self.lock_stripes)]
            self.lock_timeout = lock_timeout_ms / 1000.0  # 转换为秒
        
        self.bucket_locks = [threading.RLock() for _ in range(size)]  # 桶锁
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # 扩容相关
        self.load_factor_threshold = load_factor_threshold  # 负载因子阈值
        self.resizing_lock = threading.RLock()  # 扩容专用锁
        self.is_resizing = False  # 是否正在扩容
        
        # 事件和回调
        self.event_callbacks = {
            'node_added': [],
            'node_updated': [],
            'node_removed': [],
            'relation_added': [],
            'relation_removed': [],
            'resize': [],
            'transaction_started': [],
            'transaction_committed': [],
            'transaction_aborted': []
        }
        if enable_learning:
            # 初始化机器学习引擎
            self.ml_engine = MLEngine(self)
        # 设置IONNode类作为节点类
        self.node_class = self.IONNode
        
        # 事务管理
        self.active_transactions = {}  # 事务ID -> 事务对象
        self.lock_manager = LockManager()  # 锁管理器
        self.transaction_counter = 0  # 事务计数器
        self.default_isolation_level = IsolationLevel.SERIALIZABLE  # 默认隔离级别
        self.transaction_timeout = 30  # 默认事务超时时间（秒）

        # 并发控制
        self.concurrency_mode = 'optimistic'  # 并发控制模式: 'optimistic' 或 'pessimistic'
        self.version_counter = 0  # 用于乐观并发控制的版本计数
        self.node_versions = {}  # 节点键 -> 最新版本号

        # 智能线程池管理
        self.adaptive_threading = True  # 是否启用自适应线程池
        self.min_workers = 2  # 最小工作线程数
        self.max_workers = max_workers  # 最大工作线程数
        self.worker_usage_stats = []  # 线程使用统计
        self.worker_adjustment_interval = 60  # 自动调整线程数的间隔（秒）
        self.last_worker_adjustment = time.time()  # 上次调整线程数的时间
        
        # 添加复合索引支持
        self.compound_indices = {}  # 复合索引: (index_type1, index_type2) -> {复合键: [节点列表]}
        self.compound_index_types = set()  # 已创建的复合索引类型
        
        # 添加索引锁来优化并发控制
        self.metadata_index_lock = threading.RLock()
        self.tag_index_lock = threading.RLock()
        self.value_type_index_lock = threading.RLock()
        self.relation_type_index_lock = threading.RLock()
        self.compound_index_lock = threading.RLock()
        self.value_index_lock = threading.RLock()  # 新增: 值索引锁
        
        # 添加批量索引更新队列
        self.index_update_queue = []
        self.index_update_lock = threading.RLock()
        self.index_update_threshold = 100  # 队列长度达到阈值时触发批量更新
        self.index_update_timer = None  # 定时任务
        self.index_update_interval = 5.0  # 定时更新间隔(秒)
        self.is_updating_indices = False  # 是否正在批量更新索引
        
        # 优化参数
        self.enable_memory_optimization = enable_memory_optimization
        self.index_type = index_type
        self.cache_strategy = cache_strategy
        self.memory_limit_mb = memory_limit_mb
        self.persistence_enabled = persistence_enabled
        self.persistence_path = persistence_path
        self.auto_optimize = auto_optimize
        
        # 针对大数据集的优化选项
        self.partition_config = partition_config
        self.parallel_query = parallel_query
        self.index_compression = index_compression
        self.large_dataset_mode = large_dataset_mode
        self.max_partition_size = max_partition_size
        self.dynamic_partition = dynamic_partition
        
        # 性能优化增强选项
        self.query_cache_enabled = query_cache_enabled
        self.query_cache_size = query_cache_size
        self.batch_buffer_size = batch_buffer_size
        self.node_pool_enabled = node_pool_enabled
        self.node_reuse_threshold = node_reuse_threshold
        self.value_compression = value_compression
        self.async_indexing = async_indexing
        
        # 同步系统初始化
        self.enable_sync = enable_sync
        if enable_sync:
            self.sync_manager = SyncManager()
            self.sync_manager.set_async_mode(sync_async_mode)
            self.sync_batch_size = sync_batch_size
            self.sync_batch_timeout = sync_batch_timeout
            
            # 注册内置的同步处理器示例（用户可以选择使用）
            self._register_builtin_sync_handlers()
        else:
            self.sync_manager = None
        self.partition_config = partition_config or {"strategy": "hash", "field": None}
        self.parallel_query = parallel_query
        self.index_compression = index_compression
        self.large_dataset_mode = large_dataset_mode
        self.max_partition_size = max_partition_size
        self.dynamic_partition = dynamic_partition
        
        # 性能优化增强: 查询缓存
        self.query_cache_enabled = query_cache_enabled
        if query_cache_enabled:
            self.query_cache = {}  # 查询结果缓存
            self.query_cache_max_size = query_cache_size
            self.query_cache_hits = 0
            self.query_cache_misses = 0
            self.query_cache_lock = threading.RLock()
        
        # 性能优化增强: 节点对象池
        self.node_pool_enabled = node_pool_enabled
        if node_pool_enabled:
            self.node_pool = {}  # 键类型 -> [可重用节点列表]
            self.node_pool_lock = threading.RLock()
            self.node_reuse_threshold = node_reuse_threshold
        
        # 性能优化增强: 值压缩
        self.value_compression = value_compression
        
        # 性能优化增强: 异步索引
        self.async_indexing = async_indexing
        if async_indexing:
            self.indexing_queue = deque(maxlen=10000)
            self.indexing_thread = threading.Thread(
                target=self._async_indexing_worker,
                daemon=True
            )
            self.indexing_thread.start()
        
        # 性能优化增强: 批处理缓冲
        self.batch_buffer_size = batch_buffer_size
        self.batch_buffer = []
        self.batch_buffer_lock = threading.RLock()
        
        # 如果启用大数据集模式，自动调整多个参数
        if large_dataset_mode:
            self.enable_memory_optimization = True
            self.parallel_query = True
            if self.memory_limit_mb is None:
                self.memory_limit_mb = 2048  # 默认2GB内存限制
            if self.cache_strategy == "none":
                self.cache_strategy = "lfu"  # 大数据集默认使用LFU缓存
            if size < 100000:
                # 大数据集模式下，默认使用更大的桶大小
                self._resize(100000)
        
        # 初始化优化相关组件
        self._init_optimizations()
        
        # 数据分区支持
        self.partitions = {"default": set()}  # 分区名 -> 节点键集合
        self.node_partition_mapping = {}  # 节点键 -> 分区名
        self.partition_locks = {"default": threading.RLock()}  # 分区锁
        self.partition_stats = {"default": {"access_count": 0, "hit_ratio": 0}}  # 分区统计
        
        # 分区查询优化
        if self.parallel_query:
            # 创建分区专用线程池
            self.partition_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=min(len(self.partitions), max_workers)
            )
            # 分区查询缓存
            self.partition_query_cache = {}
            self.partition_query_cache_lock = threading.RLock()
        
        # 内存管理
        self.memory_monitor_active = enable_memory_optimization  
        self.memory_monitor_thread = None
        self.memory_last_gc_time = time.time()
        self.memory_gc_interval = 300  # 5分钟检查一次内存状态
        self.memory_usage_stats = []  # 内存使用统计
        
        # 缓存系统
        self._init_cache_system()
        
        # 启动内存监控
        if self.memory_monitor_active:
            self._start_memory_monitor()
            
        # 创建初始分区结构
        if self.partition_config["strategy"] != "hash" and self.partition_config["field"] is not None:
            self._initialize_partitions()
        
        # 表查找功能初始化
        self.row_index = {}  # 行索引: row_id -> [节点列表]
        self.table_index = {}  # 表索引: table_name -> {row_id: 节点}
        self.row_index_lock = threading.RLock()  # 行索引锁
        self.table_index_lock = threading.RLock()  # 表索引锁
    
    def _register_builtin_sync_handlers(self):
        """注册内置的同步处理器示例"""
        if not self.sync_manager:
            return
            
        # 用户可以选择性地启用这些处理器
        # 这里只是注册，不自动启用
        pass  # 用户需要手动注册他们需要的处理器
    
    def register_sync_handler(self, name: str, handler: ISyncHandler):
        """注册自定义同步处理器
        
        Args:
            name: 处理器名称
            handler: 实现了ISyncHandler接口的处理器实例
        """
        if self.sync_manager:
            self.sync_manager.register_handler(name, handler)
        else:
            raise RuntimeError("同步系统未启用，请在初始化时设置 enable_sync=True")
    
    def unregister_sync_handler(self, name: str):
        """注销同步处理器
        
        Args:
            name: 处理器名称
        """
        if self.sync_manager:
            self.sync_manager.unregister_handler(name)
    
    def get_sync_handler(self, name: str) -> ISyncHandler:
        """获取指定的同步处理器
        
        Args:
            name: 处理器名称
            
        Returns:
            ISyncHandler: 处理器实例，如果不存在返回None
        """
        if self.sync_manager:
            return self.sync_manager.get_handler(name)
        return None
    
    def list_sync_handlers(self) -> List[str]:
        """列出所有注册的同步处理器名称
        
        Returns:
            List[str]: 处理器名称列表
        """
        if self.sync_manager:
            return self.sync_manager.list_handlers()
        return []
    
    def set_sync_async_mode(self, async_mode: bool):
        """设置同步系统的异步模式
        
        Args:
            async_mode: 是否启用异步模式
        """
        if self.sync_manager:
            self.sync_manager.set_async_mode(async_mode)
    
    def get_sync_queue_size(self) -> int:
        """获取同步事件队列大小
        
        Returns:
            int: 队列中待处理的事件数量
        """
        if self.sync_manager:
            return self.sync_manager.get_queue_size()
        return 0
    
    def clear_sync_queue(self):
        """清空同步事件队列"""
        if self.sync_manager:
            self.sync_manager.clear_queue()
    
    def _trigger_sync_event(self, operation: SyncOperation, node=None, old_value=None, 
                           new_value=None, metadata=None, relations=None, context=None):
        """触发同步事件
        
        Args:
            operation: 操作类型
            node: 相关节点
            old_value: 旧值
            new_value: 新值
            metadata: 元数据
            relations: 关系信息
            context: 上下文信息
        """
        if self.sync_manager:
            event = SyncEvent(
                operation=operation,
                node=node,
                old_value=old_value,
                new_value=new_value,
                metadata=metadata,
                relations=relations,
                context=context
            )
            self.sync_manager.trigger_event(event)
    
    def _trigger_sync_batch_events(self, events: List[SyncEvent]):
        """触发批量同步事件
        
        Args:
            events: 事件列表
        """
        if self.sync_manager and events:
            self.sync_manager.trigger_batch_events(events)
    
    def _initialize_partitions(self):
        """根据分区策略初始化分区结构"""
        strategy = self.partition_config["strategy"]
        field = self.partition_config["field"]
        
        if strategy == "range" and field:
            # 创建范围分区（例如按时间、ID范围分区）
            self.create_partition("range_low", f"低范围 {field} 分区")
            self.create_partition("range_medium", f"中范围 {field} 分区")
            self.create_partition("range_high", f"高范围 {field} 分区")
            
        elif strategy == "field" and field:
            # 创建字段值分区（例如按状态、类型分区）
            self.create_partition(f"{field}_none", f"{field} 为空分区")
            # 其它分区会在数据插入时动态创建
    
    def create_dynamic_partition(self, partition_key):
        """根据分区键动态创建分区
        
        参数:
            partition_key: 分区键
            
        返回:
            str: 分区名称
        """
        strategy = self.partition_config["strategy"]
        field = self.partition_config["field"]
        
        if not self.dynamic_partition:
            return "default"  # 如果不启用动态分区，使用默认分区
            
        if strategy == "field" and field:
            # 创建基于字段值的分区
            partition_name = f"{field}_{partition_key}"
            if partition_name not in self.partitions:
                self.create_partition(partition_name, f"{field}={partition_key} 分区")
            return partition_name
            
        return "default"  # 默认返回默认分区
    
    def split_partition(self, partition_name):
        """当分区大小超过限制时，将分区分裂成多个子分区
        
        参数:
            partition_name: 要分裂的分区名称
            
        返回:
            list: 新创建的分区名称列表
        """
        if partition_name not in self.partitions:
            return []
            
        with self.partition_locks[partition_name]:
            nodes = list(self.partitions[partition_name])
            if len(nodes) <= self.max_partition_size:
                return []  # 分区大小未超过限制，无需分裂
                
            # 创建子分区
            sub_partitions = []
            for i in range(2):  # 分裂为两个子分区
                sub_name = f"{partition_name}_{i}"
                if sub_name not in self.partitions:
                    self.create_partition(sub_name, f"{partition_name} 子分区 {i}")
                    sub_partitions.append(sub_name)
            
            # 重新分配节点
            mid = len(nodes) // 2
            for i, node_key in enumerate(nodes):
                sub_idx = 0 if i < mid else 1
                sub_name = f"{partition_name}_{sub_idx}"
                
                # 获取节点
                node = self.get_node_by_key(node_key)
                if node:
                    self.assign_node_to_partition(node, sub_name)
            
            return sub_partitions
    
    def merge_partitions(self, partitions_to_merge, new_partition_name=None):
        """合并多个分区为一个分区
        
        参数:
            partitions_to_merge: 要合并的分区名称列表
            new_partition_name: 新分区名称，如果为None则使用第一个分区的名称
            
        返回:
            str: 合并后的分区名称
        """
        if not partitions_to_merge or len(partitions_to_merge) < 2:
            return None
            
        # 验证所有分区都存在
        for p in partitions_to_merge:
            if p not in self.partitions:
                return None
                
        # 确定目标分区名称
        target_partition = new_partition_name or partitions_to_merge[0]
        if new_partition_name and new_partition_name not in self.partitions:
            self.create_partition(new_partition_name, f"合并分区 {new_partition_name}")
            
        # 将所有其他分区的节点移动到目标分区
        for p in partitions_to_merge:
            if p != target_partition:
                self.delete_partition(p, move_nodes_to=target_partition)
                
        return target_partition
    
    def get_partition_for_query(self, query_conditions):
        """根据查询条件确定需要查询的分区
        
        参数:
            query_conditions: 查询条件字典
            
        返回:
            list: 需要查询的分区名称列表
        """
        if not self.partition_config["field"]:
            return list(self.partitions.keys())  # 返回所有分区
            
        field = self.partition_config["field"]
        strategy = self.partition_config["strategy"]
        
        # 如果查询条件中包含分区字段
        if field in query_conditions:
            field_value = query_conditions[field]
            
            if strategy == "field":
                # 字段值分区
                partition_name = f"{field}_{field_value}"
                if partition_name in self.partitions:
                    return [partition_name]
                else:
                    return [f"{field}_none"]  # 返回空值分区
                    
            elif strategy == "range":
                # 范围分区
                try:
                    value = float(field_value)
                    if value < 100:
                        return ["range_low"]
                    elif value < 1000:
                        return ["range_medium"]
                    else:
                        return ["range_high"]
                except:
                    return ["default"]
        
        # 没有匹配的分区条件，返回所有分区
        return list(self.partitions.keys())
    
    def _init_optimizations(self):
        """初始化优化组件"""
        # 索引优化
        if self.index_type == "lsm":
            # 实现LSM树索引逻辑
            self.lsm_memtable = {}  # 内存表
            self.lsm_sstables = []  # 排序字符串表
            self.lsm_compaction_threshold = 4  # 合并阈值
            self.lsm_compaction_lock = threading.RLock()
            self.lsm_writes_since_compaction = 0
        elif self.index_type == "b+tree":
            # 实现B+树索引逻辑
            self.btree_order = 128  # B+树阶数
            self.btree_indices = {}  # 字段 -> B+树索引
        
        # 批处理优化
        self.batch_queue = []
        self.batch_queue_lock = threading.RLock()
        self.batch_size_threshold = 1000
        self.batch_flush_interval = 1.0  # 秒
        self.batch_flush_timer = None
        
        # 自动优化
        if self.auto_optimize:
            self.optimization_stats = {
                "query_patterns": Counter(),  # 查询模式统计
                "access_frequency": Counter(),  # 节点访问频率
                "operation_latency": [],  # 操作延迟统计
                "last_optimization": time.time()
            }
            # 计划自动优化任务
            self._schedule_auto_optimization()
    
    def _init_cache_system(self):
        """初始化缓存系统"""
        # 节点缓存
        self.node_cache = {}
        self.node_cache_max_size = 10000
        
        # 路径缓存
        self.path_cache = {}
        self.path_cache_max_size = 1000
        
        # 查询结果缓存
        self.query_cache = {}
        self.query_cache_max_size = 500
        
        # 缓存统计
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }
        
        # 根据缓存策略初始化辅助数据结构
        if self.cache_strategy == "lru":
            # LRU缓存辅助数据
            self.node_cache_access_time = {}  # 节点键 -> 最后访问时间
            self.path_cache_access_time = {}  # 缓存键 -> 最后访问时间
            self.query_cache_access_time = {}  # 查询缓存键 -> 最后访问时间
        elif self.cache_strategy == "lfu":
            # LFU缓存辅助数据
            self.node_cache_frequency = Counter()  # 节点键 -> 访问频率
            self.path_cache_frequency = Counter()  # 缓存键 -> 访问频率
            self.query_cache_frequency = Counter()  # 查询缓存键 -> 访问频率
    
    def _start_memory_monitor(self):
        """启动内存监控线程"""
        def memory_monitor_loop():
            while self.memory_monitor_active:
                try:
                    # 检查内存使用情况
                    if self.memory_limit_mb:
                        # 获取当前进程的内存使用
                        import psutil
                        process = psutil.Process(os.getpid())
                        mem_info = process.memory_info()
                        current_mb = mem_info.rss / (1024 * 1024)
                        
                        # 记录内存使用统计
                        self.memory_usage_stats.append((time.time(), current_mb))
                        if len(self.memory_usage_stats) > 100:
                            self.memory_usage_stats = self.memory_usage_stats[-100:]
                        
                        # 如果超过限制，触发垃圾回收
                        if current_mb > self.memory_limit_mb:
                            self._run_garbage_collection()
                    
                    # 定期垃圾回收
                    now = time.time()
                    if now - self.memory_last_gc_time > self.memory_gc_interval:
                        self._run_garbage_collection(force=True)
                        self.memory_last_gc_time = now
                        
                except Exception as e:
                    logging.error(f"内存监控错误: {e}")
                
                # 休眠
                time.sleep(30)  # 30秒检查一次
        
        # 创建并启动内存监控线程
        self.memory_monitor_thread = threading.Thread(
            target=memory_monitor_loop, 
            daemon=True,  # 设为守护线程，程序退出时自动结束
            name="ION-MemoryMonitor"
        )
        self.memory_monitor_thread.start()
    
    def _run_garbage_collection(self, force=False):
        """运行垃圾回收，清理未使用的资源"""
        if not self.enable_memory_optimization and not force:
            return
            
        # 记录开始时间
        start_time = time.time()
        logging.info("开始执行垃圾回收...")
        
        try:
            # 1. 清理节点对象池
            with self.node_pool_lock:
                self.node_pool.clear()
            
            # 2. 清理未使用的索引
            self._cleanup_unused_indices()
            
            # 3. 清理缓存
            self._cleanup_caches()
            
            # 4. 调用Python垃圾回收器
            import gc
            gc.collect()
            
            # 记录结束时间和释放的内存
            end_time = time.time()
            logging.info(f"垃圾回收完成，耗时: {end_time - start_time:.2f}秒")
            
            # 触发事件
            self._trigger_event('gc_completed', duration=end_time - start_time)
            
        except Exception as e:
            logging.error(f"垃圾回收失败: {e}")
    
    def _cleanup_unused_indices(self):
        """清理未使用的索引"""
        # 清理空的元数据索引
        with self.metadata_index_lock:
            empty_keys = [k for k, v in self.metadata_index.items() if not v]
            for k in empty_keys:
                del self.metadata_index[k]
        
        # 清理空的标签索引
        with self.tag_index_lock:
            empty_tags = [t for t, nodes in self.tag_index.items() if not nodes]
            for t in empty_tags:
                del self.tag_index[t]
        
        # 清理空的值类型索引
        with self.value_type_index_lock:
            empty_types = [t for t, nodes in self.value_type_index.items() if not nodes]
            for t in empty_types:
                del self.value_type_index[t]
        
        # 清理空的复合索引
        with self.compound_index_lock:
            for index_types in list(self.compound_indices.keys()):
                empty_keys = [k for k, nodes in self.compound_indices[index_types].items() if not nodes]
                for k in empty_keys:
                    del self.compound_indices[index_types][k]
    
    def _cleanup_caches(self):
        """清理缓存"""
        if self.cache_strategy == "lru":
            self._cleanup_lru_caches()
        elif self.cache_strategy == "lfu":
            self._cleanup_lfu_caches()
        else:
            # 默认清理策略 - 当缓存超出最大大小时清理
            self._cleanup_default_caches()
    
    def _cleanup_lru_caches(self):
        """基于LRU策略清理缓存"""
        # 清理节点缓存
        if len(self.node_cache) > self.node_cache_max_size:
            # 按访问时间排序
            sorted_items = sorted(self.node_cache_access_time.items(), key=lambda x: x[1])
            # 删除最旧的20%
            num_to_remove = max(1, int(self.node_cache_max_size * 0.2))
            for key, _ in sorted_items[:num_to_remove]:
                if key in self.node_cache:
                    del self.node_cache[key]
                    del self.node_cache_access_time[key]
                    self.cache_stats["evictions"] += 1
        
        # 清理路径缓存
        if len(self.path_cache) > self.path_cache_max_size:
            sorted_items = sorted(self.path_cache_access_time.items(), key=lambda x: x[1])
            num_to_remove = max(1, int(self.path_cache_max_size * 0.2))
            for key, _ in sorted_items[:num_to_remove]:
                if key in self.path_cache:
                    del self.path_cache[key]
                    del self.path_cache_access_time[key]
                    self.cache_stats["evictions"] += 1
        
        # 清理查询缓存
        if len(self.query_cache) > self.query_cache_max_size:
            sorted_items = sorted(self.query_cache_access_time.items(), key=lambda x: x[1])
            num_to_remove = max(1, int(self.query_cache_max_size * 0.2))
            for key, _ in sorted_items[:num_to_remove]:
                if key in self.query_cache:
                    del self.query_cache[key]
                    del self.query_cache_access_time[key]
                    self.cache_stats["evictions"] += 1
    
    def _cleanup_lfu_caches(self):
        """基于LFU策略清理缓存"""
        # 清理节点缓存
        if len(self.node_cache) > self.node_cache_max_size:
            # 按访问频率排序
            sorted_items = sorted(self.node_cache_frequency.items(), key=lambda x: x[1])
            # 删除使用最少的20%
            num_to_remove = max(1, int(self.node_cache_max_size * 0.2))
            for key, _ in sorted_items[:num_to_remove]:
                if key in self.node_cache:
                    del self.node_cache[key]
                    del self.node_cache_frequency[key]
                    self.cache_stats["evictions"] += 1
        
        # 清理路径缓存
        if len(self.path_cache) > self.path_cache_max_size:
            sorted_items = sorted(self.path_cache_frequency.items(), key=lambda x: x[1])
            num_to_remove = max(1, int(self.path_cache_max_size * 0.2))
            for key, _ in sorted_items[:num_to_remove]:
                if key in self.path_cache:
                    del self.path_cache[key]
                    del self.path_cache_frequency[key]
                    self.cache_stats["evictions"] += 1
        
        # 清理查询缓存
        if len(self.query_cache) > self.query_cache_max_size:
            sorted_items = sorted(self.query_cache_frequency.items(), key=lambda x: x[1])
            num_to_remove = max(1, int(self.query_cache_max_size * 0.2))
            for key, _ in sorted_items[:num_to_remove]:
                if key in self.query_cache:
                    del self.query_cache[key]
                    del self.query_cache_frequency[key]
                    self.cache_stats["evictions"] += 1
    
    def _cleanup_default_caches(self):
        """基本缓存清理"""
        # 清理节点缓存
        if len(self.node_cache) > self.node_cache_max_size:
            # 随机删除一些键
            num_to_remove = max(1, int(self.node_cache_max_size * 0.2))
            keys_to_remove = list(self.node_cache.keys())[:num_to_remove]
            for key in keys_to_remove:
                del self.node_cache[key]
                self.cache_stats["evictions"] += 1
        
        # 清理路径缓存
        if len(self.path_cache) > self.path_cache_max_size:
            num_to_remove = max(1, int(self.path_cache_max_size * 0.2))
            keys_to_remove = list(self.path_cache.keys())[:num_to_remove]
            for key in keys_to_remove:
                del self.path_cache[key]
                self.cache_stats["evictions"] += 1
        
        # 清理查询缓存
        if len(self.query_cache) > self.query_cache_max_size:
            num_to_remove = max(1, int(self.query_cache_max_size * 0.2))
            keys_to_remove = list(self.query_cache.keys())[:num_to_remove]
            for key in keys_to_remove:
                del self.query_cache[key]
                self.cache_stats["evictions"] += 1
    
    def _schedule_auto_optimization(self):
        """调度自动优化任务"""
        if not self.auto_optimize:
            return
            
        def auto_optimize_task():
            while self.auto_optimize:
                try:
                    # 每6小时运行一次自动优化
                    time.sleep(6 * 60 * 60)
                    if not self.auto_optimize:
                        break
                        
                    # 运行优化任务
                    self._run_auto_optimization()
                    
                except Exception as e:
                    logging.error(f"自动优化任务失败: {e}")
        
        # 创建并启动自动优化线程
        threading.Thread(
            target=auto_optimize_task, 
            daemon=True,
            name="ION-AutoOptimizer"
        ).start()
    
    def _run_auto_optimization(self):
        """运行自动优化操作"""
        # 记录开始时间
        start_time = time.time()
        logging.info("开始执行自动优化...")
        
        try:
            # 1. 分析查询模式
            self._analyze_query_patterns()
            
            # 2. 优化索引
            self._optimize_indices()
            
            # 3. 重新平衡数据分区
            self._rebalance_partitions()
            
            # 4. 调整缓存大小
            self._adjust_cache_sizes()
            
            # 记录最后优化时间
            self.optimization_stats["last_optimization"] = time.time()
            
            # 记录结束时间
            end_time = time.time()
            logging.info(f"自动优化完成，耗时: {end_time - start_time:.2f}秒")
            
            # 触发事件
            self._trigger_event('optimization_completed', duration=end_time - start_time)
            
        except Exception as e:
            logging.error(f"自动优化失败: {e}")
    
    def create_partition(self, partition_name, description=None):
        """创建新的数据分区
        
        参数:
            partition_name: 分区名称
            description: 分区描述
            
        返回:
            bool: 是否创建成功
        """
        if partition_name in self.partitions:
            return False  # 分区已存在
            
        with self.lock:
            self.partitions[partition_name] = set()
            self.partition_locks[partition_name] = threading.RLock()
            self.partition_stats[partition_name] = {
                "access_count": 0,
                "hit_ratio": 0,
                "created_time": time.time(),
                "description": description or f"分区 {partition_name}",
                "last_accessed": time.time(),
                "node_count": 0
            }
            
            # 如果启用并行查询，更新分区查询线程池
            if self.parallel_query and hasattr(self, 'partition_executor'):
                # 关闭当前线程池
                self.partition_executor.shutdown(wait=False)
                # 创建新的线程池，适应当前分区数量
                self.partition_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(len(self.partitions), self.max_workers)
                )
                
            return True
    
    def delete_partition(self, partition_name, move_nodes_to=None):
        """删除数据分区
        
        参数:
            partition_name: 分区名称
            move_nodes_to: 将节点移动到指定分区，如果为None则删除节点
            
        返回:
            bool: 是否删除成功
        """
        if partition_name not in self.partitions or partition_name == "default":
            return False  # 分区不存在或尝试删除默认分区
            
        with self.lock:
            # 获取分区中的所有节点
            nodes_to_move = list(self.partitions[partition_name])
            
            if move_nodes_to and move_nodes_to in self.partitions:
                # 将节点移动到目标分区
                for node_key in nodes_to_move:
                    node = self.get_node_by_key(node_key)
                    if node:
                        self.assign_node_to_partition(node, move_nodes_to)
            else:
                # 从分区映射中删除节点
                for node_key in nodes_to_move:
                    if node_key in self.node_partition_mapping:
                        del self.node_partition_mapping[node_key]
            
            # 删除分区相关数据
            del self.partitions[partition_name]
            del self.partition_locks[partition_name]
            del self.partition_stats[partition_name]
            
            return True
    
    def assign_node_to_partition(self, node, partition_name):
        """将节点分配到指定分区
        
        参数:
            node: 节点对象
            partition_name: 分区名称
            
        返回:
            bool: 是否分配成功
        """
        if partition_name not in self.partitions:
            # 如果分区不存在且启用动态分区，则创建新分区
            if self.dynamic_partition:
                self.create_partition(partition_name)
            else:
                return False
        
        node_key = node.key
        
        # 从当前分区移除节点
        current_partition = self.node_partition_mapping.get(node_key, "default")
        if current_partition in self.partitions:
            with self.partition_locks[current_partition]:
                if node_key in self.partitions[current_partition]:
                    self.partitions[current_partition].remove(node_key)
                    # 更新分区统计
                    self.partition_stats[current_partition]["node_count"] = \
                        len(self.partitions[current_partition])
        
        # 将节点添加到新分区
        with self.partition_locks[partition_name]:
            self.partitions[partition_name].add(node_key)
            self.node_partition_mapping[node_key] = partition_name
            
            # 更新分区统计
            self.partition_stats[partition_name]["node_count"] = \
                len(self.partitions[partition_name])
            self.partition_stats[partition_name]["last_accessed"] = time.time()
            
            # 检查分区大小，如果超过阈值则分裂
            if self.dynamic_partition and len(self.partitions[partition_name]) > self.max_partition_size:
                self.split_partition(partition_name)
        
        return True
    
    def get_partition_stats(self, partition_name=None):
        """获取分区统计信息
        
        参数:
            partition_name: 分区名称，如果为None则返回所有分区的统计
            
        返回:
            dict: 分区统计信息
        """
        if partition_name:
            if partition_name in self.partition_stats:
                stats = self.partition_stats[partition_name].copy()
                # 添加一些实时计算的统计
                if partition_name in self.partitions:
                    stats["current_size"] = len(self.partitions[partition_name])
                return stats
            return None
        
        # 返回所有分区的统计
        result = {}
        for name, stats in self.partition_stats.items():
            result[name] = stats.copy()
            if name in self.partitions:
                result[name]["current_size"] = len(self.partitions[name])
        
        # 添加总体统计
        result["_summary"] = {
            "total_partitions": len(self.partitions),
            "total_nodes": sum(len(p) for p in self.partitions.values()),
            "avg_partition_size": sum(len(p) for p in self.partitions.values()) / max(1, len(self.partitions))
        }
        
        return result
        
    def get_node_partition(self, node_key):
        """获取节点所在的分区
        
        参数:
            node_key: 节点键
            
        返回:
            str: 分区名称
        """
        return self.node_partition_mapping.get(node_key, "default")
    
    def get_partition_for_query(self, query_conditions):
        """根据查询条件确定需要查询的分区
        
        参数:
            query_conditions: 查询条件字典
            
        返回:
            list: 需要查询的分区名称列表
        """
        if not self.partition_config["field"]:
            return list(self.partitions.keys())  # 返回所有分区
            
        field = self.partition_config["field"]
        strategy = self.partition_config["strategy"]
        
        # 如果查询条件中包含分区字段
        if field in query_conditions:
            field_value = query_conditions[field]
            
            if strategy == "field":
                # 字段值分区
                partition_name = f"{field}_{field_value}"
                if partition_name in self.partitions:
                    return [partition_name]
                else:
                    return [f"{field}_none"]  # 返回空值分区
                    
            elif strategy == "range":
                # 范围分区
                try:
                    value = float(field_value)
                    if value < 100:
                        return ["range_low"]
                    elif value < 1000:
                        return ["range_medium"]
                    else:
                        return ["range_high"]
                except:
                    return ["default"]
        
        # 没有匹配的分区条件，返回所有分区
        return list(self.partitions.keys())
    
    def _rebalance_partitions(self):
        """重新平衡分区"""
        if len(self.partitions) <= 1:
            return  # 只有一个分区，无需平衡
            
        partition_sizes = {name: len(nodes) for name, nodes in self.partitions.items()}
        
        # 计算平均大小
        total_nodes = sum(partition_sizes.values())
        avg_size = total_nodes / len(self.partitions)
        
        # 找出过大和过小的分区
        oversized = []
        undersized = []
        
        for name, size in partition_sizes.items():
            if size > avg_size * 1.2:  # 超过平均大小20%
                oversized.append((name, size))
            elif size < avg_size * 0.8:  # 低于平均大小20%
                undersized.append((name, size))
                
        # 按大小倒序排序
        oversized.sort(key=lambda x: x[1], reverse=True)
        undersized.sort(key=lambda x: x[1])
        
        # 移动节点来平衡分区
        for over_name, over_size in oversized:
            # 当前分区是否仍然过大
            current_size = len(self.partitions[over_name])
            if current_size <= avg_size * 1.1:  # 允许10%的误差
                continue
                
            # 需要移动的节点数
            to_move = int(current_size - avg_size)
            
            for under_name, _ in undersized:
                # 目标分区是否仍然过小
                under_current_size = len(self.partitions[under_name])
                if under_current_size >= avg_size * 0.9:  # 允许10%的误差
                    continue
                    
                # 确定移动数量
                move_count = min(to_move, int(avg_size - under_current_size))
                if move_count <= 0:
                    continue
                    
                # 从过大分区中选择节点移动
                nodes_to_move = list(self.partitions[over_name])[:move_count]
                
                # 移动节点
                for node_key in nodes_to_move:
                    with self.partition_locks[over_name]:
                        self.partitions[over_name].remove(node_key)
                        
                    with self.partition_locks[under_name]:
                        self.partitions[under_name].add(node_key)
                        self.node_partition_mapping[node_key] = under_name
                        
                        # 更新节点分区信息
                        node_obj = self.get_node_by_key(node_key)
                        if node_obj:
                            node_obj._partition = under_name
                
                # 更新移动计数
                to_move -= move_count
                if to_move <= 0:
                    break
    
    def _analyze_query_patterns(self):
        """分析查询模式，更新优化统计信息"""
        # 分析查询频率最高的字段，考虑创建索引
        query_patterns = self.optimization_stats["query_patterns"]
        if query_patterns:
            top_fields = query_patterns.most_common(5)
            
            # 对高频查询字段创建索引
            for field, count in top_fields:
                if count > 20 and ":" in field:  # 最低阈值，格式如 "metadata:status"
                    parts = field.split(":", 1)
                    if len(parts) == 2:
                        index_type, field_name = parts
                        if index_type == "metadata" and field_name not in self.btree_indices:
                            # 为高频元数据字段创建B+树索引
                            if self.index_type == "b+tree":
                                self._create_btree_index_for_field(field_name)
                
        # 根据访问频率调整缓存分配
        access_freq = self.optimization_stats["access_frequency"]
        if access_freq:
            # 根据访问频率调整各缓存大小比例
            total_accesses = sum(access_freq.values())
            if total_accesses > 0:
                path_ratio = sum(v for k, v in access_freq.items() if k.startswith("path:")) / total_accesses
                query_ratio = sum(v for k, v in access_freq.items() if k.startswith("query:")) / total_accesses
                node_ratio = 1 - path_ratio - query_ratio
                
                # 根据比例调整最大缓存大小
                total_cache = self.node_cache_max_size + self.path_cache_max_size + self.query_cache_max_size
                
                # 计算新的缓存大小（最小保持原来的20%）
                new_node_cache_size = max(int(total_cache * node_ratio), int(self.node_cache_max_size * 0.2))
                new_path_cache_size = max(int(total_cache * path_ratio), int(self.path_cache_max_size * 0.2))
                new_query_cache_size = max(int(total_cache * query_ratio), int(self.query_cache_max_size * 0.2))
                
                # 确保总和不变
                total_new = new_node_cache_size + new_path_cache_size + new_query_cache_size
                if total_new != total_cache:
                    # 按比例调整
                    factor = total_cache / total_new
                    new_node_cache_size = int(new_node_cache_size * factor)
                    new_path_cache_size = int(new_path_cache_size * factor)
                    new_query_cache_size = total_cache - new_node_cache_size - new_path_cache_size
                
                # 更新缓存大小
                self.node_cache_max_size = new_node_cache_size
                self.path_cache_max_size = new_path_cache_size
                self.query_cache_max_size = new_query_cache_size
    
    def _create_btree_index_for_field(self, field_name):
        """为特定字段创建B+树索引"""
        if self.index_type != "b+tree" or field_name in self.btree_indices:
            return
            
        # 创建B+树索引
        try:
            # 初始化B+树索引
            self.btree_indices[field_name] = {}
            
            # 填充索引
            for bucket in self.buckets:
                for node in bucket:
                    if node.metadata and field_name in node.metadata:
                        value = node.metadata[field_name]
                        if value not in self.btree_indices[field_name]:
                            self.btree_indices[field_name][value] = []
                        self.btree_indices[field_name][value].append(node)
                        
            logging.info(f"为字段 '{field_name}' 创建了B+树索引")
        except Exception as e:
            logging.error(f"创建B+树索引失败: {e}")
            if field_name in self.btree_indices:
                del self.btree_indices[field_name]
    
    def _adjust_cache_sizes(self):
        """根据内存限制和使用情况调整缓存大小"""
        if not self.memory_limit_mb or not self.enable_memory_optimization:
            return
            
        try:
            # 获取当前内存使用情况
            import psutil
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            current_mb = mem_info.rss / (1024 * 1024)
            
            # 计算缓存使用的估计内存
            cache_size = len(self.node_cache) + len(self.path_cache) + len(self.query_cache)
            
            # 如果超过内存限制的70%，减少缓存大小
            if current_mb > self.memory_limit_mb * 0.7:
                reduction_factor = 0.8  # 减少20%
                self.node_cache_max_size = int(self.node_cache_max_size * reduction_factor)
                self.path_cache_max_size = int(self.path_cache_max_size * reduction_factor)
                self.query_cache_max_size = int(self.query_cache_max_size * reduction_factor)
                
                # 清理部分缓存
                self._cleanup_caches()
                
                logging.info(f"由于内存压力，缓存大小减少至: 节点({self.node_cache_max_size}), 路径({self.path_cache_max_size}), 查询({self.query_cache_max_size})")
            
            # 如果内存使用低于限制的40%，可以增加缓存大小
            elif current_mb < self.memory_limit_mb * 0.4:
                growth_factor = 1.2  # 增加20%
                self.node_cache_max_size = int(self.node_cache_max_size * growth_factor)
                self.path_cache_max_size = int(self.path_cache_max_size * growth_factor)
                self.query_cache_max_size = int(self.query_cache_max_size * growth_factor)
                
                logging.info(f"由于内存充足，缓存大小增加至: 节点({self.node_cache_max_size}), 路径({self.path_cache_max_size}), 查询({self.query_cache_max_size})")
                
        except ImportError:
            logging.warning("无法导入psutil模块，跳过缓存大小调整")
        except Exception as e:
            logging.error(f"调整缓存大小失败: {e}")
            
    def configure_cache(self, memory_limit_mb=None, cache_strategy=None, 
                       node_cache_size=None, path_cache_size=None, query_cache_size=None,
                       persistence_interval=None):
        """配置缓存系统
        
        参数:
            memory_limit_mb: 内存限制(MB)
            cache_strategy: 缓存策略 ('none', 'lru', 'lfu')
            node_cache_size: 节点缓存大小
            path_cache_size: 路径缓存大小
            query_cache_size: 查询缓存大小
            persistence_interval: 持久化间隔(秒)
            
        返回:
            self
        """
        changes_made = False
        
        # 更新内存限制
        if memory_limit_mb is not None:
            self.memory_limit_mb = memory_limit_mb
            changes_made = True
            
        # 更新缓存策略
        if cache_strategy in ('none', 'lru', 'lfu'):
            old_strategy = self.cache_strategy
            self.cache_strategy = cache_strategy
            
            # 如果策略改变，重新初始化缓存系统
            if old_strategy != cache_strategy:
                self._init_cache_system()
                changes_made = True
                
        # 更新缓存大小
        if node_cache_size is not None:
            self.node_cache_max_size = node_cache_size
            changes_made = True
            
        if path_cache_size is not None:
            self.path_cache_max_size = path_cache_size
            changes_made = True
            
        if query_cache_size is not None:
            self.query_cache_max_size = query_cache_size
            changes_made = True
            
        # 更新持久化间隔
        if persistence_interval is not None and persistence_interval > 0:
            self.persistence_enabled = True
            self.persistence_interval = persistence_interval
            changes_made = True
            
        # 如果发生变化且启用了内存优化，清理缓存
        if changes_made and self.enable_memory_optimization:
            self._cleanup_caches()
            
        return self
    
    def hf(self, obj):
        """哈希函数，通用对象到数字的映射"""
        return obj_to_number(obj) % self.size
    
    def _check_resize(self):
        """检查是否需要扩容"""
        # 如果正在扩容中，直接返回
        if self.is_resizing:
            return
            
        current_load_factor = self.get_current_load_factor()
        
        # 当负载因子超过阈值时，触发扩容
        if current_load_factor >= self.load_factor_threshold:
            with self.resizing_lock:
                # 双重检查，避免多线程重复扩容
                if not self.is_resizing and self.get_current_load_factor() >= self.load_factor_threshold:
                    self._resize(self.size * 2)
    
    def _resize(self, new_size):
        """扩容哈希表并重新哈希所有元素"""
        with self.resizing_lock:
            self.is_resizing = True
            
            try:
                print(f"正在扩容: {self.size} -> {new_size}")
                self._trigger_event('resize', old_size=self.size, new_size=new_size)
                
                # 保存旧桶
                old_buckets = self.buckets
                old_size = self.size
                
                # 创建新桶
                self.buckets = [[] for _ in range(new_size)]
                self.size = new_size
                
                # 更新锁数组大小
                old_locks = self.bucket_locks
                self.bucket_locks = [threading.RLock() for _ in range(new_size)]
                
                # 重新哈希所有节点
                node_count = 0
                for bucket in old_buckets:
                    for node in bucket:
                        # 计算新索引
                        new_index = self.hf(node.key)
                        
                        # 添加到新桶
                        with self.bucket_locks[new_index]:
                            self.buckets[new_index].append(node)
                            node_count += 1
                
                print(f"扩容完成: 重新哈希了 {node_count} 个节点")
            finally:
                self.is_resizing = False
    
    def get_current_load_factor(self):
        """获取当前负载因子"""
        return self.count / self.size if self.size > 0 else 1.0
    
    def create_node(self, key, val=None, metadata=None, tags=None, weight=1.0, row=None):
        """创建新节点或更新现有节点，支持自动分区
        
        Args:
            key: 节点键
            val: 节点值
            metadata: 节点元数据字典
            tags: 节点标签列表
            weight: 节点权重
            row: 表行标识，用于表查找功能
            
        Returns:
            Nodes: 创建或更新的节点
        """
        # 记录访问模式
        #if hasattr(self, 'ml_engine'):
        #    self.ml_engine.record_access(key, 'create')
        metadata = metadata or {}
        tags = tags or []
        
        # 1. 检查节点是否已存在
        existing_node = self.get_node_by_key(key)
        
        # 2. 如果节点已存在，更新它
        if existing_node:
            # 仅在需要时创建事务来包装更新
            with self.begin_transaction(isolation_level=IsolationLevel.READ_COMMITTED):
                # 更新节点值
                if val is not None and existing_node.val != val:
                    existing_node.val = val
                
                # 更新节点元数据
                if metadata:
                    self.update_node_metadata(existing_node, metadata)
                
                # 更新节点标签
                if tags:
                    existing_node_tags = existing_node.tags or []
                    for tag in tags:
                        if tag not in existing_node_tags:
                            self.update_node_tag(existing_node, operation='add', tags=tag)
                
                # 更新节点权重
                if existing_node.weight != weight:
                    self.update_node_weight(existing_node, weight)
                
            return existing_node
        
        # 3. 创建新节点
        with self.lock:
            # 创建新的IONNode实例
            new_node = self.node_class(key, val, metadata, tags, weight, row)
            
            # 计算键的哈希值并确定存储桶
            bucket_index = self.hf(key)  # 使用类的哈希函数计算索引
            
            # 获取桶锁并添加节点
            with self.bucket_locks[bucket_index]:
                self.buckets[bucket_index].append(new_node)
                
                # 更新计数
                self.count += 1
                
                # 选择合适的分区
                if self.partition_config["field"] is not None and self.partition_config["field"] in metadata:
                    # 基于分区字段值选择或创建分区
                    field_value = metadata[self.partition_config["field"]]
                    
                    if self.partition_config["strategy"] == "field":
                        # 字段值分区
                        partition_name = self.create_dynamic_partition(field_value)
                    elif self.partition_config["strategy"] == "range":
                        # 范围分区 
                        try:
                            value = float(field_value)
                            if value < 100:
                                partition_name = "range_low"
                            elif value < 1000:
                                partition_name = "range_medium"
                            else:
                                partition_name = "range_high"
                        except:
                            partition_name = "default"
                    else:
                        # 哈希分区
                        partition_name = "default"
                else:
                    # 默认分区
                    partition_name = "default"
                
                # 将节点添加到选定的分区
                self.assign_node_to_partition(new_node, partition_name)
                        
                # 添加到各种索引和双向映射
                # self._schedule_index_update('add', new_node)
                
                # 直接更新索引
                self.bimap.add(new_node.key, new_node.val)
                
                # 添加到组合数据结构
                self._combined_data.put(new_node.key, new_node.val, metadata=new_node.metadata)
                
                # 索引元数据
                if new_node.metadata:
                    for m_key, m_val in new_node.metadata.items():
                        self._index_metadata(m_key, m_val, new_node)
                
                # 索引标签
                if new_node.tags:
                    for tag in new_node.tags:
                        self._index_tag(tag, new_node)
                        
                # 索引值类型
                self._index_value_type(new_node.val, new_node)
                
                # 索引行标识
                if new_node.row is not None:
                    self._index_row(new_node.row, new_node)
                
                # 触发事件回调
                self._trigger_event('node_added', node=new_node)
                
                # 触发同步事件
                self._trigger_sync_event(SyncOperation.CREATE, node=new_node, new_value=val, 
                                        metadata=metadata, context={'key': key, 'partition': partition_name})
                
                # 检查是否需要扩容
                if self.count > self.size * self.load_factor_threshold:
                    # self._schedule_resize()
                    self._check_resize()
            
            return new_node
            
    def _get_nodes_from_partitions(self, partitions, filter_func=None):
        """从指定分区获取所有节点
        
        参数:
            partitions: 分区名称列表 
            filter_func: 可选的过滤函数
            
        返回:
            list: 节点列表
        """
        results = []
        
        # 如果启用并行查询且有多个分区，使用并行执行
        if self.parallel_query and len(partitions) > 1 and hasattr(self, 'partition_executor'):
            futures = []
            for partition_name in partitions:
                if partition_name in self.partitions:
                    futures.append(self.partition_executor.submit(
                        self._get_nodes_from_single_partition, 
                        partition_name, 
                        filter_func
                    ))
            
            # 收集结果
            for future in concurrent.futures.as_completed(futures):
                try:
                    nodes = future.result()
                    results.extend(nodes)
                except Exception as e:
                    print(f"Error getting nodes from partition: {e}")
        else:
            # 串行处理
            for partition_name in partitions:
                if partition_name in self.partitions:
                    nodes = self._get_nodes_from_single_partition(partition_name, filter_func)
                    results.extend(nodes)
        
        return results
    
    def _get_nodes_from_single_partition(self, partition_name, filter_func=None):
        """从单个分区获取节点
        
        参数:
            partition_name: 分区名称
            filter_func: 可选的过滤函数
            
        返回:
            list: 节点列表
        """
        results = []
        
        # 更新分区访问统计
        if partition_name in self.partition_stats:
            self.partition_stats[partition_name]["access_count"] += 1
            self.partition_stats[partition_name]["last_accessed"] = time.time()
        
        # 获取分区中的所有节点
        with self.partition_locks[partition_name]:
            node_keys = list(self.partitions[partition_name])
        
        # 获取节点对象并应用过滤器
        for key in node_keys:
            node = self.get_node_by_key(key)
            if node:
                if filter_func is None or filter_func(node):
                    results.append(node)
        
        # 更新命中率统计
        if partition_name in self.partition_stats and len(node_keys) > 0:
            hit_ratio = len(results) / len(node_keys)
            # 使用移动平均更新命中率
            current_ratio = self.partition_stats[partition_name]["hit_ratio"]
            self.partition_stats[partition_name]["hit_ratio"] = 0.9 * current_ratio + 0.1 * hit_ratio
        
        return results
    
    def get_node_by_key(self, key):
        """通过键获取节点"""
        # 记录访问模式
        #if hasattr(self, 'ml_engine'):
        #    self.ml_engine.record_access(key, 'read')
        index = self.hf(key)
        with self.bucket_locks[index]:
            for node in self.buckets[index]:
                if node.key == key:
                    return node
        return None
    
    def get_node_by_value(self, val):
        """通过值获取节点"""
        keys = self.bimap.get_by_value(val)
        if keys:
            # 如果返回的是一个集合，取第一个键
            if isinstance(keys, set) and keys:
                key = next(iter(keys))
                return self.get_node_by_key(key)
            # 如果返回的是单个键
            elif not isinstance(keys, set):
                return self.get_node_by_key(keys)
    
        # 如果双向映射未找到，则遍历搜索
        for bucket in self.buckets:
            for node in bucket:
                if node.val == val:
                    return node
        return None
    
    def remove_node_by_key(self, key):
        """通过键删除节点"""
        # 记录访问模式
        #if hasattr(self, 'ml_engine'):
        #    self.ml_engine.record_access(key, 'delete')
        index = self.hf(key)
        with self.bucket_locks[index]:
            bucket = self.buckets[index]
            for i, node in enumerate(bucket):
                if node.key == key:
                    removed = bucket.pop(i)
                    self.count -= 1
                    
                    # 清理索引
                    self._cleanup_node_indices(removed)
                    
                    # 从组合数据结构中删除
                    self._combined_data.delete(key, keep_relationships=False)
                    
                    self._trigger_event('node_removed', node=removed)
                    
                    # 触发同步事件
                    self._trigger_sync_event(SyncOperation.DELETE, node=removed, old_value=removed.val, 
                                            context={'key': key})
                    
                    return removed
        return None
    
    def _cleanup_node_indices(self, node):
        """清理被删除节点的所有索引"""
        # 清理双向映射
        self.bimap.remove(node.key)
        
        # 清理元数据索引
        if node.metadata:
            for m_key, m_val in node.metadata.items():
                self._remove_metadata_index(m_key, m_val, node)
        
        # 清理标签索引
        for tag in node.tags:
            self._remove_tag_index(tag, node)
        
        # 清理值类型索引
        value_type = type(node.val).__name__
        if value_type in self.value_type_index and node in self.value_type_index[value_type]:
            self.value_type_index[value_type].remove(node)
            if not self.value_type_index[value_type]:
                del self.value_type_index[value_type]
        
        # 清理行索引
        if node.row is not None:
            self._remove_row_index(node.row, node)
    
    def _index_metadata(self, meta_key, meta_val, node):
        """为元数据创建索引"""
        key = f"{meta_key}:{meta_val}"
        if key not in self.metadata_index:
            self.metadata_index[key] = []
        if node not in self.metadata_index[key]:
            self.metadata_index[key].append(node)
    
    def _remove_metadata_index(self, meta_key, meta_val, node):
        """移除元数据索引"""
        key = f"{meta_key}:{meta_val}"
        if key in self.metadata_index and node in self.metadata_index[key]:
            self.metadata_index[key].remove(node)
            if not self.metadata_index[key]:
                del self.metadata_index[key]
    
    def _index_tag(self, tag, node):
        """为标签创建索引"""
        if tag not in self.tag_index:
            self.tag_index[tag] = []
        if node not in self.tag_index[tag]:
            self.tag_index[tag].append(node)
    
    def _remove_tag_index(self, tag, node):
        """移除标签索引"""
        if tag in self.tag_index and node in self.tag_index[tag]:
            self.tag_index[tag].remove(node)
            if not self.tag_index[tag]:
                del self.tag_index[tag]
    
    def _index_value_type(self, value, node):
        """为值类型创建索引"""
        value_type = type(value).__name__
        if value_type not in self.value_type_index:
            self.value_type_index[value_type] = []
        if node not in self.value_type_index[value_type]:
            self.value_type_index[value_type].append(node)
    
    def find_by_metadata(self, meta_key, meta_val):
        """通过元数据查找节点"""
        # 记录访问模式
        key = f"{meta_key}:{meta_val}"
        #if hasattr(self, 'ml_engine'):
        #    self.ml_engine.record_access(key, 'metadata')
        return self.metadata_index.get(key, [])
    
    def find_by_tag(self, tag):
        """通过标签查找节点"""
        #if hasattr(self, 'ml_engine'):
        #    self.ml_engine.record_access(tag, 'tag')
        return self.tag_index.get(tag, [])
    
    def find_by_value_type(self, type_name):
        """通过值的类型查找节点"""
        return self.value_type_index.get(type_name, [])
    
    def update_node_key(self, node, new_key):
        """更新节点的键，并同步相关索引"""
        old_key = node.key
        
        # 如果新键已存在，抛出异常
        if self.get_node_by_key(new_key) and old_key != new_key:
            raise ValueError(f"键 {new_key} 已存在")
            
        # 从旧位置移除节点
        old_index = self.hf(old_key)
        with self.bucket_locks[old_index]:
            bucket = self.buckets[old_index]
            for i, n in enumerate(bucket):
                if n.key == old_key:
                    bucket.pop(i)
                    break
        
        # 更新节点键
        node._key = new_key
        
        # 更新双向映射
        val = node.val
        self.bimap.remove(old_key)
        self.bimap.add(new_key, val)
        
        # 添加到新位置
        new_index = self.hf(new_key)
        with self.bucket_locks[new_index]:
            self.buckets[new_index].append(node)
        
        # 更新组合数据结构
        self._combined_data.delete(old_key)
        self._combined_data.put(new_key, val, metadata=node.metadata)
        
        self._trigger_event('node_updated', node=node, old_key=old_key)
        return node
    
    def update_node_value(self, node, new_val):
        """更新节点的值，并同步相关索引"""
        # 记录访问模式
        if hasattr(self, 'ml_engine'):
            self.ml_engine.record_access(node.key, 'update')
        old_val = node.val
        
        # 更新双向映射
        key = node.key
        self.bimap.remove(key)
        self.bimap.add(key, new_val)
        
        # 更新值类型索引
        old_type = type(old_val).__name__
        new_type = type(new_val).__name__
        
        if old_type != new_type:
            # 移除旧类型索引
            if old_type in self.value_type_index and node in self.value_type_index[old_type]:
                self.value_type_index[old_type].remove(node)
                if not self.value_type_index[old_type]:
                    del self.value_type_index[old_type]
            
            # 添加新类型索引
            self._index_value_type(new_val, node)
        
        # 更新节点值
        node._val = new_val
        
        # 更新组合数据结构
        self._combined_data.put(key, new_val, metadata=node.metadata)
        
        self._trigger_event('node_updated', node=node, old_val=old_val)
        
        # 触发同步事件
        self._trigger_sync_event(SyncOperation.UPDATE, node=node, old_value=old_val, 
                               new_value=new_val, context={'key': key, 'update_type': 'value'})
        
        return node
    
    def update_node_metadata(self, node, new_metadata):
        """更新节点的元数据并同步索引"""
        # 记录访问模式
        #if hasattr(self, 'ml_engine'):
        #    self.ml_engine.record_access(node.key, 'update')
        if not isinstance(new_metadata, dict):
            raise TypeError("元数据必须是字典类型")
            
        old_metadata = node.metadata.copy() if node.metadata else {}
        
        # 如果节点没有metadata属性，初始化它
        if node.metadata is None:
            node._metadata = {}
            
        # 获取旧元数据的键值对
        old_meta_pairs = set()
        if old_metadata:
            for key, value in old_metadata.items():
                old_meta_pairs.add((key, value))
        
        # 更新节点的元数据
        node._metadata.update(new_metadata)
        
        # 获取新元数据的键值对
        new_meta_pairs = set()
        for key, value in node.metadata.items():
            new_meta_pairs.add((key, value))
            
        # 需要移除的元数据索引
        to_remove = old_meta_pairs - new_meta_pairs
        
        # 需要添加的元数据索引
        to_add = new_meta_pairs - old_meta_pairs
        
        # 移除旧的元数据索引
        for meta_key, meta_val in to_remove:
            self._remove_metadata_index(meta_key, meta_val, node)
        
        # 添加新的元数据索引
        for meta_key, meta_val in to_add:
            self._index_metadata(meta_key, meta_val, node)
        
        # 更新组合数据结构
        self._combined_data.set_node_metadata(node.key, node.metadata)
        
        self._trigger_event('node_updated', node=node, metadata_changed=True)
        
        # 从复合索引中删除节点
        self._remove_node_from_compound_indices(node)
        
        # 重新添加到复合索引
        for index_types in self.compound_index_types:
            self._index_node_to_compound(node, index_types)
        
        return node
    
    def update_node_relations(self, node, new_relations):
        """更新节点的关系列表"""
        old_relations = node.r.copy()
        
        # 移除旧的关系类型索引
        for relation in old_relations:
            rel_type = relation.get('type')
            if rel_type and rel_type in self.relation_type_index:
                if node in self.relation_type_index[rel_type]:
                    self.relation_type_index[rel_type].remove(node)
                    if not self.relation_type_index[rel_type]:
                        del self.relation_type_index[rel_type]
        
        # 更新关系列表
        node._r = new_relations or []
        
        # 添加新的关系类型索引
        for relation in node.r:
            rel_type = relation.get('type')
            if rel_type:
                if rel_type not in self.relation_type_index:
                    self.relation_type_index[rel_type] = []
                if node not in self.relation_type_index[rel_type]:
                    self.relation_type_index[rel_type].append(node)
                    
                # 添加到组合数据结构
                target_node = relation['node']
                if hasattr(target_node, 'key'):
                    target_key = target_node.key
                else:
                    target_key = target_node
                
                self._combined_data.add_relationship(node.key, target_key)
                
                if relation.get('metadata'):
                    self._combined_data.add_relationship_with_metadata(
                        node.key, target_key, relation['metadata'])
        
        self._trigger_event('node_updated', node=node, relations_changed=True)
        return node
    
    def add_relationship(self, source, target, rel_type=None, rel_weight=1.0, metadata=None):
        """添加关系 (支持多种输入类型)"""
        # 记录访问模式
        if hasattr(self, 'ml_engine'):
            source_node = self._get_node_from_input(source)
            target_node = self._get_node_from_input(target)
            if source_node:
                self.ml_engine.record_access(source_node.key, 'relation_add')
            if target_node:
                self.ml_engine.record_access(target_node.key, 'relation_add')
        source_node = self._get_node_from_input(source)
        target_node = self._get_node_from_input(target)
        
        if source_node and target_node:
            source_node.add_relation(target_node, rel_type, rel_weight, metadata)
            
            # 更新关系类型索引
            if rel_type:
                if rel_type not in self.relation_type_index:
                    self.relation_type_index[rel_type] = []
                if source_node not in self.relation_type_index[rel_type]:
                    self.relation_type_index[rel_type].append(source_node)
            
            # 更新组合数据结构
            self._combined_data.add_relationship(source_node.key, target_node.key)
            if metadata:
                self._combined_data.add_relationship_with_metadata(
                    source_node.key, target_node.key, metadata)
            
            self._trigger_event('relation_added', 
                               source=source_node, 
                               target=target_node, 
                               rel_type=rel_type)
            
            # 触发同步事件
            self._trigger_sync_event(SyncOperation.ADD_RELATION, 
                                   node=source_node,
                                   relations={'target': target_node, 'type': rel_type, 'weight': rel_weight},
                                   metadata=metadata,
                                   context={'source_key': source_node.key, 'target_key': target_node.key})
            
            return True
        return False
    
    def _get_node_from_input(self, input_data):
        """从各种输入类型获取节点"""
        if isinstance(input_data, self.IONNode):
            return input_data
        elif hasattr(input_data, 'key'):
            return self.get_node_by_key(input_data.key)
        else:
            # 尝试作为键查找
            node = self.get_node_by_key(input_data)
            if node:
                return node
            # 尝试作为值查找
            return self.get_node_by_value(input_data)
    
    def remove_relationship(self, source, target, rel_type=None):
        """移除两个节点间的关系"""
        # 记录访问模式
        if hasattr(self, 'ml_engine'):
            source_node = self._get_node_from_input(source)
            target_node = self._get_node_from_input(target)
            if source_node:
                self.ml_engine.record_access(source_node.key, 'relation_remove')
            if target_node:
                self.ml_engine.record_access(target_node.key, 'relation_remove')
        source_node = self._get_node_from_input(source)
        target_node = self._get_node_from_input(target)
        
        if source_node and target_node:
            # 找到并移除关系
            for i, rel in enumerate(source_node.r):
                if rel['node'] == target_node and (rel_type is None or rel['type'] == rel_type):
                    removed_rel = source_node.r.pop(i)
                    
                    # 更新关系类型索引
                    removed_type = removed_rel.get('type')
                    if removed_type and removed_type in self.relation_type_index:
                        # 检查节点是否还有其他相同类型的关系
                        has_same_type = False
                        for other_rel in source_node.r:
                            if other_rel.get('type') == removed_type:
                                has_same_type = True
                                break
                        
                        if not has_same_type:
                            self.relation_type_index[removed_type].remove(source_node)
                            if not self.relation_type_index[removed_type]:
                                del self.relation_type_index[removed_type]
                    
                    # 从组合数据结构中移除关系
                    self._combined_data.remove_relationship(source_node.key, target_node.key)
                    
                    self._trigger_event('relation_removed', 
                                      source=source_node, 
                                      target=target_node, 
                                      rel_type=removed_type)
                    
                    # 触发同步事件
                    self._trigger_sync_event(SyncOperation.REMOVE_RELATION,
                                           node=source_node,
                                           relations={'target': target_node, 'type': removed_type},
                                           context={'source_key': source_node.key, 'target_key': target_node.key})
                    
                    return True
        return False
    
    def _reset_visited(self):
        """重置所有节点的访问状态"""
        for bucket in self.buckets:
            for node in bucket:
                node.visited = False
    
    def search_path(self, start, end, max_depth=10, rel_type=None):
        """
        查找从start到end的路径，使用双向BFS优化查找效率
        
        双向BFS将复杂度从O(V^d)降低到O(V^(d/2))，其中V是分支因子，d是路径长度
        """
        start_time = time.time()
        start_node = self._get_node_from_input(start)
        end_node = self._get_node_from_input(end)
        
        if not start_node or not end_node:
            return None
            
        # 特殊情况：如果起点和终点是同一个节点
        if start_node == end_node:
            return [start_node]
            
        # 重置所有节点的访问状态
        self._reset_visited()
        
        # 前向搜索队列（从起点出发）
        forward_queue = deque([(start_node, 0)])  # (节点, 深度)
        forward_visited = {start_node: None}  # 节点 -> 前驱节点
        
        # 后向搜索队列（从终点出发）
        backward_queue = deque([(end_node, 0)])  # (节点, 深度)
        backward_visited = {end_node: None}  # 节点 -> 前驱节点
        
        # 相遇节点
        intersection_node = None
        
        # 交替执行前向和后向搜索
        while forward_queue and backward_queue:
            # 检查是否达到最大深度
            if forward_queue[0][1] + backward_queue[0][1] > max_depth:
                break
                
            # 前向搜索一步
            current, depth = forward_queue.popleft()
            
            # 获取所有关系
            relations = current.get_relations(rel_type)
            for next_node in relations:
                if next_node not in forward_visited:
                    forward_visited[next_node] = current
                    forward_queue.append((next_node, depth + 1))
                    
                    # 检查是否与后向搜索相遇
                    if next_node in backward_visited:
                        intersection_node = next_node
                        break
            
            # 如果找到相遇点，跳出循环
            if intersection_node:
                break
            
            # 后向搜索一步
            current, depth = backward_queue.popleft()
            
            # 获取所有关系（需要找到指向当前节点的所有节点）
            # 注意：这里需要遍历所有节点来查找指向当前节点的关系
            for bucket in self.buckets:
                for node in bucket:
                    if not node.visited:  # 避免重复检查
                        for rel in node.r:
                            if rel['node'] == current:
                                if node not in backward_visited:
                                    backward_visited[node] = current
                                    backward_queue.append((node, depth + 1))
                                    
                                    # 检查是否与前向搜索相遇
                                    if node in forward_visited:
                                        intersection_node = node
                                        break
                        # 标记为已访问，避免下次重复检查        
                        node.visited = True
            
            # 如果找到相遇点，跳出循环
            if intersection_node:
                break
        
        # 如果找到相遇点，构建路径
        if intersection_node:
            # 构建前向路径（从起点到相遇点）
            forward_path = []
            node = intersection_node
            while node:
                forward_path.append(node)
                node = forward_visited[node]
            forward_path.reverse()  # 反转，使其从起点开始
            
            # 构建后向路径（从相遇点到终点，不包括相遇点）
            backward_path = []
            node = backward_visited[intersection_node]
            while node:
                backward_path.append(node)
                node = backward_visited[node]
            path=forward_path + backward_path
            # 记录查询性能
            if hasattr(self, 'ml_engine'):
                execution_time = time.time() - start_time
                result_count = len(path) if path else 0
                self.ml_engine.record_query_performance('search_path', execution_time, result_count)
            # 合并路径（不重复添加相遇点）
            return forward_path + backward_path
            
        # 未找到路径
        return None
        
    def search_path_optimized(self, start, end, max_depth=10, rel_type=None):
        """
        优化版本的路径搜索，支持自动选择最佳算法
        
        根据图的特性和搜索条件自动选择单向或双向BFS，并构建反向索引加速搜索
        """
        start_time = time.time()
        start_node = self._get_node_from_input(start)
        end_node = self._get_node_from_input(end)
        
        if not start_node or not end_node:
            return None
            
        # 特殊情况：如果起点和终点是同一个节点
        if start_node == end_node:
            return [start_node]
            
        # 重置所有节点的访问状态
        self._reset_visited()
        
        # 构建反向索引（目标节点 -> 源节点列表）
        reverse_index = {}
        
        # 预处理：为终点构建反向引用以加速后向搜索
        # 如果路径长度较短，预处理所有节点的反向引用
        # 如果最大深度 <= 4，预处理可能更高效
        if max_depth <= 4:
            for bucket in self.buckets:
                for node in bucket:
                    for rel in node.r:
                        target = rel['node']
                        if rel_type is None or rel.get('type') == rel_type:
                            if target not in reverse_index:
                                reverse_index[target] = []
                            reverse_index[target].append(node)
        
        # 前向搜索队列（从起点出发）
        forward_queue = deque([(start_node, 0)])  # (节点, 深度)
        forward_visited = {start_node: None}  # 节点 -> 前驱节点
        
        # 后向搜索队列（从终点出发）
        backward_queue = deque([(end_node, 0)])  # (节点, 深度)
        backward_visited = {end_node: None}  # 节点 -> 前驱节点
        
        # 相遇节点
        intersection_node = None
        
        # 交替执行前向和后向搜索
        while forward_queue and backward_queue:
            # 检查是否达到最大深度
            current_forward_depth = forward_queue[0][1]
            current_backward_depth = backward_queue[0][1]
            
            if current_forward_depth + current_backward_depth > max_depth:
                break
                
            # 选择较小的队列扩展（优化平衡性）
            if len(forward_queue) <= len(backward_queue):
                # 前向搜索一步
                current, depth = forward_queue.popleft()
                
                # 获取所有关系
                relations = current.get_relations(rel_type)
                for next_node in relations:
                    if next_node not in forward_visited:
                        forward_visited[next_node] = current
                        forward_queue.append((next_node, depth + 1))
                        
                        # 检查是否与后向搜索相遇
                        if next_node in backward_visited:
                            intersection_node = next_node
                            break
            else:
                # 后向搜索一步
                current, depth = backward_queue.popleft()
                
                # 使用预构建的反向索引或动态查找入边
                if reverse_index:
                    # 使用预构建的反向索引
                    prev_nodes = reverse_index.get(current, [])
                    for prev_node in prev_nodes:
                        if prev_node not in backward_visited:
                            backward_visited[prev_node] = current
                            backward_queue.append((prev_node, depth + 1))
                            
                            # 检查是否与前向搜索相遇
                            if prev_node in forward_visited:
                                intersection_node = prev_node
                                break
                else:
                    # 动态查找入边
                    for bucket in self.buckets:
                        for node in bucket:
                            if node not in backward_visited:  # 避免重复检查
                                for rel in node.r:
                                    if rel['node'] == current and (rel_type is None or rel.get('type') == rel_type):
                                        backward_visited[node] = current
                                        backward_queue.append((node, depth + 1))
                                        
                                        # 检查是否与前向搜索相遇
                                        if node in forward_visited:
                                            intersection_node = node
                                            break
                                
                                # 如果找到相遇点，提前退出
                                if intersection_node:
                                    break
                        if intersection_node:
                            break
            
            # 如果找到相遇点，跳出循环
            if intersection_node:
                break
        
        # 如果找到相遇点，构建路径
        if intersection_node:
            # 构建前向路径（从起点到相遇点）
            forward_path = []
            node = intersection_node
            while node:
                forward_path.append(node)
                node = forward_visited[node]
            forward_path.reverse()  # 反转，使其从起点开始
            
            # 构建后向路径（从相遇点到终点，不包括相遇点）
            backward_path = []
            node = backward_visited[intersection_node]
            while node:
                backward_path.append(node)
                node = backward_visited[node]
            # 记录查询性能
            path=forward_path + backward_path
            if hasattr(self, 'ml_engine'):
                execution_time = time.time() - start_time
                result_count = len(path) if path else 0
                self.ml_engine.record_query_performance('search_path_optimized', execution_time, result_count)
            # 合并路径（不重复添加相遇点）
            return forward_path + backward_path
            
        # 未找到路径
        return None
    
    def find_related(self, node_input, rel_type=None, max_depth=2):
        """查找与节点相关的所有节点"""
        start_node = self._get_node_from_input(node_input)
        start_time = time.time()
        if not start_node:
            return []
            
        self._reset_visited()
        result = set()
        
        def dfs(node, depth):
            if depth > max_depth:
                return
                
            relations = node.get_relations(rel_type)
            for related_node in relations:
                if not related_node.visited:
                    related_node.visited = True
                    result.add(related_node)
                    dfs(related_node, depth + 1)
                    
        start_node.visited = True
        dfs(start_node, 1)
        if hasattr(self, 'ml_engine'):
            execution_time = time.time() - start_time
            result_count = len(result) if result else 0
            self.ml_engine.record_query_performance('find_related', execution_time, result_count)
        return list(result)
    
    def find_common_related(self, node1, node2, rel_type=None, max_depth=2):
        """查找两个节点共同关联的节点"""
        related1 = set(self.find_related(node1, rel_type, max_depth))
        related2 = set(self.find_related(node2, rel_type, max_depth))
        return list(related1.intersection(related2))
    
    def calculate_centrality(self):
        """计算每个节点的中心度(基于关系数量)"""
        centrality = {}
        for bucket in self.buckets:
            for node in bucket:
                centrality[node] = len(node.r)
        return centrality
    
    def find_nodes_by_value_pattern(self, pattern_func):
        """根据值的模式查找节点"""
        results = []
        for bucket in self.buckets:
            for node in bucket:
                if pattern_func(node.val):
                    results.append(node)
        return results
    
    def find_most_connected(self, top_n=10):
        """查找关系最多的节点"""
        all_nodes = []
        for bucket in self.buckets:
            all_nodes.extend(bucket)
            
        return sorted(all_nodes, key=lambda x: len(x.r), reverse=True)[:top_n]
    
    def get_community(self, start_node=None):
        """用连通分量获取社区结构"""
        if not start_node and not self.start:
            return []
            
        start = start_node or self.start[0]
        start_node = self._get_node_from_input(start)
        
        if not start_node:
            return []
            
        self._reset_visited()
        community = []
        
        def dfs(node):
            node.visited = True
            community.append(node)
            
            for relation in node.r:
                next_node = relation['node']
                if not next_node.visited:
                    dfs(next_node)
                    
        dfs(start_node)
        return community
    
    def advanced_search(self, criteria=None, metadata_filters=None, tag_filters=None, 
                        value_type=None, rel_type=None, max_results=None):
        """
        执行高级搜索，可组合多种条件
        
        参数:
            criteria: 值匹配函数
            metadata_filters: 元数据过滤 {key: value} 或 {key: [value1, value2]}
            tag_filters: 标签过滤列表
            value_type: 值类型名称
            rel_type: 关系类型
            max_results: 最大结果数量
            
        返回:
            符合条件的节点列表
        """
        # 不同条件的匹配结果
        start_time = time.time()
        matching_sets = []
        
        # 1. 按值匹配
        if criteria:
            matched = []
            for bucket in self.buckets:
                for node in bucket:
                    if criteria(node.val):
                        matched.append(node)
            matching_sets.append(set(matched))
        
        # 2. 按元数据匹配
        if metadata_filters:
            metadata_matched = set()
            for meta_key, meta_val in metadata_filters.items():
                # 支持多值匹配
                if isinstance(meta_val, list):
                    for val in meta_val:
                        metadata_matched.update(self.find_by_metadata(meta_key, val))
                else:
                    metadata_matched.update(self.find_by_metadata(meta_key, meta_val))
            matching_sets.append(metadata_matched)
        
        # 3. 按标签匹配
        if tag_filters:
            tag_matched = set()
            for tag in tag_filters:
                tag_matched.update(self.find_by_tag(tag))
            matching_sets.append(tag_matched)
        
        # 4. 按值类型匹配
        if value_type:
            matching_sets.append(set(self.find_by_value_type(value_type)))
        
        # 5. 按关系类型匹配
        if rel_type:
            if rel_type in self.relation_type_index:
                matching_sets.append(set(self.relation_type_index[rel_type]))
            else:
                matching_sets.append(set())
        
        # 如果没有条件，则返回所有节点
        if not matching_sets:
            result = []
            for bucket in self.buckets:
                result.extend(bucket)
            return result[:max_results] if max_results else result
        
        # 取交集
        result = set.intersection(*matching_sets)
        
        # 限制结果数量
        result_list = list(result)
        # 记录查询性能
        # 记录查询性能
        if hasattr(self, 'ml_engine'):
            execution_time = time.time() - start_time
            result_count = len(result_list) if result_list else 0
            self.ml_engine.record_query_performance('advanced_search', execution_time, result_count)
        return result_list[:max_results] if max_results else result_list
    
    def get_stats(self):
        """获取数据库状态统计"""
        bucket_loads = [len(bucket) for bucket in self.buckets]
        non_empty_buckets = sum(1 for load in bucket_loads if load > 0)
        
        total_relations = 0
        relation_types = set()
        for bucket in self.buckets:
            for node in bucket:
                total_relations += len(node.r)
                for rel in node.r:
                    if rel.get('type'):
                        relation_types.add(rel.get('type'))
        
        return {
            'node_count': self.count,
            'bucket_count': self.size,
            'load_factor': self.get_current_load_factor(),
            'max_bucket_load': max(bucket_loads) if bucket_loads else 0,
            'min_bucket_load': min(bucket_loads) if bucket_loads else 0,
            'avg_bucket_load': self.count / self.size if self.size > 0 else 0,
            'non_empty_buckets': non_empty_buckets,
            'bucket_utilization': non_empty_buckets / self.size if self.size > 0 else 0,
            'total_relations': total_relations,
            'relation_types_count': len(relation_types),
            'metadata_indices_count': len(self.metadata_index),
            'tag_indices_count': len(self.tag_index),
            'value_type_indices_count': len(self.value_type_index)
        }
    
    def optimize(self):
        """优化数据库结构，包括负载均衡和清理空桶"""
        # 计算每个桶的负载
        bucket_loads = [len(bucket) for bucket in self.buckets]
        avg_load = self.count / self.size
        
        print(f"当前状态: 总节点数={self.count}, 桶数量={self.size}, 平均负载={avg_load:.2f}")
        print(f"最大桶负载={max(bucket_loads)}, 最小桶负载={min(bucket_loads)}")
        
        # 如果负载不均衡，考虑重新哈希
        if max(bucket_loads) > avg_load * 3 and self.count > 1000:
            # 扩容以改善分布
            self._resize(self.size * 2)
            return True
            
        # 如果空间利用率过低，考虑缩容
        if self.get_current_load_factor() < 0.25 and self.size > 1024:
            self._resize(max(1024, self.size // 2))
            return True
            
        return False
    
    def save_to_file(self, filename):
        """保存数据库到文件"""
        try:
            print("开始保存数据库...")
            
            with open(filename, 'wb') as f:
                pickle.dump(self, f, protocol=2)
                
            print(f"成功保存到: {filename}")
            return True
        except Exception as e:
            print(f"保存失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    @classmethod
    def load_from_file(cls, filename, max_workers=4):
        """从文件加载数据库"""
        try:
            print(f"开始从 {filename} 加载数据库...")
            
            with open(filename, 'rb') as f:
                ion = pickle.load(f)
            
            # 如果加载的对象不是ION实例，尝试按旧格式加载
            if not isinstance(ion, cls):
                print("使用旧格式加载...")
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                    
                # 创建新实例
                ion = cls(size=data.get('size', 1024), 
                max_workers=max_workers,
                         load_factor_threshold=data.get('load_factor_threshold', 0.75))
                         
                # 使用_prepare_serialization格式的数据加载
                if 'buckets' in data and isinstance(data['buckets'], list):
                    # ... 恢复节点和索引的代码保持不变 ...
                    pass
            
            # 确保使用指定的max_workers
            if hasattr(ion, 'executor') and ion.executor:
                try:
                    ion.executor.shutdown(wait=False)
                except:
                    pass
                ion.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
            
            print("数据库加载成功")
            return ion
        except Exception as e:
            print(f"加载失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # 事件系统
    def _trigger_event(self, event_type, **kwargs):
        """触发事件回调"""
        if event_type in self.event_callbacks:
            for callback in self.event_callbacks[event_type]:
                try:
                    callback(**kwargs)
                except Exception as e:
                    print(f"事件回调错误: {e}")
                
    def register_callback(self, event_type, callback):
        """注册事件回调"""
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []
        self.event_callbacks[event_type].append(callback)
        return self
        
    def unregister_callback(self, event_type, callback):
        """注销事件回调"""
        if event_type in self.event_callbacks and callback in self.event_callbacks[event_type]:
            self.event_callbacks[event_type].remove(callback)
        return self
    
    # 多线程批处理
    def parallel_batch_process(self, input_list, process_func):
        """并行处理多个节点或操作"""
        futures = []
        results = []
        
        for item in input_list:
            future = self.executor.submit(process_func, item)
            futures.append(future)
            
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append(None)
                print(f"Error in parallel processing: {e}")
                
        return results
        
    def batch_create_nodes(self, node_data_list):
        """批量创建节点"""
        # 预先检查是否需要扩容
        if self.count + len(node_data_list) > self.size * self.load_factor_threshold:
            self._resize(self.size * 2)
        
        # 处理传入的不同数据格式
        def process_node_data(data):
            # 处理元组格式 (key, val) 或 (key, val, metadata)
            if isinstance(data, tuple):
                if len(data) == 2:
                    return self.create_node(data[0], data[1])
                elif len(data) == 3:
                    return self.create_node(data[0], data[1], data[2])
                elif len(data) >= 4:
                    return self.create_node(data[0], data[1], data[2], data[3])
                else:
                    print(f"错误：元组数据格式不正确 {data}")
                    return None
            # 处理字典格式 {'key': key, 'val': val, ...}
            elif isinstance(data, dict):
                if 'key' not in data:
                    print(f"错误：字典数据缺少'key'字段 {data}")
                    return None
                return self.create_node(
                    data['key'], 
                    data.get('val'),
                    data.get('metadata'),
                    data.get('tags'),
                    data.get('weight', 1.0)
                )
            else:
                print(f"错误：不支持的数据类型 {type(data)}")
                return None
        
        return self.parallel_batch_process(node_data_list, process_node_data)
        
    def batch_add_relationships(self, relationship_list):
        """批量添加关系"""
        return self.parallel_batch_process(
            relationship_list,
            lambda rel: self.add_relationship(
                rel['source'], 
                rel['target'], 
                rel.get('type'), 
                rel.get('weight', 1.0),
                rel.get('metadata')
            )
        )
        
    def batch_update_node_metadata(self, nodes, metadata_dict):
        """批量更新节点元数据"""
        return self.parallel_batch_process(
            nodes,
            lambda node: self.update_node_metadata(node, metadata_dict)
        )
        
    def batch_add_tags(self, nodes, tags):
        """批量添加标签"""
        def add_tags_to_node(node):
            node_obj = self._get_node_from_input(node)
            if node_obj:
                for tag in tags:
                    node_obj.add_tag(tag)
            return node_obj
            
        return self.parallel_batch_process(nodes, add_tags_to_node)
        
    def batch_update_node_weights(self, nodes, weight_value):
        """批量更新节点权重"""
        def update_weight(node):
            node_obj = self._get_node_from_input(node)
            if node_obj:
                node_obj.weight = weight_value
            return node_obj
            
        return self.parallel_batch_process(nodes, update_weight)
    
    # 字典类似方法
    def __getitem__(self, key):
        """支持通过索引访问值（ion[key]）"""
        node = self.get_node_by_key(key)
        if node is None:
            raise KeyError(f"键 '{key}' 不存在")
        return node
        
    def __setitem__(self, key, value):
        """支持通过索引设置值（ion[key] = value）"""
        if isinstance(value, dict) and 'val' in value:
            # 如果是包含节点信息的字典
            metadata = value.get('metadata')
            tags = value.get('tags')
            weight = value.get('weight', 1.0)
            self.create_node(key, value['val'], metadata, tags, weight)
        else:
            # 否则直接将value作为节点值
            self.create_node(key, value)
            
    def __delitem__(self, key):
        """支持通过del删除节点（del ion[key]）"""
        result = self.remove_node_by_key(key)
        if result is None:
            raise KeyError(f"键 '{key}' 不存在")
            
    def __contains__(self, key):
        """支持in操作符（key in ion）"""
        return self.get_node_by_key(key) is not None
        
    def __len__(self):
        """返回节点数量"""
        return self.count
        
    def keys(self):
        """返回所有键的迭代器"""
        for bucket in self.buckets:
            for node in bucket:
                yield node.key
                
    def values(self):
        """返回所有值的迭代器"""
        for bucket in self.buckets:
            for node in bucket:
                yield node.val
                
    def items(self):
        """返回所有键值对的迭代器"""
        for bucket in self.buckets:
            for node in bucket:
                yield (node.key, node.val)

    def fuzzy_search(self, query, search_in=None, max_results=50, similarity_threshold=0.6, case_sensitive=False):
        """
        模糊搜索节点，支持键、值、元数据和标签的模糊匹配
        
        参数:
            query: 搜索查询字符串
            search_in: 要搜索的字段列表 ['keys', 'values', 'metadata', 'tags']，默认全部
            max_results: 最大结果数量
            similarity_threshold: 相似度阈值 (0.0-1.0)，越高越精确
            case_sensitive: 是否区分大小写
            
        返回:
            {
                'keys': [(节点, 相似度), ...],
                'values': [(节点, 相似度), ...],
                'metadata': [(节点, 元数据键, 元数据值, 相似度), ...],
                'tags': [(节点, 标签, 相似度), ...]
            }
        """
        start_time = time.time()
        if search_in is None:
            search_in = ['keys', 'values', 'metadata', 'tags']
            
        results = {
            'keys': [],
            'values': [],
            'metadata': [],
            'tags': []
        }
        
        # 确保查询是字符串
        if not isinstance(query, str):
            query = str(query)
            
        # 如果不区分大小写，将查询转换为小写
        if not case_sensitive:
            query = query.lower()
            
        # 正则表达式模式 (用于额外的匹配方式)
        try:
            pattern = re.compile(query, re.IGNORECASE if not case_sensitive else 0)
            use_regex = True
        except re.error:
            use_regex = False
        
        # 用于字符串相似度计算
        def calculate_similarity(s1, s2):
            if not case_sensitive:
                s1 = s1.lower()
                
            # 使用不同的相似度度量方法
            # 1. 序列匹配比率
            ratio = difflib.SequenceMatcher(None, s1, s2).ratio()
            
            # 2. 包含关系检查 (子串匹配)
            contains_bonus = 0.3 if s2 in s1 else 0
            
            # 3. 前缀匹配额外加分
            prefix_bonus = 0.2 if s1.startswith(s2) else 0
            
            # 4. 正则表达式匹配 (如果可用)
            regex_bonus = 0.1 if use_regex and pattern.search(s1) else 0
            
            # 组合相似度分数 (最大为1.0)
            combined_score = min(1.0, ratio + contains_bonus + prefix_bonus + regex_bonus)
            
            return combined_score
        
        # 1. 搜索键
        if 'keys' in search_in:
            for bucket in self.buckets:
                for node in bucket:
                    if hasattr(node.key, '__str__'):  # 确保键可以转换为字符串
                        key_str = str(node.key)
                        similarity = calculate_similarity(key_str, query)
                        if similarity >= similarity_threshold:
                            results['keys'].append((node, similarity))
            
            # 按相似度排序
            results['keys'].sort(key=lambda x: x[1], reverse=True)
            if max_results:
                results['keys'] = results['keys'][:max_results]
        
        # 2. 搜索值
        if 'values' in search_in:
            for bucket in self.buckets:
                for node in bucket:
                    if hasattr(node.val, '__str__'):  # 确保值可以转换为字符串
                        val_str = str(node.val)
                        similarity = calculate_similarity(val_str, query)
                        if similarity >= similarity_threshold:
                            results['values'].append((node, similarity))
                    # 对于字典类型的值，搜索其内容
                    elif isinstance(node.val, dict):
                        for k, v in node.val.items():
                            if hasattr(k, '__str__') and hasattr(v, '__str__'):
                                k_str = str(k)
                                v_str = str(v)
                                k_similarity = calculate_similarity(k_str, query)
                                v_similarity = calculate_similarity(v_str, query)
                                max_similarity = max(k_similarity, v_similarity)
                                if max_similarity >= similarity_threshold:
                                    results['values'].append((node, max_similarity))
                                    break  # 找到匹配项后跳出内部循环
            
            # 按相似度排序
            results['values'].sort(key=lambda x: x[1], reverse=True)
            if max_results:
                results['values'] = results['values'][:max_results]
        
        # 3. 搜索元数据
        if 'metadata' in search_in:
            for bucket in self.buckets:
                for node in bucket:
                    if node.metadata:
                        for meta_key, meta_val in node.metadata.items():
                            # 检查元数据键
                            if hasattr(meta_key, '__str__'):
                                meta_key_str = str(meta_key)
                                key_similarity = calculate_similarity(meta_key_str, query)
                                if key_similarity >= similarity_threshold:
                                    results['metadata'].append((node, meta_key, meta_val, key_similarity))
                            
                            # 检查元数据值
                            if hasattr(meta_val, '__str__'):
                                meta_val_str = str(meta_val)
                                val_similarity = calculate_similarity(meta_val_str, query)
                                if val_similarity >= similarity_threshold:
                                    results['metadata'].append((node, meta_key, meta_val, val_similarity))
            
            # 按相似度排序
            results['metadata'].sort(key=lambda x: x[3], reverse=True)
            if max_results:
                results['metadata'] = results['metadata'][:max_results]
        
        # 4. 搜索标签
        if 'tags' in search_in:
            for bucket in self.buckets:
                for node in bucket:
                    if node.tags:
                        for tag in node.tags:
                            if hasattr(tag, '__str__'):
                                tag_str = str(tag)
                                similarity = calculate_similarity(tag_str, query)
                                if similarity >= similarity_threshold:
                                    results['tags'].append((node, tag, similarity))
            
            # 按相似度排序
            results['tags'].sort(key=lambda x: x[2], reverse=True)
            if max_results:
                results['tags'] = results['tags'][:max_results]
        # 记录查询性能
        if hasattr(self, 'ml_engine'):
            execution_time = time.time() - start_time
            result_count = len(results) if results else 0
            self.ml_engine.record_query_performance('fuzzy_search', execution_time, result_count)
        return results
        
    def advanced_fuzzy_search(self, query, options=None):
        """
        高级模糊搜索，支持更多自定义选项
        
        参数:
            query: 搜索查询
            options: 配置选项字典，可包含以下键:
                - search_in: 要搜索的字段列表
                - max_results: 最大结果数
                - similarity_threshold: 相似度阈值
                - case_sensitive: 是否区分大小写
                - match_mode: 匹配模式 ('contains', 'prefix', 'suffix', 'exact', 'regex', 'fuzzy', 'all')
                - combine_results: 是否合并所有类型的结果
                - similarity_func: 自定义相似度计算函数
                - value_transform_func: 值转换函数，用于预处理比较值
                - query_transform_func: 查询转换函数，用于预处理查询
                
        返回:
            单一列表或按类型分组的字典，取决于combine_results选项
        """
        # 默认选项
        default_options = {
            'search_in': ['keys', 'values', 'metadata', 'tags'],
            'max_results': 50,
            'similarity_threshold': 0.5,
            'case_sensitive': False,
            'match_mode': 'all',  # 'contains', 'prefix', 'suffix', 'exact', 'regex', 'fuzzy', 'all'
            'combine_results': False,
            'similarity_func': None,  # 自定义相似度函数
            'value_transform_func': None,  # 值转换函数
            'query_transform_func': None   # 查询转换函数
        }
        
        # 合并选项
        if options is None:
            options = {}
        for key, default_val in default_options.items():
            if key not in options:
                options[key] = default_val
        
        # 转换查询
        if options['query_transform_func']:
            transformed_query = options['query_transform_func'](query)
        else:
            transformed_query = query
            
        if not options['case_sensitive'] and isinstance(transformed_query, str):
            transformed_query = transformed_query.lower()
            
        # 匹配函数
        def match_value(value, query, mode):
            if not hasattr(value, '__str__'):
                return 0
                
            # 转换值
            if options['value_transform_func']:
                str_value = options['value_transform_func'](value)
            else:
                str_value = str(value)
                
            if not options['case_sensitive'] and isinstance(str_value, str):
                str_value = str_value.lower()
            
            # 如果有自定义相似度函数，使用它
            if options['similarity_func']:
                return options['similarity_func'](str_value, query)
                
            # 使用不同的匹配模式
            if mode == 'contains' or mode == 'all':
                if isinstance(query, str) and isinstance(str_value, str) and query in str_value:
                    return 0.8
            
            if mode == 'prefix' or mode == 'all':
                if isinstance(query, str) and isinstance(str_value, str) and str_value.startswith(query):
                    return 0.9
                    
            if mode == 'suffix' or mode == 'all':
                if isinstance(query, str) and isinstance(str_value, str) and str_value.endswith(query):
                    return 0.85
                    
            if mode == 'exact' or mode == 'all':
                if str_value == query:
                    return 1.0
                    
            if mode == 'regex' or mode == 'all':
                if isinstance(query, str) and isinstance(str_value, str):
                    try:
                        if re.search(query, str_value):
                            return 0.75
                    except re.error:
                        pass
                        
            if mode == 'fuzzy' or mode == 'all':
                if isinstance(query, str) and isinstance(str_value, str):
                    return difflib.SequenceMatcher(None, str_value, query).ratio()
                    
            return 0
        
        # 结果集
        results = {
            'keys': [],
            'values': [],
            'metadata': [],
            'tags': []
        }
        
        # 1. 搜索键
        if 'keys' in options['search_in']:
            for bucket in self.buckets:
                for node in bucket:
                    similarity = match_value(node.key, transformed_query, options['match_mode'])
                    if similarity >= options['similarity_threshold']:
                        results['keys'].append((node, similarity))
            
            results['keys'].sort(key=lambda x: x[1], reverse=True)
            if options['max_results']:
                results['keys'] = results['keys'][:options['max_results']]
        
        # 2. 搜索值
        if 'values' in options['search_in']:
            for bucket in self.buckets:
                for node in bucket:
                    # 直接匹配值
                    similarity = match_value(node.val, transformed_query, options['match_mode'])
                    
                    # 如果是字典，检查所有键值对
                    if similarity < options['similarity_threshold'] and isinstance(node.val, dict):
                        for k, v in node.val.items():
                            k_similarity = match_value(k, transformed_query, options['match_mode'])
                            v_similarity = match_value(v, transformed_query, options['match_mode'])
                            similarity = max(similarity, k_similarity, v_similarity)
                            if similarity >= options['similarity_threshold']:
                                break
                    
                    if similarity >= options['similarity_threshold']:
                        results['values'].append((node, similarity))
            
            results['values'].sort(key=lambda x: x[1], reverse=True)
            if options['max_results']:
                results['values'] = results['values'][:options['max_results']]
        
        # 3. 搜索元数据
        if 'metadata' in options['search_in']:
            for bucket in self.buckets:
                for node in bucket:
                    if node.metadata:
                        for meta_key, meta_val in node.metadata.items():
                            key_similarity = match_value(meta_key, transformed_query, options['match_mode'])
                            val_similarity = match_value(meta_val, transformed_query, options['match_mode'])
                            max_similarity = max(key_similarity, val_similarity)
                            
                            if max_similarity >= options['similarity_threshold']:
                                results['metadata'].append((node, meta_key, meta_val, max_similarity))
            
            results['metadata'].sort(key=lambda x: x[3], reverse=True)
            if options['max_results']:
                results['metadata'] = results['metadata'][:options['max_results']]
        
        # 4. 搜索标签
        if 'tags' in options['search_in']:
            for bucket in self.buckets:
                for node in bucket:
                    for tag in node.tags:
                        similarity = match_value(tag, transformed_query, options['match_mode'])
                        if similarity >= options['similarity_threshold']:
                            results['tags'].append((node, tag, similarity))
            
            results['tags'].sort(key=lambda x: x[2], reverse=True)
            if options['max_results']:
                results['tags'] = results['tags'][:options['max_results']]
        
        # 合并结果（如果需要）
        if options['combine_results']:
            combined = []
            for node, similarity in results['keys']:
                combined.append(('key', node, similarity))
                
            for node, similarity in results['values']:
                combined.append(('value', node, similarity))
                
            for node, meta_key, meta_val, similarity in results['metadata']:
                combined.append(('metadata', node, meta_key, meta_val, similarity))
                
            for node, tag, similarity in results['tags']:
                combined.append(('tag', node, tag, similarity))
                
            # 按相似度排序
            combined.sort(key=lambda x: x[-1], reverse=True)
            if options['max_results']:
                combined = combined[:options['max_results']]
                
            return combined
            
        return results

    def update_node_tag(self, node, operation='add', tags=None, clear_existing=False):
        """
        更新节点的标签
        
        参数:
            node: 要更新的节点或节点键
            operation: 操作类型，'add'(添加)、'remove'(移除)或'set'(设置)
            tags: 要添加、移除或设置的标签，可以是单个标签或标签列表
            clear_existing: 是否清除现有标签(与set操作配合使用)
            
        返回:
            更新后的节点
        """
        # 获取节点对象
        node_obj = self._get_node_from_input(node)
        if not node_obj:
            raise ValueError(f"找不到节点: {node}")
            
        # 确保tags是可迭代的（如果是单个字符串，转换为列表）
        if tags is not None:
            if isinstance(tags, str):
                tags = [tags]
            elif not hasattr(tags, '__iter__'):
                tags = [tags]  # 处理其他单个非迭代对象
            
        # 根据操作类型处理标签
        if operation == 'add' and tags:
            # 添加新标签
            for tag in tags:
                node_obj.add_tag(tag)
                
        elif operation == 'remove' and tags:
            # 移除指定标签
            for tag in tags:
                node_obj.remove_tag(tag)
                
        elif operation == 'set':
            # 如果需要清除现有标签
            if clear_existing:
                # 保存现有标签的副本，以便稍后清除索引
                old_tags = set(node_obj.tags)
                
                # 清空标签集合
                node_obj._tags.clear()
                
                # 移除旧标签的索引
                for tag in old_tags:
                    self._remove_tag_index(tag, node_obj)
            
            # 设置新标签
            if tags:
                for tag in tags:
                    node_obj.add_tag(tag)
                    
        # 触发节点更新事件
        self._trigger_event('node_updated', node=node_obj, tags_changed=True)
        
        return node_obj
        
    def update_node_weight(self, node, weight):
        """
        更新节点的权重
        
        参数:
            node: 要更新的节点或节点键
            weight: 新的权重值(浮点数)
            
        返回:
            更新后的节点
        """
        # 获取节点对象
        node_obj = self._get_node_from_input(node)
        if not node_obj:
            raise ValueError(f"找不到节点: {node}")
            
        # 保存旧权重以便事件通知
        old_weight = node_obj.weight
        
        # 更新权重
        try:
            node_obj.weight = float(weight)
        except (TypeError, ValueError):
            raise ValueError(f"权重必须是一个有效的数值: {weight}")
            
        # 触发节点更新事件
        self._trigger_event('node_updated', node=node_obj, old_weight=old_weight)
        
        return node_obj
        
    def get_node_tags(self, node):
        """
        获取节点的所有标签
        
        参数:
            node: 节点或节点键
            
        返回:
            包含所有标签的集合
        """
        node_obj = self._get_node_from_input(node)
        return set(node_obj.tags) if node_obj else set()
        
    def get_nodes_by_tag(self, tag):
        """
        获取具有指定标签的所有节点
        
        参数:
            tag: 标签
            
        返回:
            节点列表
        """
        return self.find_by_tag(tag)
        
    def get_nodes_by_weight_range(self, min_weight=None, max_weight=None):
        """
        获取权重在指定范围内的所有节点
        
        参数:
            min_weight: 最小权重(包含)，None表示不限制
            max_weight: 最大权重(包含)，None表示不限制
            
        返回:
            符合条件的节点列表
        """
        result = []
        
        # 创建过滤函数
        def weight_filter(node):
            if min_weight is not None and node.weight < min_weight:
                return False
            if max_weight is not None and node.weight > max_weight:
                return False
            return True
            
        # 遍历所有节点
        for bucket in self.buckets:
            for node in bucket:
                if weight_filter(node):
                    result.append(node)
                    
        return result
        
    def batch_update_tags(self, node_list, operation='add', tags=None, clear_existing=False):
        """
        批量更新多个节点的标签
        
        参数:
            node_list: 要更新的节点列表
            operation: 操作类型，'add'(添加)、'remove'(移除)或'set'(设置)
            tags: 要添加、移除或设置的标签，可以是单个标签或标签列表
            clear_existing: 是否清除现有标签(与set操作配合使用)
            
        返回:
            成功更新的节点列表
        """
        # 确保tags参数格式正确
        if tags is not None:
            if isinstance(tags, str):
                tags = [tags]  # 单个字符串标签
            elif not hasattr(tags, '__iter__') or isinstance(tags, dict):
                tags = [tags]  # 其他非迭代对象或字典
        
        def update_tags(node):
            try:
                return self.update_node_tag(node, operation, tags, clear_existing)
            except Exception as e:
                print(f"更新节点 {node} 的标签时出错: {e}")
                return None
                
            return self.parallel_batch_process(node_list, update_tags)
        
    def batch_update_weights(self, nodes_weights):
        """
        批量更新多个节点的权重
        
        参数:
            nodes_weights: [(node, weight), ...] 格式的列表
            
        返回:
            成功更新的节点列表
        """
        def update_weight(item):
            try:
                node, weight = item
                return self.update_node_weight(node, weight)
            except Exception as e:
                print(f"更新节点权重时出错: {e}")
                return None
                
        return self.parallel_batch_process(nodes_weights, update_weight)
    
    def __getstate__(self):
        """自定义序列化行为，排除不可序列化的对象"""
        state = self.__dict__.copy()
        
        # 移除所有锁对象和线程池
        state.pop('lock', None)
        state.pop('resizing_lock', None)
        state.pop('bucket_locks', None)
        state.pop('executor', None)
        
        # 处理事件回调（函数可能无法序列化）
        state.pop('event_callbacks', None)
        
        # 处理循环引用
        # 将节点中对ION的引用置为None，防止循环引用
        for bucket in state['buckets']:
            for node in bucket:
                if hasattr(node, '_ion_reference'):
                    node._ion_reference = None
        
        # 处理ICombinedDataStructure（它可能也包含锁）
        if '_combined_data' in state:
            try:
                # 尝试使用其可能实现的自定义序列化方法
                combined_data = state['_combined_data']
                if hasattr(combined_data, '__getstate__'):
                    pass  # 让它使用自己的__getstate__
                else:
                    # 如果没有自定义序列化，我们移除它
                    state['_combined_data'] = None
            except:
                state['_combined_data'] = None
        
        return state
    
    def __setstate__(self, state):
        """自定义反序列化行为，重新创建不可序列化的对象"""
        self.__dict__.update(state)
        
        # 重新创建所有锁对象
        self.lock = threading.RLock()
        self.resizing_lock = threading.RLock()
        self.bucket_locks = [threading.RLock() for _ in range(self.size)]
        
        # 创建新的线程池
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        # 重新初始化事件回调系统
        self.event_callbacks = {
            'node_added': [],
            'node_updated': [],
            'node_removed': [],
            'relation_added': [],
            'relation_removed': [],
            'resize': [],
            'transaction_started': [],
            'transaction_committed': [],
            'transaction_aborted': []
        }
        
        # 重新创建或更新ICombinedDataStructure
        if not hasattr(self, '_combined_data') or self._combined_data is None:
            self._combined_data = ICombinedDataStructure(
                initial_capacity=self.size, 
                load_factor=self.load_factor_threshold
            )
            
            # 从ION节点数据重建combined_data
            for bucket in self.buckets:
                for node in bucket:
                    self._combined_data.put(
                        node.key, 
                        node.val, 
                        metadata=node.metadata
                    )
                    
                    # 恢复关系
                    for relation in node.r:
                        target_node = relation['node']
                        if hasattr(target_node, 'key'):
                            self._combined_data.add_relationship(
                                node.key, 
                                target_node.key
                            )
        
        # 恢复节点到ION的引用
        for bucket in self.buckets:
            for node in bucket:
                if hasattr(node, '_ion_reference'):
                    node._ion_reference = self
        
        return self

    def astar_search(self, start, goal, heuristic_func=None, weight_func=None, rel_type=None, max_iterations=10000):
        """
        使用A*算法搜索从起点到终点的最优路径
        
        参数:
            start: 起始节点或节点键
            goal: 目标节点或节点键
            heuristic_func: 启发函数，计算当前节点到目标的估计距离，格式为 func(current, goal) -> float
                            默认为空间欧几里得距离(如果节点有坐标)或1.0
            weight_func: 边权重函数，计算两个相邻节点间的实际距离，格式为 func(current, next, relation) -> float
                         默认使用关系的weight属性或1.0
            rel_type: 要考虑的关系类型，None表示所有类型
            max_iterations: 最大迭代次数，防止无限循环
            
        返回:
            (path, cost): 路径节点列表和总成本的元组，如果没有路径则返回(None, float('inf'))
        """
        # 获取节点对象
        start_node = self._get_node_from_input(start)
        goal_node = self._get_node_from_input(goal)
        
        if not start_node or not goal_node:
            return None, float('inf')
            
        # 特殊情况：如果起点和终点是同一个节点
        if start_node == goal_node:
            return [start_node], 0.0
            
        # 默认启发函数 - 如果节点有x,y坐标属性则使用欧几里得距离，否则返回1
        if heuristic_func is None:
            def default_heuristic(current, goal):
                # 如果节点有坐标属性，使用欧几里得距离
                if (hasattr(current, 'metadata') and hasattr(goal, 'metadata') and
                    'x' in current.metadata and 'y' in current.metadata and
                    'x' in goal.metadata and 'y' in goal.metadata):
                    return ((current.metadata['x'] - goal.metadata['x']) ** 2 + 
                            (current.metadata['y'] - goal.metadata['y']) ** 2) ** 0.5
                # 否则使用关系权重
                elif hasattr(current, 'weight') and hasattr(goal, 'weight'):
                    return abs(current.weight - goal.weight)
                # 默认返回1
                return 1.0
                
            heuristic_func = default_heuristic
            
        # 默认权重函数 - 使用关系的weight属性或1.0
        if weight_func is None:
            def default_weight_func(current, next_node, relation):
                return relation.get('weight', 1.0)
                
            weight_func = default_weight_func
        
        # 初始化数据结构
        open_set = []  # 优先队列，(f, 节点)
        closed_set = set()  # 已访问节点
        
        # 记录来源节点和g值
        came_from = {}  # 节点 -> 前驱节点
        g_score = {start_node: 0}  # 节点 -> 从起点到该节点的成本
        f_score = {start_node: heuristic_func(start_node, goal_node)}  # 节点 -> f值
        
        # 将起始节点加入开放集
        heapq.heappush(open_set, (f_score[start_node], id(start_node), start_node))
        
        iterations = 0
        
        # A*搜索主循环
        while open_set and iterations < max_iterations:
            iterations += 1
            
            # 取出f值最小的节点
            _, _, current = heapq.heappop(open_set)
            
            # 如果到达目标，构建并返回路径
            if current == goal_node:
                path = self._reconstruct_path(came_from, current)
                return path, g_score[current]
                
            # 将当前节点加入已访问集
            closed_set.add(current)
            
            # 获取当前节点的所有相邻节点
            for relation in current.r:
                # 如果指定了关系类型，跳过不匹配的关系
                if rel_type is not None and relation.get('type') != rel_type:
                    continue
                    
                next_node = relation['node']
                
                # 如果邻居已经访问过，跳过
                if next_node in closed_set:
                    continue
                
                # 计算通过当前节点到达邻居的g值
                tentative_g = g_score[current] + weight_func(current, next_node, relation)
                
                # 如果找到了更好的路径 或 这是第一次看到这个邻居
                if next_node not in g_score or tentative_g < g_score[next_node]:
                    # 更新路径和分数
                    came_from[next_node] = current
                    g_score[next_node] = tentative_g
                    f_score[next_node] = tentative_g + heuristic_func(next_node, goal_node)
                    
                    # 将邻居加入开放集
                    if not any(node == next_node for _, _, node in open_set):
                        heapq.heappush(open_set, (f_score[next_node], id(next_node), next_node))
        
        # 如果无法到达目标节点
        return None, float('inf')
    
    def _reconstruct_path(self, came_from, current):
        """从来源映射中重建完整路径"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()  # 反转路径，使其从起点开始
        return path
        
    def astar_search_advanced(self, start, goal, options=None):
        """
        高级版A*搜索，支持更多自定义选项
        
        参数:
            start: 起始节点或节点键
            goal: 目标节点或节点键
            options: 配置选项字典，可包含以下键:
                - heuristic: 启发函数
                - weight_func: 权重函数
                - rel_type: 关系类型
                - max_iterations: 最大迭代次数
                - bidirectional: 是否使用双向A*算法
                - admissible_heuristic: 是否使用可采纳启发（不高估真实代价）
                - cost_limit: 最大成本限制
                - weight_property: 使用哪个节点属性作为权重
                - tie_breaking: 相同f值时的排序方式
                - multi_goal: 是否支持多目标搜索（goal可以是列表）
                - early_stop: 提前终止条件函数
                
        返回:
            (path, cost, stats): 路径、总成本和搜索统计信息的元组
        """
        # 默认选项
        default_options = {
            'heuristic': None,
            'weight_func': None,
            'rel_type': None,
            'max_iterations': 10000,
            'bidirectional': False,
            'admissible_heuristic': True,
            'cost_limit': float('inf'),
            'weight_property': 'weight',
            'tie_breaking': 'path_length',  # 'path_length', 'g_value', 'h_value'
            'multi_goal': False,
            'early_stop': None,  # 自定义提前终止函数
            'include_stats': False  # 是否包含搜索统计信息
        }
        
        # 合并选项
        if options is None:
            options = {}
        for key, default_val in default_options.items():
            if key not in options:
                options[key] = default_val
        
        # 获取节点对象
        start_node = self._get_node_from_input(start)
        
        # 处理多目标情况
        if options['multi_goal'] and isinstance(goal, (list, tuple, set)):
            goal_nodes = [self._get_node_from_input(g) for g in goal]
            goal_nodes = [g for g in goal_nodes if g is not None]
        else:
            goal_nodes = [self._get_node_from_input(goal)]
            
        if not goal_nodes or not start_node:
            return None, float('inf'), {}
            
        # 如果起点和任一终点是同一个节点
        if start_node in goal_nodes:
            return [start_node], 0.0, {'iterations': 0, 'visited': 1}
        
        # 权重属性
        weight_prop = options['weight_property']
        
        # 默认启发函数
        if options['heuristic'] is None:
            def default_heuristic(current, goal):
                # 如果节点有坐标属性，使用欧几里得距离
                if (hasattr(current, 'metadata') and hasattr(goal, 'metadata') and
                    'x' in current.metadata and 'y' in current.metadata and
                    'x' in goal.metadata and 'y' in goal.metadata):
                    return ((current.metadata['x'] - goal.metadata['x']) ** 2 + 
                            (current.metadata['y'] - goal.metadata['y']) ** 2) ** 0.5
                # 使用指定属性作为权重
                elif hasattr(current, weight_prop) and hasattr(goal, weight_prop):
                    try:
                        return abs(getattr(current, weight_prop) - getattr(goal, weight_prop))
                    except:
                        return 1.0
                # 尝试使用元数据中的属性
                elif (hasattr(current, 'metadata') and hasattr(goal, 'metadata') and
                      weight_prop in current.metadata and weight_prop in goal.metadata):
                    try:
                        return abs(current.metadata[weight_prop] - goal.metadata[weight_prop])
                    except:
                        return 1.0
                # 默认返回1
                return 1.0
                
            options['heuristic'] = default_heuristic
            
        # 默认权重函数
        if options['weight_func'] is None:
            def default_weight_func(current, next_node, relation):
                if 'weight' in relation:
                    return relation['weight']
                # 尝试使用指定属性作为权重
                elif (hasattr(current, weight_prop) and hasattr(next_node, weight_prop)):
                    try:
                        return abs(getattr(next_node, weight_prop) - getattr(current, weight_prop))
                    except:
                        return 1.0
                return 1.0
                
            options['weight_func'] = default_weight_func
            
        # 统计信息
        stats = {
            'iterations': 0,
            'visited': 0,
            'expanded': 0,
            'max_queue_size': 0
        }
        
        # 根据是否使用双向搜索选择不同的算法
        if options['bidirectional']:
            return self._bidirectional_astar(start_node, goal_nodes, options, stats)
        else:
            return self._standard_astar(start_node, goal_nodes, options, stats)
    
    def _standard_astar(self, start_node, goal_nodes, options, stats):
        """标准A*搜索实现"""
        # 提取选项
        heuristic_func = options['heuristic']
        weight_func = options['weight_func']
        rel_type = options['rel_type']
        max_iterations = options['max_iterations']
        cost_limit = options['cost_limit']
        tie_breaking = options['tie_breaking']
        early_stop = options['early_stop']
        
        # 初始化数据结构
        open_set = []  # 优先队列
        closed_set = set()  # 已访问节点
        
        # 记录来源节点和g值
        came_from = {}  # 节点 -> 前驱节点
        g_score = {start_node: 0}  # 节点 -> 起点到该节点的成本
        
        # 计算到每个目标的启发值，取最小值
        def min_heuristic(node):
            return min(heuristic_func(node, goal) for goal in goal_nodes)
            
        f_score = {start_node: min_heuristic(start_node)}
        
        # 加入起始节点
        # 使用不同的tie-breaking策略
        if tie_breaking == 'path_length':
            tie_breaker = 0  # 路径长度（优先短路径）
        elif tie_breaking == 'g_value':
            tie_breaker = g_score[start_node]  # g值（优先总成本低）
        elif tie_breaking == 'h_value':
            tie_breaker = min_heuristic(start_node)  # h值（优先离目标近）
        else:
            tie_breaker = 0
            
        heapq.heappush(open_set, (f_score[start_node], tie_breaker, id(start_node), start_node))
        
        # 最大队列大小统计
        stats['max_queue_size'] = 1
        
        # A*搜索主循环
        while open_set and stats['iterations'] < max_iterations:
            stats['iterations'] += 1
            
            # 取出f值最小的节点
            _, _, _, current = heapq.heappop(open_set)
            
            # 更新统计
            stats['visited'] += 1
            
            # 如果到达目标，构建并返回路径
            if current in goal_nodes:
                path = self._reconstruct_path(came_from, current)
                return path, g_score[current], stats
                
            # 如果提供了提前终止条件，检查是否满足
            if early_stop and early_stop(current, g_score[current], stats):
                path = self._reconstruct_path(came_from, current)
                return path, g_score[current], stats
                
            # 将当前节点加入已访问集
            closed_set.add(current)
            
            # 获取所有相邻节点
            for relation in current.r:
                # 如果指定了关系类型，跳过不匹配的关系
                if rel_type is not None and relation.get('type') != rel_type:
                    continue
                    
                next_node = relation['node']
                
                # 如果邻居已访问过，跳过
                if next_node in closed_set:
                    continue
                
                # 计算通过当前节点到达邻居的g值
                tentative_g = g_score[current] + weight_func(current, next_node, relation)
                
                # 如果超出成本限制，跳过
                if tentative_g > cost_limit:
                    continue
                
                # 如果找到了更好的路径 或 这是第一次看到这个邻居
                if next_node not in g_score or tentative_g < g_score[next_node]:
                    # 更新路径和分数
                    came_from[next_node] = current
                    g_score[next_node] = tentative_g
                    f_score[next_node] = tentative_g + min_heuristic(next_node)
                    
                    # 根据tie-breaking策略计算额外的优先级信息
                    if tie_breaking == 'path_length':
                        path_length = 0
                        temp = next_node
                        while temp in came_from:
                            path_length += 1
                            temp = came_from[temp]
                        tie_breaker = path_length
                    elif tie_breaking == 'g_value':
                        tie_breaker = tentative_g
                    elif tie_breaking == 'h_value':
                        tie_breaker = min_heuristic(next_node)
                    else:
                        tie_breaker = 0
                    
                    # 将邻居加入开放集
                    if not any(node == next_node for _, _, _, node in open_set):
                        heapq.heappush(open_set, (f_score[next_node], tie_breaker, id(next_node), next_node))
                        
                        # 更新统计
                        stats['expanded'] += 1
                        stats['max_queue_size'] = max(stats['max_queue_size'], len(open_set))
        
        # 如果无法到达目标节点
        return None, float('inf'), stats
    
    def _bidirectional_astar(self, start_node, goal_nodes, options, stats):
        """双向A*搜索实现（仅支持单一目标）"""
        # 双向搜索仅支持单一目标
        if len(goal_nodes) != 1:
            print("警告: 双向A*搜索仅支持单一目标, 使用第一个目标")
        goal_node = goal_nodes[0]
        
        # 提取选项
        heuristic_func = options['heuristic']
        weight_func = options['weight_func']
        rel_type = options['rel_type']
        max_iterations = options['max_iterations']
        cost_limit = options['cost_limit']
        
        # 正向搜索的数据结构
        open_set_fwd = []  # 优先队列
        closed_set_fwd = set()  # 已访问节点
        g_score_fwd = {start_node: 0}  # 节点 -> 从起点到该节点的成本
        f_score_fwd = {start_node: heuristic_func(start_node, goal_node)}
        came_from_fwd = {}  # 节点 -> 前驱节点
        
        # 反向搜索的数据结构
        open_set_bwd = []  # 优先队列
        closed_set_bwd = set()  # 已访问节点
        g_score_bwd = {goal_node: 0}  # 节点 -> 从终点到该节点的成本
        f_score_bwd = {goal_node: heuristic_func(goal_node, start_node)}
        came_from_bwd = {}  # 节点 -> 前驱节点
        
        # 将起始节点和目标节点加入对应的开放集
        heapq.heappush(open_set_fwd, (f_score_fwd[start_node], id(start_node), start_node))
        heapq.heappush(open_set_bwd, (f_score_bwd[goal_node], id(goal_node), goal_node))
        
        # 最佳相遇点和对应的成本
        best_meeting_point = None
        best_cost = float('inf')
        
        # 双向A*搜索主循环
        while open_set_fwd and open_set_bwd and stats['iterations'] < max_iterations:
            stats['iterations'] += 1
            
            # 决定从哪个方向扩展
            if len(open_set_fwd) <= len(open_set_bwd):
                # 从正向扩展
                current_f, _, current = heapq.heappop(open_set_fwd)
                
                # 如果当前节点的f值已经大于最佳成本，就可以提前终止
                if current_f >= best_cost:
                    break
                
                # 如果当前节点已经被反向搜索访问过，可能找到了更好的路径
                if current in g_score_bwd:
                    path_cost = g_score_fwd[current] + g_score_bwd[current]
                    if path_cost < best_cost:
                        best_meeting_point = current
                        best_cost = path_cost
                
                # 如果当前节点已经被访问过，跳过
                if current in closed_set_fwd:
                    continue
                
                stats['visited'] += 1
                closed_set_fwd.add(current)
                
                # 扩展邻居
                for relation in current.r:
                    if rel_type is not None and relation.get('type') != rel_type:
                        continue
                        
                    next_node = relation['node']
                    
                    if next_node in closed_set_fwd:
                        continue
                    
                    tentative_g = g_score_fwd[current] + weight_func(current, next_node, relation)
                    
                    if tentative_g > cost_limit:
                        continue
                    
                    if next_node not in g_score_fwd or tentative_g < g_score_fwd[next_node]:
                        came_from_fwd[next_node] = current
                        g_score_fwd[next_node] = tentative_g
                        f_score_fwd[next_node] = tentative_g + heuristic_func(next_node, goal_node)
                        
                        # 检查是否与反向搜索相遇
                        if next_node in g_score_bwd:
                            path_cost = tentative_g + g_score_bwd[next_node]
                            if path_cost < best_cost:
                                best_meeting_point = next_node
                                best_cost = path_cost
                        
                        heapq.heappush(open_set_fwd, (f_score_fwd[next_node], id(next_node), next_node))
                        stats['expanded'] += 1
                        stats['max_queue_size'] = max(stats['max_queue_size'], len(open_set_fwd) + len(open_set_bwd))
            else:
                # 从反向扩展（逻辑类似）
                current_f, _, current = heapq.heappop(open_set_bwd)
                
                if current_f >= best_cost:
                    break
                
                if current in g_score_fwd:
                    path_cost = g_score_fwd[current] + g_score_bwd[current]
                    if path_cost < best_cost:
                        best_meeting_point = current
                        best_cost = path_cost
                
                if current in closed_set_bwd:
                    continue
                
                stats['visited'] += 1
                closed_set_bwd.add(current)
                
                # 注意：反向搜索需要找到哪些节点有指向当前节点的关系
                # 这个操作可能比较昂贵，因为我们没有存储入边
                for bucket in self.buckets:
                    for node in bucket:
                        for relation in node.r:
                            if relation['node'] == current:
                                if rel_type is not None and relation.get('type') != rel_type:
                                    continue
                                    
                                prev_node = node
                                
                                if prev_node in closed_set_bwd:
                                    continue
                                
                                tentative_g = g_score_bwd[current] + weight_func(prev_node, current, relation)
                                
                                if tentative_g > cost_limit:
                                    continue
                                
                                if prev_node not in g_score_bwd or tentative_g < g_score_bwd[prev_node]:
                                    came_from_bwd[prev_node] = current
                                    g_score_bwd[prev_node] = tentative_g
                                    f_score_bwd[prev_node] = tentative_g + heuristic_func(prev_node, start_node)
                                    
                                    if prev_node in g_score_fwd:
                                        path_cost = g_score_fwd[prev_node] + tentative_g
                                        if path_cost < best_cost:
                                            best_meeting_point = prev_node
                                            best_cost = path_cost
                                    
                                    heapq.heappush(open_set_bwd, (f_score_bwd[prev_node], id(prev_node), prev_node))
                                    stats['expanded'] += 1
                                    stats['max_queue_size'] = max(stats['max_queue_size'], len(open_set_fwd) + len(open_set_bwd))
        
        # 如果找到了相遇点，构建完整路径
        if best_meeting_point is not None and best_cost < float('inf'):
            # 构建正向路径（从起点到相遇点）
            forward_path = self._reconstruct_path(came_from_fwd, best_meeting_point)
            
            # 构建反向路径（从终点到相遇点）
            backward_path = self._reconstruct_path(came_from_bwd, best_meeting_point)
            backward_path.reverse()
            
            # 合并路径（不重复添加相遇点）
            complete_path = forward_path + backward_path[1:]
            
            return complete_path, best_cost, stats
            
        # 没有找到路径
        return None, float('inf'), stats

    # ---- 事务管理方法 ----
    
    def begin_transaction(self, isolation_level=None, timeout=None):
        """
        开始新事务
        
        参数:
            isolation_level: 事务隔离级别，默认使用ION实例的默认隔离级别
            timeout: 事务超时时间，默认使用ION实例的默认超时时间
            
        返回:
            新创建的事务对象
        """
        if isolation_level is None:
            isolation_level = self.default_isolation_level
            
        if timeout is None:
            timeout = self.transaction_timeout
            
        # 创建新事务
        transaction = Transaction(self, isolation_level, timeout)
        
        # 注册事务
        with self.lock:
            self.active_transactions[transaction.id] = transaction
            self.transaction_counter += 1
            
        # 触发事件
        self._trigger_event('transaction_started', transaction=transaction)
        
        return transaction
    
    def commit_transaction(self, transaction_id):
        """
        提交指定事务
        
        参数:
            transaction_id: 要提交的事务ID
            
        返回:
            bool: 是否成功提交
            
        异常:
            TransactionError: 如果事务不存在或无法提交
        """
        with self.lock:
            if transaction_id not in self.active_transactions:
                raise TransactionError(f"事务不存在: {transaction_id}")
                
            transaction = self.active_transactions[transaction_id]
            
        # 提交事务
        result = transaction.commit()
        
        # 清理
        with self.lock:
            if transaction_id in self.active_transactions:
                del self.active_transactions[transaction_id]
                
        return result
    
    def abort_transaction(self, transaction_id, reason=None):
        """
        中止指定事务
        
        参数:
            transaction_id: 要中止的事务ID
            reason: 中止原因
            
        返回:
            bool: 是否成功中止
            
        异常:
            TransactionError: 如果事务不存在
        """
        with self.lock:
            if transaction_id not in self.active_transactions:
                raise TransactionError(f"事务不存在: {transaction_id}")
                
            transaction = self.active_transactions[transaction_id]
            
        # 中止事务
        result = transaction.abort(reason=reason)
        
        # 清理
        with self.lock:
            if transaction_id in self.active_transactions:
                del self.active_transactions[transaction_id]
                
        return result
    
    def get_transaction(self, transaction_id):
        """
        获取指定事务对象
        
        参数:
            transaction_id: 事务ID
            
        返回:
            Transaction: 事务对象，不存在则返回None
        """
        with self.lock:
            return self.active_transactions.get(transaction_id)
    
    def get_active_transactions(self):
        """
        获取所有活跃事务
        
        返回:
            dict: 事务ID -> 事务对象
        """
        with self.lock:
            return dict(self.active_transactions)
    
    def _check_transaction_conflicts(self, transaction):
        """
        检查事务冲突
        
        参数:
            transaction: 要检查的事务对象
            
        异常:
            ConcurrencyError: 如果检测到冲突
        """
        # 根据并发控制模式选择不同的冲突检测策略
        if self.concurrency_mode == 'optimistic':
            self._check_optimistic_conflicts(transaction)
        else:
            # 悲观并发控制依赖于锁，不需要额外的冲突检测
            pass
    
    def _check_optimistic_conflicts(self, transaction):
        """
        乐观并发控制的冲突检测
        
        参数:
            transaction: 要检查的事务对象
            
        异常:
            ConcurrencyError: 如果检测到冲突
        """
        # 根据隔离级别进行不同的检查
        if transaction.isolation_level == IsolationLevel.SERIALIZABLE:
            # 对于串行化级别，检查读取集和写入集是否与其他事务冲突
            for other_txn_id, other_txn in self.active_transactions.items():
                if other_txn_id == transaction.id or other_txn.state != TransactionState.ACTIVE:
                    continue
                    
                # 检查是否有冲突
                if (transaction.modified_nodes.intersection(other_txn.read_set) or 
                    transaction.read_set.intersection(other_txn.modified_nodes)):
                    raise ConcurrencyError(f"事务 {transaction.id} 与事务 {other_txn_id} 存在冲突")
        
        # 对于READ_COMMITTED和REPEATABLE_READ，检查节点的版本号
        if transaction.isolation_level in (IsolationLevel.READ_COMMITTED, IsolationLevel.REPEATABLE_READ):
            for node_key in transaction.modified_nodes:
                if node_key in self.node_versions:
                    current_version = self.node_versions[node_key]
                    # 获取事务中记录的版本号
                    for entry in transaction.log:
                        if entry.operation in ('update', 'delete') and entry.node_key == node_key:
                            if hasattr(entry, 'version') and entry.version < current_version:
                                raise ConcurrencyError(f"节点 {node_key} 已被其他事务修改")
    
    def _release_transaction_locks(self, transaction_id):
        """
        释放事务持有的所有锁
        
        参数:
            transaction_id: 事务ID
            
        返回:
            bool: 是否成功释放所有锁
        """
        return self.lock_manager.release_lock(transaction_id)
    
    def _rollback_create_node(self, node_key):
        """回滚创建节点操作"""
        self.remove_node_by_key(node_key)
    
    def _rollback_update_node(self, node_key, old_value, old_metadata=None, 
                             old_tags=None, old_relationships=None):
        """回滚更新节点操作"""
        node = self.get_node_by_key(node_key)
        if node:
            # 恢复旧值
            node._val = old_value
            
            # 恢复旧元数据
            if old_metadata is not None:
                node._metadata = old_metadata.copy() if old_metadata else {}
                
            # 恢复旧标签
            if old_tags is not None:
                node._tags = set(old_tags)
                
            # 恢复旧关系
            if old_relationships is not None:
                node._r = old_relationships.copy() if old_relationships else []
    
    def _rollback_delete_node(self, node_key, old_value, old_metadata=None, 
                             old_tags=None, old_relationships=None):
        """回滚删除节点操作"""
        # 重新创建被删除的节点
        new_node = self.create_node(
            node_key, old_value, 
            metadata=old_metadata, 
            tags=old_tags
        )
        
        # 如果有关系信息，恢复关系
        if old_relationships:
            new_node._r = old_relationships.copy()
    
    def _rollback_add_relationship(self, relationship_info):
        """回滚添加关系操作"""
        source_key = relationship_info.get('source')
        target_key = relationship_info.get('target')
        rel_type = relationship_info.get('type')
        
        if source_key and target_key:
            source_node = self.get_node_by_key(source_key)
            target_node = self.get_node_by_key(target_key)
            
            if source_node and target_node:
                self.remove_relationship(source_node, target_node, rel_type)
    
    def _rollback_remove_relationship(self, relationship_info):
        """回滚删除关系操作"""
        source_key = relationship_info.get('source')
        target_key = relationship_info.get('target')
        rel_type = relationship_info.get('type')
        rel_weight = relationship_info.get('weight', 1.0)
        rel_metadata = relationship_info.get('metadata')
        
        if source_key and target_key:
            source_node = self.get_node_by_key(source_key)
            target_node = self.get_node_by_key(target_key)
            
            if source_node and target_node:
                self.add_relationship(source_node, target_node, rel_type, rel_weight, rel_metadata)
    
    # ---- 并发控制增强方法 ----
    
    def acquire_read_lock(self, transaction_id, node_key, timeout=None):
        """
        为事务获取节点的读锁
        
        参数:
            transaction_id: 事务ID
            node_key: 节点键
            timeout: 超时时间
            
        返回:
            bool: 是否成功获取锁
        """
        return self.lock_manager.acquire_lock(
            transaction_id, 
            self._get_resource_id(node_key), 
            LockType.SHARED, 
            timeout
        )
    
    def acquire_write_lock(self, transaction_id, node_key, timeout=None):
        """
        为事务获取节点的写锁
        
        参数:
            transaction_id: 事务ID
            node_key: 节点键
            timeout: 超时时间
            
        返回:
            bool: 是否成功获取锁
        """
        return self.lock_manager.acquire_lock(
            transaction_id, 
            self._get_resource_id(node_key), 
            LockType.EXCLUSIVE, 
            timeout
        )
    
    def _get_resource_id(self, node_key):
        """从节点键获取资源ID"""
        return f"node:{node_key}"
    
    def _get_relationship_resource_id(self, source_key, target_key):
        """从关系获取资源ID"""
        return f"rel:{source_key}:{target_key}"
    
    def set_concurrency_mode(self, mode):
        """
        设置并发控制模式
        
        参数:
            mode: 'optimistic' 或 'pessimistic'
            
        返回:
            self
        """
        if mode not in ('optimistic', 'pessimistic'):
            raise ValueError("并发控制模式必须是 'optimistic' 或 'pessimistic'")
            
        self.concurrency_mode = mode
        return self
    
    def set_default_isolation_level(self, level):
        """
        设置默认事务隔离级别
        
        参数:
            level: IsolationLevel枚举值
            
        返回:
            self
        """
        if not isinstance(level, IsolationLevel):
            raise ValueError("隔离级别必须是IsolationLevel枚举")
            
        self.default_isolation_level = level
        return self
    
    # ---- 创建具有事务支持的节点操作 ----
    
    def create_node_in_transaction(self, txn_id, key, val, metadata=None, tags=None, weight=1.0):
        """
        在事务中创建节点
        
        参数:
            txn_id: 事务ID
             key: 节点键
             val: 节点值
            metadata: 元数据
            tags: 标签
            weight: 权重
            
        返回:
            创建的节点
        """
        transaction = self.get_transaction(txn_id)
        if not transaction:
            raise TransactionError(f"事务不存在: {txn_id}")
            
        # 在悲观并发控制模式下获取锁
        if self.concurrency_mode == 'pessimistic':
            # 为节点键获取排他锁
            self.acquire_write_lock(txn_id, key, transaction.timeout)
            
                    # 创建节点
        node = self.create_node(key, val, metadata, tags, weight)
        
        # 记录日志
        transaction.add_log_entry(
            'create', 
            node_key=key, 
            new_value=val, 
            metadata=metadata, 
            tags=tags if tags else None
        )
        
        # 更新版本号
        with self.lock:
            self.version_counter += 1
            self.node_versions[key] = self.version_counter
            
        return node
    
    def update_node_in_transaction(self, txn_id, node, new_val=None, metadata=None, tags=None):
        """
        在事务中更新节点
        
        参数:
            txn_id: 事务ID
            node: 节点或节点键
            new_val: 新值，None表示不更新
            metadata: 要更新的元数据，None表示不更新
            tags: 要设置的标签，None表示不更新
            
        返回:
            更新后的节点
        """
        transaction = self.get_transaction(txn_id)
        if not transaction:
            raise TransactionError(f"事务不存在: {txn_id}")
            
        # 获取节点对象
        node_obj = self._get_node_from_input(node)
        if not node_obj:
            raise ValueError(f"找不到节点: {node}")
            
        node_key = node_obj.key
        
        # 在悲观并发控制模式下获取锁
        if self.concurrency_mode == 'pessimistic':
            # 为节点键获取排他锁
            self.acquire_write_lock(txn_id, node_key, transaction.timeout)
            
        # 保存旧值用于日志
        old_val = node_obj.val
        old_metadata = node_obj.metadata.copy() if node_obj.metadata else {}
        old_tags = set(node_obj.tags) if node_obj.tags else set()
        
        # 更新值
        if new_val is not None:
            self.update_node_value(node_obj, new_val)
            
        # 更新元数据
        if metadata is not None:
            self.update_node_metadata(node_obj, metadata)
            
        # 更新标签
        if tags is not None:
            self.update_node_tag(node_obj, 'set', tags, clear_existing=True)
            
        # 记录日志
        entry = transaction.add_log_entry(
            'update', 
            node_key=node_key, 
            old_value=old_val, 
            new_value=new_val if new_val is not None else node_obj.val, 
            metadata={'old': old_metadata, 'new': metadata} if metadata is not None else None,
            tags={'old': old_tags, 'new': tags} if tags is not None else None
        )
        
        # 为乐观并发控制记录版本
        if self.concurrency_mode == 'optimistic':
            with self.lock:
                self.version_counter += 1
                current_version = self.version_counter
                self.node_versions[node_key] = current_version
                entry.version = current_version
                
        return node_obj
    
    def remove_node_in_transaction(self, txn_id, node):
        """
        在事务中删除节点
        
        参数:
            txn_id: 事务ID
            node: 节点或节点键
            
        返回:
            被删除的节点，如果未找到则返回None
        """
        transaction = self.get_transaction(txn_id)
        if not transaction:
            raise TransactionError(f"事务不存在: {txn_id}")
            
        # 获取节点对象
        node_obj = self._get_node_from_input(node)
        if not node_obj:
            return None
            
        node_key = node_obj.key
        
        # 在悲观并发控制模式下获取锁
        if self.concurrency_mode == 'pessimistic':
            # 为节点键获取排他锁
            self.acquire_write_lock(txn_id, node_key, transaction.timeout)
            
        # 保存旧值用于日志
        old_val = node_obj.val
        old_metadata = node_obj.metadata.copy() if node_obj.metadata else {}
        old_tags = set(node_obj.tags) if node_obj.tags else set()
        old_relationships = node_obj.r.copy() if node_obj.r else []
        
        # 删除节点
        removed_node = self.remove_node_by_key(node_key)
        
        if removed_node:
            # 记录日志
            transaction.add_log_entry(
                'delete', 
                node_key=node_key, 
                old_value=old_val, 
                metadata=old_metadata, 
                tags=old_tags, 
                relationships=old_relationships
            )
            
            # 更新版本
            with self.lock:
                if node_key in self.node_versions:
                    del self.node_versions[node_key]
                    
        return removed_node
    
    def add_relationship_in_transaction(self, txn_id, source, target, rel_type=None, 
                                       rel_weight=1.0, metadata=None):
        """
        在事务中添加关系
        
        参数:
            txn_id: 事务ID
            source: 源节点或节点键
            target: 目标节点或节点键
            rel_type: 关系类型
            rel_weight: 关系权重
            metadata: 关系元数据
            
        返回:
            bool: 是否成功添加关系
        """
        transaction = self.get_transaction(txn_id)
        if not transaction:
            raise TransactionError(f"事务不存在: {txn_id}")
            
        # 获取节点对象
        source_node = self._get_node_from_input(source)
        target_node = self._get_node_from_input(target)
        
        if not source_node or not target_node:
            return False
            
        source_key = source_node.key
        target_key = target_node.key
        
        # 在悲观并发控制模式下获取锁
        if self.concurrency_mode == 'pessimistic':
            # 为源节点和目标节点获取意向排他锁
            self.lock_manager.acquire_lock(
                txn_id, 
                self._get_resource_id(source_key), 
                LockType.INTENT_EXCLUSIVE, 
                transaction.timeout
            )
            self.lock_manager.acquire_lock(
                txn_id, 
                self._get_resource_id(target_key), 
                LockType.INTENT_EXCLUSIVE, 
                transaction.timeout
            )
            # 为关系获取排他锁
            self.lock_manager.acquire_lock(
                txn_id, 
                self._get_relationship_resource_id(source_key, target_key), 
                LockType.EXCLUSIVE, 
                transaction.timeout
            )
            
        # 添加关系
        result = self.add_relationship(source_node, target_node, rel_type, rel_weight, metadata)
        
        if result:
            # 记录日志
            relationship_info = {
                'source': source_key,
                'target': target_key,
                'type': rel_type,
                'weight': rel_weight,
                'metadata': metadata
            }
            
            transaction.add_log_entry(
                'add_relationship', 
                relationships=relationship_info
            )
            
            # 更新版本
            with self.lock:
                self.version_counter += 1
                rel_version = self.version_counter
                rel_id = self._get_relationship_resource_id(source_key, target_key)
                self.node_versions[rel_id] = rel_version
                
        return result
        
    def remove_relationship_in_transaction(self, txn_id, source, target, rel_type=None):
        """
        在事务中删除关系
        
        参数:
            txn_id: 事务ID
            source: 源节点或节点键
            target: 目标节点或节点键
            rel_type: 关系类型
            
        返回:
            bool: 是否成功删除关系
        """
        transaction = self.get_transaction(txn_id)
        if not transaction:
            raise TransactionError(f"事务不存在: {txn_id}")
            
        # 获取节点对象
        source_node = self._get_node_from_input(source)
        target_node = self._get_node_from_input(target)
        
        if not source_node or not target_node:
            return False
            
        source_key = source_node.key
        target_key = target_node.key
        
        # 在悲观并发控制模式下获取锁
        if self.concurrency_mode == 'pessimistic':
            # 为源节点和目标节点获取意向排他锁
            self.lock_manager.acquire_lock(
                txn_id, 
                self._get_resource_id(source_key), 
                LockType.INTENT_EXCLUSIVE, 
                transaction.timeout
            )
            # 为关系获取排他锁
            self.lock_manager.acquire_lock(
                txn_id, 
                self._get_relationship_resource_id(source_key, target_key), 
                LockType.EXCLUSIVE, 
                transaction.timeout
            )
            
        # 查找关系信息用于日志
        relationship_info = None
        for relation in source_node.r:
            if relation['node'] == target_node and (rel_type is None or relation['type'] == rel_type):
                relationship_info = {
                    'source': source_key,
                    'target': target_key,
                    'type': relation.get('type'),
                    'weight': relation.get('weight', 1.0),
                    'metadata': relation.get('metadata')
                }
                break
                
        # 删除关系
        result = self.remove_relationship(source_node, target_node, rel_type)
        
        if result and relationship_info:
            # 记录日志
            transaction.add_log_entry(
                'remove_relationship', 
                relationships=relationship_info
            )
            
            # 更新版本
            with self.lock:
                self.version_counter += 1
                rel_id = self._get_relationship_resource_id(source_key, target_key)
                if rel_id in self.node_versions:
                    del self.node_versions[rel_id]
                    
        return result
    
    # ---- 智能线程池方法 ----
    
    def adjust_worker_count(self, force=False):
        """
        根据使用情况自动调整工作线程数
        
        参数:
            force: 是否强制调整，忽略时间间隔
            
        返回:
            bool: 是否进行了调整
        """
        if not self.adaptive_threading:
            return False
            
        now = time.time()
        if not force and now - self.last_worker_adjustment < self.worker_adjustment_interval:
            return False
            
        self.last_worker_adjustment = now
        
        # 计算最近的线程使用情况
        recent_stats = self.worker_usage_stats[-10:] if self.worker_usage_stats else []
        if not recent_stats:
            return False
            
        avg_usage = sum(s.get('usage_ratio', 0) for s in recent_stats) / len(recent_stats)
        
        # 当前工作线程数
        current_workers = self.executor._max_workers
        
        # 根据使用率调整线程数
        if avg_usage > 0.8 and current_workers < self.max_workers:
            # 使用率高，增加线程
            new_workers = min(current_workers + 2, self.max_workers)
            self._resize_thread_pool(new_workers)
            return True
        elif avg_usage < 0.3 and current_workers > self.min_workers:
            # 使用率低，减少线程
            new_workers = max(current_workers - 1, self.min_workers)
            self._resize_thread_pool(new_workers)
            return True
            
        return False
    
    def _resize_thread_pool(self, new_size):
        """调整线程池大小"""
        try:
            # 关闭当前线程池
            self.executor.shutdown(wait=False)
            
            # 创建新线程池
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=new_size)
            
            print(f"线程池大小已调整: {new_size}")
            return True
        except Exception as e:
            print(f"调整线程池失败: {e}")
            # 如果失败，确保创建新线程池
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.executor._max_workers)
            return False
    
    def update_worker_usage_stats(self):
        """更新工作线程使用统计"""
        if not hasattr(self.executor, '_threads'):
            return
            
        # 线程池中的线程数
        total_threads = self.executor._max_workers
        
        # 活跃线程数（近似值）
        active_threads = len([t for t in self.executor._threads if t.is_alive()])
        
        # 使用比例
        usage_ratio = active_threads / total_threads if total_threads > 0 else 0
        
        # 添加统计
        stat = {
            'timestamp': time.time(),
            'total_threads': total_threads,
            'active_threads': active_threads,
            'usage_ratio': usage_ratio,
            'queue_size': len(self.executor._work_queue.queue) if hasattr(self.executor._work_queue, 'queue') else 0
        }
        
        self.worker_usage_stats.append(stat)
        
        # 限制统计历史大小
        if len(self.worker_usage_stats) > 100:
            self.worker_usage_stats = self.worker_usage_stats[-100:]
        
        # 考虑调整线程池大小
        self.adjust_worker_count()
        
    # ---- 事务集成到现有方法 ----
    
    def get_node_by_key_in_transaction(self, txn_id, key):
        """
        在事务中通过键获取节点
        
        参数:
            txn_id: 事务ID
            key: 节点键
            
        返回:
            节点对象，如果未找到则返回None
        """
        transaction = self.get_transaction(txn_id)
        if not transaction:
            raise TransactionError(f"事务不存在: {txn_id}")
            
        # 在悲观并发控制模式下获取锁
        if self.concurrency_mode == 'pessimistic':
            # 根据隔离级别决定锁类型
            if transaction.isolation_level in (IsolationLevel.REPEATABLE_READ, IsolationLevel.SERIALIZABLE):
                lock_type = LockType.SHARED
            else:
                # READ_UNCOMMITTED和READ_COMMITTED不需要长期持有锁
                lock_type = None
                
            if lock_type:
                self.lock_manager.acquire_lock(
                    txn_id, 
                    self._get_resource_id(key), 
                    lock_type, 
                    transaction.timeout
                )
                
        # 获取节点
        node = self.get_node_by_key(key)
        
        if node:
            # 记录读取操作
            transaction.add_log_entry('read', node_key=key)
            
        return node
        
    def parallel_batch_process_in_transaction(self, txn_id, input_list, process_func):
        """
        在事务中并行处理多个节点或操作
        
        参数:
            txn_id: 事务ID
            input_list: 输入列表
            process_func: 处理函数
            
        返回:
            处理结果列表
        """
        transaction = self.get_transaction(txn_id)
        if not transaction:
            raise TransactionError(f"事务不存在: {txn_id}")
            
        # 包装处理函数以支持事务
        def transaction_wrapper(item):
            return process_func(txn_id, item)
            
        # 更新线程使用统计
        self.update_worker_usage_stats()
        
        # 并行处理
        return self.parallel_batch_process(input_list, transaction_wrapper)
    
    # ---- 并发迭代器方法 ----
    
    def get_concurrent_iterator(self, bucket_indices=None, batch_size=100):
        """
        获取支持并发的节点迭代器
        
        参数:
            bucket_indices: 要迭代的桶索引列表，None表示所有桶
            batch_size: 每批处理的节点数
            
        返回:
            迭代器，生成节点批次
        """
        # 如果未指定桶索引，使用所有桶
        if bucket_indices is None:
            bucket_indices = range(self.size)
            
        # 保证桶索引在有效范围内
        bucket_indices = [i for i in bucket_indices if 0 <= i < self.size]
        
        # 分批迭代所有桶
        for i in bucket_indices:
            nodes_batch = []
            
            # 获取桶锁
            with self.bucket_locks[i]:
                # 复制节点引用以避免在迭代过程中修改列表
                nodes = list(self.buckets[i])
                
            # 分批处理节点
            for node in nodes:
                nodes_batch.append(node)
                
                if len(nodes_batch) >= batch_size:
                    yield nodes_batch
                    nodes_batch = []
                    
            # 处理最后一批
            if nodes_batch:
                yield nodes_batch
                
    def process_nodes_concurrently(self, processor_func, bucket_indices=None, 
                                   batch_size=100, max_workers=None):
        """
        并发处理所有节点
        
        参数:
            processor_func: 节点处理函数 func(node) -> result
            bucket_indices: 要处理的桶索引列表，None表示所有桶
            batch_size: 每批处理的节点数
            max_workers: 最大工作线程数，None表示使用执行器的默认值
            
        返回:
            处理结果列表
        """
        # 获取节点迭代器
        node_iterator = self.get_concurrent_iterator(bucket_indices, batch_size)
        
        results = []
        futures = []
        
        # 如果未指定max_workers，使用执行器的默认值
        actual_max_workers = max_workers or self.executor._max_workers
        
        # 创建临时执行器或使用现有执行器
        if max_workers:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        else:
            executor = self.executor
            
        try:
            # 更新线程使用统计
            self.update_worker_usage_stats()
            
            # 提交批处理任务
            for nodes_batch in node_iterator:
                future = executor.submit(self._process_nodes_batch, nodes_batch, processor_func)
                futures.append(future)
                
            # 收集结果
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    print(f"处理节点批次时出错: {e}")
                    
        finally:
            # 如果创建了临时执行器，关闭它
            if max_workers:
                executor.shutdown()
                    
            return results
        
    def _process_nodes_batch(self, nodes_batch, processor_func):
        """处理节点批次"""
        results = []
        for node in nodes_batch:
            try:
                result = processor_func(node)
                results.append(result)
            except Exception as e:
                print(f"处理节点 {node.key} 时出错: {e}")
                results.append(None)
        return results
    
    # ---- 智能事务方法 ----
    
    def batch_transaction(self, operations, isolation_level=None, timeout=None, retry_count=3):
        """
        执行批量操作的智能事务
        
        参数:
            operations: 操作列表，每个操作是一个字典，包含操作类型和参数
            isolation_level: 事务隔离级别
            timeout: 事务超时时间
            retry_count: 冲突时的重试次数
            
        返回:
            字典，包含操作结果和状态
        """
        results = []
        attempts = 0
        success = False
        
        while attempts < retry_count and not success:
            attempts += 1
            
            try:
                # 开始事务
                txn = self.begin_transaction(isolation_level, timeout)
                txn_id = txn.id
                
                # 执行操作
                op_results = []
                for op in operations:
                    op_type = op.get('type')
                    op_params = op.get('params', {})
                    
                    if op_type == 'create_node':
                        result = self.create_node_in_transaction(
                            txn_id,
                            op_params.get('key'),
                            op_params.get('val'),
                            op_params.get('metadata'),
                            op_params.get('tags'),
                            op_params.get('weight', 1.0)
                        )
                    elif op_type == 'update_node':
                        result = self.update_node_in_transaction(
                            txn_id,
                            op_params.get('node'),
                            op_params.get('new_val'),
                            op_params.get('metadata'),
                            op_params.get('tags')
                        )
                    elif op_type == 'remove_node':
                        result = self.remove_node_in_transaction(
                            txn_id,
                            op_params.get('node')
                        )
                    elif op_type == 'add_relationship':
                        result = self.add_relationship_in_transaction(
                            txn_id,
                            op_params.get('source'),
                            op_params.get('target'),
                            op_params.get('rel_type'),
                            op_params.get('rel_weight', 1.0),
                            op_params.get('metadata')
                        )
                    elif op_type == 'remove_relationship':
                        result = self.remove_relationship_in_transaction(
                            txn_id,
                            op_params.get('source'),
                            op_params.get('target'),
                            op_params.get('rel_type')
                        )
                    elif op_type == 'get_node':
                        result = self.get_node_by_key_in_transaction(
                            txn_id,
                            op_params.get('key')
                        )
                    else:
                        result = {"error": f"未知操作类型: {op_type}"}
                        
                    op_results.append({
                        "type": op_type,
                        "result": result
                    })
                    
                # 提交事务
                self.commit_transaction(txn_id)
                success = True
                
                results = {
                    "success": True,
                    "operations": op_results,
                    "attempts": attempts
                }
                
            except (ConcurrencyError, DeadlockError, TimeoutError) as e:
                # 如果发生并发冲突，尝试重试
                if attempts < retry_count:
                    print(f"事务冲突，正在重试 ({attempts}/{retry_count}): {e}")
                    # 短暂延迟后重试
                    time.sleep(0.1 * attempts)
                else:
                    # 重试次数用尽，返回错误
                    results = {
                        "success": False,
                        "error": str(e),
                        "attempts": attempts
                    }
            except Exception as e:
                # 其他错误，直接返回
                results = {
                    "success": False,
                    "error": str(e),
                    "attempts": attempts
                }
                break
                
        return results
    
    def smart_update(self, selector, updates, upsert=False, multi=False):
        """
        智能更新操作，类似MongoDB的updateOne/updateMany
        
        参数:
            selector: 节点选择条件，支持标签、元数据等
            updates: 要应用的更新，支持值、元数据、标签等
            upsert: 如果未找到匹配节点，是否创建新节点
            multi: 是否更新所有匹配的节点
            
        返回:
            更新结果
        """
        # 开始事务
        txn = self.begin_transaction()
        txn_id = txn.id
        
        try:
            # 查找匹配的节点
            matching_nodes = self._find_nodes_by_selector(selector)
            
            if not matching_nodes and upsert:
                # 未找到匹配节点且upsert为True，创建新节点
                key = selector.get('key') or str(uuid.uuid4())
                val = updates.get('val') or selector.get('val') or None
                metadata = updates.get('metadata') or selector.get('metadata') or {}
                tags = updates.get('tags') or selector.get('tags') or set()
                weight = updates.get('weight') or selector.get('weight') or 1.0
                
                node = self.create_node_in_transaction(txn_id, key, val, metadata, tags, weight)
                updated_nodes = [node]
                
            elif matching_nodes:
                # 找到匹配节点，应用更新
                updated_nodes = []
                
                # 如果multi为False，只更新第一个匹配的节点
                nodes_to_update = matching_nodes if multi else [matching_nodes[0]]
                
                for node in nodes_to_update:
                    updated = self.update_node_in_transaction(
                        txn_id,
                        node,
                        updates.get('val'),
                        updates.get('metadata'),
                        updates.get('tags')
                    )
                    
                    # 更新权重（如果提供）
                    if 'weight' in updates:
                        node.weight = float(updates['weight'])
                        
                    updated_nodes.append(updated)
            else:
                # 未找到匹配节点且upsert为False
                updated_nodes = []
                
            # 提交事务
            self.commit_transaction(txn_id)
            
            return {
                "success": True,
                "matched_count": len(matching_nodes),
                "modified_count": len(updated_nodes),
                "upserted": not matching_nodes and upsert,
                "updated_nodes": updated_nodes
            }
            
        except Exception as e:
            # 发生错误，中止事务
            self.abort_transaction(txn_id, str(e))
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def _find_nodes_by_selector(self, selector):
        """根据选择器查找节点"""
        matching_nodes = []
        
        # 如果提供了键，直接查找
        if 'key' in selector:
            node = self.get_node_by_key(selector['key'])
            if node:
                matching_nodes.append(node)
                return matching_nodes
        
        # 使用元数据条件
        if 'metadata' in selector and isinstance(selector['metadata'], dict):
            meta_matches = []
            for meta_key, meta_val in selector['metadata'].items():
                nodes = self.find_by_metadata(meta_key, meta_val)
                meta_matches.extend(nodes)
                
            if meta_matches:
                matching_nodes.extend(meta_matches)
        
        # 使用标签条件
        if 'tags' in selector:
            tags = selector['tags']
            if isinstance(tags, (list, set)):
                for tag in tags:
                    tag_nodes = self.find_by_tag(tag)
                    matching_nodes.extend(tag_nodes)
            else:
                # 单个标签
                tag_nodes = self.find_by_tag(tags)
                matching_nodes.extend(tag_nodes)
        
        # 去重
        seen = set()
        unique_nodes = []
        for node in matching_nodes:
            if id(node) not in seen:
                seen.add(id(node))
                unique_nodes.append(node)
                
        return unique_nodes
    
    # ---- 自动事务管理方法 ----
    
    def auto_transaction(self, func):
        """
        自动事务装饰器，将函数包装在事务中
        
        用法:
            @ion.auto_transaction
            def my_function(ion, param1, param2):
                # 对ION的操作将自动在事务中执行
                pass
                
        参数:
            func: 要包装的函数
            
        返回:
            包装后的函数
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 开始事务
            txn = self.begin_transaction()
            txn_id = txn.id
            
            try:
                # 执行函数，将事务ID添加到参数中
                if 'txn_id' in kwargs:
                    # 如果已经提供了事务ID，使用提供的
                    result = func(*args, **kwargs)
                else:
                    # 否则添加新的事务ID
                    kwargs['txn_id'] = txn_id
                    result = func(*args, **kwargs)
                
                # 提交事务
                self.commit_transaction(txn_id)
                return result
                
            except Exception as e:
                # 发生错误，中止事务
                self.abort_transaction(txn_id, str(e))
                raise
                
        return wrapper
    
    # ---- 安全增强方法 ----
    
    def secure_operation(self, func, permission_check=None, audit_log=True):
        """
        安全操作装饰器，添加权限检查和审计日志
        
        用法:
            @ion.secure_operation(permission_check=check_func)
            def sensitive_operation(ion, param1, param2):
                # 敏感操作
                pass
                
        参数:
            func: 要保护的函数
            permission_check: 权限检查函数 func(operation, params) -> bool
            audit_log: 是否记录审计日志
            
        返回:
            包装后的函数
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 提取操作信息
            operation = func.__name__
            
            # 如果提供了权限检查函数，执行检查
            if permission_check:
                if not permission_check(operation, kwargs):
                    raise PermissionError(f"操作 {operation} 的权限被拒绝")
            
            # 记录审计日志
            if audit_log:
                self._add_audit_log(operation, kwargs)
                
            # 执行操作
            return func(*args, **kwargs)
            
        return wrapper
    
    def _add_audit_log(self, operation, params):
        """添加审计日志"""
        log_entry = {
            'timestamp': time.time(),
            'operation': operation,
            'params': {k: v for k, v in params.items() if k != 'password'}, # 排除敏感信息
            'transaction_id': params.get('txn_id')
        }
        
        # 实际中，这里可能会将日志写入数据库或文件
        print(f"审计日志: {log_entry}")

# 新增的安全异常类
    class PermissionError(Exception):
        """权限错误，表示用户没有执行操作的权限"""
        pass

# 在ION类中添加以下方法

    def astar_search_in_transaction(self, txn_id, start, goal, heuristic_func=None,
                                   weight_func=None, rel_type=None, max_iterations=10000):
        """
        在事务中执行A*搜索算法
        
        参数:
            txn_id: 事务ID
            start: 起始节点或节点键
            goal: 目标节点或节点键
            heuristic_func: 启发函数
            weight_func: 权重函数
            rel_type: 关系类型
            max_iterations: 最大迭代次数
            
        返回:
            (path, cost): 路径和总成本的元组
        """
        transaction = self.get_transaction(txn_id)
        if not transaction:
            raise TransactionError(f"事务不存在: {txn_id}")
            
        # 获取节点对象，并记录读取操作
        start_node = self._get_node_from_input_with_txn(txn_id, start)
        goal_node = self._get_node_from_input_with_txn(txn_id, goal)
        
        if not start_node or not goal_node:
            return None, float('inf')
            
        # 在悲观并发控制模式下获取路径上可能的节点的锁
        if self.concurrency_mode == 'pessimistic':
            # 对起点和终点获取共享锁
            if transaction.isolation_level in (IsolationLevel.REPEATABLE_READ, IsolationLevel.SERIALIZABLE):
                self.acquire_read_lock(txn_id, start_node.key, transaction.timeout)
                self.acquire_read_lock(txn_id, goal_node.key, transaction.timeout)
        
        # 执行A*搜索
        return self.astar_search(start_node, goal_node, heuristic_func, weight_func, rel_type, max_iterations)
    
    def _get_node_from_input_with_txn(self, txn_id, input_data):
        """在事务中从各种输入类型获取节点，并记录读取"""
        node = self._get_node_from_input(input_data)
        if node:
            # 记录读取操作
            transaction = self.get_transaction(txn_id)
            if transaction:
                transaction.add_log_entry('read', node_key=node.key)
        return node
    
    def astar_search_advanced_in_transaction(self, txn_id, start, goal, options=None):
        """
        在事务中执行高级A*搜索
        
        参数:
            txn_id: 事务ID
            start: 起始节点或节点键
            goal: 目标节点或节点键
            options: 搜索选项
            
        返回:
            (path, cost, stats): 路径、成本和统计信息的元组
        """
        transaction = self.get_transaction(txn_id)
        if not transaction:
            raise TransactionError(f"事务不存在: {txn_id}")
            
        # 获取节点对象，并记录读取操作
        start_node = self._get_node_from_input_with_txn(txn_id, start)
        
        # 处理多目标情况
        if options and options.get('multi_goal') and isinstance(goal, (list, tuple, set)):
            goal_nodes = [self._get_node_from_input_with_txn(txn_id, g) for g in goal]
            goal_nodes = [g for g in goal_nodes if g is not None]
        else:
            goal_node = self._get_node_from_input_with_txn(txn_id, goal)
            goal_nodes = [goal_node] if goal_node else []
            
        if not goal_nodes or not start_node:
            return None, float('inf'), {}
            
        # 在悲观并发控制模式下获取锁
        if self.concurrency_mode == 'pessimistic':
            # 对起点和终点获取共享锁
            if transaction.isolation_level in (IsolationLevel.REPEATABLE_READ, IsolationLevel.SERIALIZABLE):
                self.acquire_read_lock(txn_id, start_node.key, transaction.timeout)
                for goal_node in goal_nodes:
                    self.acquire_read_lock(txn_id, goal_node.key, transaction.timeout)
        
        # 执行高级A*搜索
        return self.astar_search_advanced(start_node, goal_nodes if len(goal_nodes) > 1 else goal_nodes[0], options)

    # ---- 性能优化方法 ----
    
    def optimize_for_path_finding(self, max_cached_paths=1000):
        """
        优化路径查找性能
        
        参数:
            max_cached_paths: 最大缓存路径数
            
        返回:
            self
        """
        # 初始化路径缓存
        self.path_cache = {}
        self.path_cache_max_size = max_cached_paths
        self.path_cache_hits = 0
        self.path_cache_misses = 0
        
        # 创建路径索引
        self._build_path_indices()
        
        return self
    
    def _build_path_indices(self):
        """构建路径索引以加速路径查找"""
        # 初始化路径索引结构
        self.path_indices = {
            'hubs': set(),  # 高连接度节点（枢纽）
            'shortcuts': {},  # 节点对之间的快捷路径
            'communities': []  # 社区或集群
        }
        
        # 识别高连接度节点（枢纽）
        hub_nodes = self.find_most_connected(top_n=int(self.count * 0.05) or 10)  # 取前5%的高连接度节点
        for node in hub_nodes:
            if len(node.r) > 5:  # 至少有5个连接才算枢纽
                self.path_indices['hubs'].add(node)
        
        # 为高连接度节点之间预计算路径
        for hub1 in self.path_indices['hubs']:
            for hub2 in self.path_indices['hubs']:
                if hub1 != hub2:
                    path, cost = self.astar_search(hub1, hub2)
                    if path:
                        key = (hub1.key, hub2.key)
                        self.path_indices['shortcuts'][key] = (path, cost)
        
        # 识别社区结构
        if self.start:
            self.path_indices['communities'] = [self.get_community(start) for start in self.start]
        
        print(f"路径索引已构建: {len(self.path_indices['hubs'])} 个枢纽, "
              f"{len(self.path_indices['shortcuts'])} 个快捷路径, "
              f"{len(self.path_indices['communities'])} 个社区")
    
    def cached_astar_search(self, start, goal, heuristic_func=None, weight_func=None, rel_type=None, 
                           max_iterations=10000, bypass_cache=False):
        """
        带缓存的A*搜索，利用预计算的路径和缓存加速搜索
        
        参数:
            start: 起始节点或键
            goal: 目标节点或键
            heuristic_func, weight_func, rel_type, max_iterations: 与astar_search相同
            bypass_cache: 是否绕过缓存直接搜索
            
        返回:
            (path, cost): 路径和成本的元组
        """
        # 检查是否启用了缓存
        if not hasattr(self, 'path_cache'):
            return self.astar_search(start, goal, heuristic_func, weight_func, rel_type, max_iterations)
        
        # 获取节点对象
        start_node = self._get_node_from_input(start)
        goal_node = self._get_node_from_input(goal)
        
        if not start_node or not goal_node:
            return None, float('inf')
        
        # 生成缓存键
        cache_key = (start_node.key, goal_node.key, rel_type)
        
        # 如果不绕过缓存，先检查缓存
        if not bypass_cache and cache_key in self.path_cache:
            self.path_cache_hits += 1
            return self.path_cache[cache_key]
        
        # 检查是否可以使用快捷路径
        if not bypass_cache:
            # 如果起点和终点都是枢纽，直接使用预计算的路径
            if start_node in self.path_indices['hubs'] and goal_node in self.path_indices['hubs']:
                shortcut_key = (start_node.key, goal_node.key)
                if shortcut_key in self.path_indices['shortcuts']:
                    self.path_cache_hits += 1
                    return self.path_indices['shortcuts'][shortcut_key]
            
            # 如果起点和终点分别连接到不同的枢纽，尝试通过枢纽路径
            start_hubs = [n['node'] for n in start_node.r if n['node'] in self.path_indices['hubs']]
            goal_hubs = [n['node'] for n in goal_node.r if n['node'] in self.path_indices['hubs']]
            
            if start_hubs and goal_hubs:
                best_path = None
                best_cost = float('inf')
                
                for start_hub in start_hubs:
                    for goal_hub in goal_hubs:
                        # 检查枢纽间是否有快捷路径
                        hub_key = (start_hub.key, goal_hub.key)
                        if hub_key in self.path_indices['shortcuts']:
                            hub_path, hub_cost = self.path_indices['shortcuts'][hub_key]
                            
                            # 计算完整路径成本
                            start_to_hub_cost = self._calculate_edge_cost(start_node, start_hub, weight_func)
                            hub_to_goal_cost = self._calculate_edge_cost(goal_hub, goal_node, weight_func)
                            total_cost = start_to_hub_cost + hub_cost + hub_to_goal_cost
                            
                            if total_cost < best_cost:
                                # 构建完整路径
                                full_path = [start_node] + hub_path[1:-1] + [goal_node]
                                best_path = full_path
                                best_cost = total_cost
                
                if best_path:
                    # 缓存并返回结果
                    self.path_cache[cache_key] = (best_path, best_cost)
                    # 维护缓存大小
                    if len(self.path_cache) > self.path_cache_max_size:
                        self._prune_path_cache()
                    return best_path, best_cost
        
        # 如果没有找到缓存或快捷路径，执行常规A*搜索
        self.path_cache_misses += 1
        path, cost = self.astar_search(start_node, goal_node, heuristic_func, weight_func, rel_type, max_iterations)
        
        # 缓存结果（如果找到路径）
        if path:
            self.path_cache[cache_key] = (path, cost)
            # 维护缓存大小
            if len(self.path_cache) > self.path_cache_max_size:
                self._prune_path_cache()
        
        return path, cost
    
    def _calculate_edge_cost(self, node1, node2, weight_func=None):
        """计算两个相邻节点之间的边成本"""
        # 默认权重函数
        if weight_func is None:
            def default_weight_func(current, next_node, relation):
                return relation.get('weight', 1.0)
            weight_func = default_weight_func
        
        # 查找从node1到node2的关系
        for relation in node1.r:
            if relation['node'] == node2:
                return weight_func(node1, node2, relation)
        
        # 如果没有直接关系，返回默认成本
        return 1.0
    
    def _prune_path_cache(self):
        """维护路径缓存大小"""
        # 简单策略：移除最旧的20%的条目
        num_to_remove = max(1, int(len(self.path_cache) * 0.2))
        keys_to_remove = list(self.path_cache.keys())[:num_to_remove]
        
        for key in keys_to_remove:
            del self.path_cache[key]
    
    def get_path_finding_stats(self):
        """获取路径查找统计信息"""
        if not hasattr(self, 'path_cache'):
            return {
                'cache_hits': 0,
                'cache_misses': 0,
                'total_queries': 0,
                'hit_ratio': 0,
                'cache_size': 0
            }
                
        total = self.path_cache_hits + self.path_cache_misses
        hit_ratio = self.path_cache_hits / total if total > 0 else 0
        
        return {
            'cache_hits': self.path_cache_hits,
            'cache_misses': self.path_cache_misses,
            'total_queries': total,
            'hit_ratio': hit_ratio,
            'cache_size': len(self.path_cache) if hasattr(self, 'path_cache') else 0
        }

    def invalidate_path_cache(self, node_keys=None):
        """
        使路径缓存失效
        
        参数:
            node_keys: 要使相关缓存失效的节点键列表，None表示清空所有缓存
            
        返回:
            移除的缓存条目数
        """
        if not hasattr(self, 'path_cache'):
            return 0
        
        if node_keys is None:
            # 清空整个缓存
            count = len(self.path_cache)
            self.path_cache.clear()
            return count
        else:
            # 只清除涉及指定节点的缓存
            keys_to_remove = []
            for cache_key in self.path_cache:
                start_key, end_key, _ = cache_key
                if start_key in node_keys or end_key in node_keys:
                    keys_to_remove.append(cache_key)
            
            for key in keys_to_remove:
                del self.path_cache[key]
                
            return len(keys_to_remove)
    
    @staticmethod
    def performance_monitor(func):
        """
        性能监控装饰器，记录函数执行时间
        
        使用方法:
        @ION.performance_monitor
        def my_function(args):
            # 函数体
            pass
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # 记录执行时间
            func_name = func.__name__
            print(f"性能统计: {func_name} 执行时间 {execution_time:.6f} 秒")
            
            # 保存性能数据到静态字典中
            if not hasattr(ION.performance_monitor, 'stats'):
                ION.performance_monitor.stats = {}
            
            if func_name not in ION.performance_monitor.stats:
                ION.performance_monitor.stats[func_name] = {
                    'calls': 0,
                    'total_time': 0,
                    'min_time': float('inf'),
                    'max_time': 0,
                    'avg_time': 0
                }
                
            stats = ION.performance_monitor.stats[func_name]
            stats['calls'] += 1
            stats['total_time'] += execution_time
            stats['min_time'] = min(stats['min_time'], execution_time)
            stats['max_time'] = max(stats['max_time'], execution_time)
            stats['avg_time'] = stats['total_time'] / stats['calls']
            
            return result
        
        return wrapper
    
    @staticmethod
    def get_performance_stats():
        """获取性能统计数据"""
        if not hasattr(ION.performance_monitor, 'stats'):
            return {}
        return ION.performance_monitor.stats
    
    # 添加复合索引方法
    def create_compound_index(self, index_types):
        """
        创建复合索引，用于优化多条件查询
        
        参数:
            index_types: 索引类型元组，如('tag', 'metadata')或('metadata', 'value_type')
            
        返回:
            bool: 是否成功创建索引
        """
        if not isinstance(index_types, tuple) or len(index_types) < 2:
            raise ValueError("索引类型必须是至少包含两个元素的元组")
            
        # 检查索引类型是否有效
        valid_types = {'tag', 'metadata', 'value_type', 'weight'}
        for idx_type in index_types:
            if idx_type not in valid_types:
                raise ValueError(f"无效的索引类型: {idx_type}，有效类型为: {valid_types}")
                
        # 如果已经创建了该复合索引，直接返回
        if index_types in self.compound_index_types:
            return True
            
        # 初始化复合索引
        self.compound_indices[index_types] = {}
        self.compound_index_types.add(index_types)
        
        # 为所有现有节点创建复合索引
        self._build_compound_index(index_types)
        
        return True
    
    def _build_compound_index(self, index_types):
        """为所有现有节点构建指定的复合索引"""
        for bucket in self.buckets:
                for node in bucket:
                    self._index_node_to_compound(node, index_types)
    def _async_indexing_worker(self):
        """异步索引更新工作线程，处理索引队列中的操作"""
        while self.async_indexing:
            try:
                # 如果队列为空，等待一小段时间
                if not self.indexing_queue:
                    import time
                    time.sleep(0.01)
                    continue
                
                # 获取下一个索引操作
                try:
                    operation = self.indexing_queue.popleft()
                except IndexError:
                    # 队列可能在检查后变为空
                    continue
                
                # 处理操作
                op_type = operation.get('type')
                node = operation.get('node')
                data = operation.get('data')
                
                if op_type == 'add':
                    # 添加节点到索引
                    if node:
                    # 添加到双向映射
                        self.bimap.add(node.key, node.val)
                        
                        # 添加到组合数据结构
                        self._combined_data.put(node.key, node.val, metadata=node.metadata)
                    
                    # 索引元数据
                    if node.metadata:
                        for m_key, m_val in node.metadata.items():
                                self._index_metadata(m_key, m_val, node)
                    
                    # 索引标签
                    if node.tags:
                        for tag in node.tags:
                                self._index_tag(tag, node)
                         
                 # 索引值类型
                        self._index_value_type(node.val, node)
                        
                        # 添加到复合索引
                        for index_types in self.compound_index_types:
                            self._index_node_to_compound(node, index_types)
                
                elif op_type == 'remove':
                    # 从索引中移除节点
                    if node:
                        self._cleanup_node_indices(node)
                        
                        # 从复合索引中移除
                        self._remove_node_from_compound_indices(node)
                        
                        # 从组合数据结构中删除
                        self._combined_data.delete(node.key, keep_relationships=False)
                
                elif op_type == 'update':
                    # 更新索引
                    if node and data:
                        field = data.get('field')
                        old_value = data.get('old_value')
                        new_value = data.get('new_value')
                        
                        if field == 'metadata':
                            # 从复合索引中移除节点
                            self._remove_node_from_compound_indices(node)
                            
                            # 更新元数据索引
                            if old_value:
                                for meta_key, meta_val in old_value.items():
                                    if meta_key not in new_value or new_value[meta_key] != meta_val:
                                        self._remove_metadata_index(meta_key, meta_val, node)
                            
                            # 添加新索引
                            if new_value:
                                for meta_key, meta_val in new_value.items():
                                    if meta_key not in old_value or old_value[meta_key] != meta_val:
                                        self._index_metadata(meta_key, meta_val, node)
                            
                            # 重新添加到复合索引
                            for index_types in self.compound_index_types:
                                self._index_node_to_compound(node, index_types)
                        
                        elif field == 'tag':
                            # 从复合索引中移除节点
                            self._remove_node_from_compound_indices(node)
                            
                            # 更新标签索引
                            added_tags = set(new_value or []) - set(old_value or [])
                            removed_tags = set(old_value or []) - set(new_value or [])
                            
                            for tag in removed_tags:
                                self._remove_tag_index(tag, node)
                            
                            for tag in added_tags:
                                self._index_tag(tag, node)
                            
                            # 重新添加到复合索引
                            for index_types in self.compound_index_types:
                                self._index_node_to_compound(node, index_types)
                        
                        elif field == 'value':
                            # 从复合索引中移除节点
                            self._remove_node_from_compound_indices(node)
                            
                            # 更新值和值类型索引
                            old_type = type(old_value).__name__ if old_value is not None else None
                            new_type = type(new_value).__name__ if new_value is not None else None
                            
                            # 更新双向映射
                            self.bimap.remove(node.key)
                            self.bimap.add(node.key, new_value)
                            
                            # 更新值类型索引
                            if old_type != new_type:
                                if old_type and old_type in self.value_type_index and node in self.value_type_index[old_type]:
                                    self.value_type_index[old_type].remove(node)
                                    if not self.value_type_index[old_type]:
                                        del self.value_type_index[old_type]
                                
                                self._index_value_type(new_value, node)
                            
                            # 更新组合数据结构
                            self._combined_data.put(node.key, new_value, metadata=node.metadata)
                            
                            # 重新添加到复合索引
                            for index_types in self.compound_index_types:
                                self._index_node_to_compound(node, index_types)
                        
                        elif field == 'weight':
                            # 从复合索引中移除节点
                            self._remove_node_from_compound_indices(node)
                            
                            # 重新添加到复合索引
                            for index_types in self.compound_index_types:
                                self._index_node_to_compound(node, index_types)
                
            except Exception as e:
                import logging
                logging.error(f"异步索引更新错误: {e}")
                # 继续处理下一个操作，不要因为一个错误就退出循环
    def _index_node_to_compound(self, node, index_types):
        """
        将节点添加到复合索引
        
        参数:
            node: 要索引的节点
            index_types: 索引类型元组
        """
        # 获取各个索引类型的值
        index_values = []
        
        for idx_type in index_types:
            if idx_type == 'tag':
                # 对于标签，每个标签生成一个值
                values = list(node.tags) if node.tags else [None]
            elif idx_type == 'metadata':
                # 对于元数据，每个键值对生成一个值
                if node.metadata:
                    values = [f"{key}:{value}" for key, value in node.metadata.items()]
                else:
                    values = [None]
            elif idx_type == 'value_type':
                # 对于值类型，使用值的类型名称
                values = [type(node.val).__name__]
            elif idx_type == 'weight':
                # 对于权重，使用权重范围(向下取整到最接近的整数)
                weight_bucket = int(node.weight)
                values = [weight_bucket]
            else:
                values = [None]
                
            index_values.append(values)
        
        # 生成所有可能的复合键组合
        import itertools
        for combo in itertools.product(*index_values):
            if None not in combo:  # 忽略包含None的组合
                compound_key = tuple(combo)
                if compound_key not in self.compound_indices[index_types]:
                    self.compound_indices[index_types][compound_key] = []
                if node not in self.compound_indices[index_types][compound_key]:
                    self.compound_indices[index_types][compound_key].append(node)
    
    def _remove_node_from_compound_indices(self, node):
        """从所有复合索引中删除节点"""
        for index_types, index_dict in self.compound_indices.items():
            for compound_key, nodes in list(index_dict.items()):
                if node in nodes:
                    nodes.remove(node)
                    # 如果节点列表为空，删除该复合键
                    if not nodes:
                        del index_dict[compound_key]
    
    def compound_search(self, **conditions):
        """
        使用复合索引进行多条件查询
        
        参数:
            **conditions: 查询条件，如tag='active', metadata=('status', 'premium')
            
        返回:
            匹配的节点列表
        """
        # 确定要使用的索引类型
        query_types = tuple(sorted(conditions.keys()))
        
        # 检查是否有匹配的复合索引
        if query_types not in self.compound_index_types:
            # 如果没有匹配的复合索引，尝试创建
            try:
                self.create_compound_index(query_types)
            except Exception as e:
                # 如果无法创建复合索引，回退到基本查询
                print(f"无法使用复合索引: {e}，回退到基本查询")
                return self._fallback_compound_search(**conditions)
        
        # 构建查询的复合键
        query_values = []
        for idx_type in query_types:
            value = conditions.get(idx_type)
            
            if idx_type == 'tag':
                query_values.append(value)
            elif idx_type == 'metadata':
                if isinstance(value, tuple) and len(value) == 2:
                    query_values.append(f"{value[0]}:{value[1]}")
                else:
                    # 如果元数据格式不正确，回退到基本查询
                    return self._fallback_compound_search(**conditions)
            elif idx_type == 'value_type':
                query_values.append(value)
            elif idx_type == 'weight':
                # 对于权重，使用整数范围
                if isinstance(value, (int, float)):
                    query_values.append(int(value))
                else:
                    return self._fallback_compound_search(**conditions)
        
        # 查询复合索引
        compound_key = tuple(query_values)
        return self.compound_indices[query_types].get(compound_key, [])
    
    def _fallback_compound_search(self, **conditions):
        """当复合索引不可用时的回退查询方法"""
        # 从各个单独的索引获取结果并取交集
        result_sets = []
        
        for idx_type, value in conditions.items():
            if idx_type == 'tag':
                nodes = self.find_by_tag(value)
            elif idx_type == 'metadata':
                if isinstance(value, tuple) and len(value) == 2:
                    nodes = self.find_by_metadata(value[0], value[1])
                else:
                    nodes = []
            elif idx_type == 'value_type':
                nodes = self.find_by_value_type(value)
            elif idx_type == 'weight':
                if isinstance(value, (int, float)):
                    # 获取权重在整数范围内的节点
                    weight_floor = int(value)
                    weight_ceil = weight_floor + 1
                    nodes = self.get_nodes_by_weight_range(weight_floor, weight_ceil)
                else:
                    nodes = []
            else:
                nodes = []
                
            if nodes:
                result_sets.append(set(nodes))
                
        # 如果没有结果集，返回空列表
        if not result_sets:
            return []
            
        # 取交集
        result = set.intersection(*result_sets)
        return list(result)
        
    # 在创建和更新节点方法中添加对复合索引的支持
    
    
    # 批量索引更新方法
    def _schedule_index_update(self, update_type, node, data=None):
        """
        将索引更新操作加入队列，当队列长度达到阈值或定时触发时执行批量更新
        
        参数:
            update_type: 更新类型，如'add', 'remove', 'update'
            node: 相关节点
            data: 额外数据，如元数据、标签等
        """
        # 添加到生物图(bimap)
        if update_type == 'add':
            self.bimap.add(node.key, node.val)
            
            # 添加到组合数据结构
            self._combined_data.put(node.key, node.val, metadata=node.metadata)
            
            # 索引元数据
            if node.metadata:
                for m_key, m_val in node.metadata.items():
                    self._index_metadata(m_key, m_val, node)
            
            # 索引标签
            if node.tags:
                for tag in node.tags:
                    self._index_tag(tag, node)
                    
            # 索引值类型
            self._index_value_type(node.val, node)
            
        elif update_type == 'remove':
            # 清理索引
            self._cleanup_node_indices(node)
            
            # 从组合数据结构中删除
            self._combined_data.delete(node.key, keep_relationships=False)
            
        elif update_type == 'update':
            # 更新相关索引
            if data and 'field' in data and 'old_value' in data and 'new_value' in data:
                field = data['field']
                old_value = data['old_value']
                new_value = data['new_value']
                
                if field == 'metadata':
                    # 更新元数据索引
                    old_metadata, new_metadata = old_value, new_value
                    
                    # 清除旧元数据的索引
                    if old_metadata:
                        for meta_key, meta_val in old_metadata.items():
                            self._remove_metadata_index(meta_key, meta_val, node)
                    
                    # 添加新元数据的索引
                    if new_metadata:
                        for meta_key, meta_val in new_metadata.items():
                            self._index_metadata(meta_key, meta_val, node)
                            
                elif field == 'tag':
                    # 更新标签索引
                    if old_value and old_value in node.tags:
                        self._remove_tag_index(old_value, node)
                    
                    if new_value:
                        self._index_tag(new_value, node)
                        
                elif field == 'value':
                    # 更新值和值类型索引
                    if old_value != new_value:
                        # 更新双向映射
                        self.bimap.remove(node.key)
                        self.bimap.add(node.key, new_value)
                        
                        # 更新值类型索引
                        old_type = type(old_value).__name__
                        new_type = type(new_value).__name__
                        
                        if old_type != new_type:
                            if old_type in self.value_type_index and node in self.value_type_index[old_type]:
                                self.value_type_index[old_type].remove(node)
                                
                            self._index_value_type(new_value, node)
                        
                        # 更新组合数据结构
                        self._combined_data.put(node.key, new_value, metadata=node.metadata)
                        
    def _process_index_updates(self):
        """处理索引更新队列中的操作"""
        # 这个方法用于批量处理索引更新，但在这里我们暂时不需要实现具体的批处理逻辑
        pass
        
    
    
    # 改进死锁检测和处理
    def optimize_deadlock_detection(self):
        """优化死锁检测策略"""
        # 设置死锁检测间隔
        self.lock_manager.deadlock_detection_interval = 0.5  # 秒
        
        # 启动定期死锁检测
        def deadlock_detector():
            while True:
                try:
                    # 执行死锁检测
                    deadlocks = self.lock_manager.detect_all_deadlocks()
                    
                    # 处理检测到的死锁
                    for cycle in deadlocks:
                        # 选择要中止的事务（通常是最新的事务）
                        if cycle:
                            victim_txn_id = cycle[-1]
                            print(f"发现死锁，中止事务: {victim_txn_id}")
                            
                            # 找到事务并中止
                            transaction = self.get_transaction(victim_txn_id)
                            if transaction:
                                try:
                                    transaction.abort(reason="检测到死锁，自动中止")
                                except Exception as e:
                                    print(f"中止事务失败: {e}")
                except Exception as e:
                    print(f"死锁检测错误: {e}")
                    
                # 等待下一次检测
                time.sleep(self.lock_manager.deadlock_detection_interval)
        
        # 创建并启动死锁检测线程
        detector_thread = threading.Thread(target=deadlock_detector, daemon=True)
        detector_thread.start()
        
        return detector_thread
    
    # 在事务管理器中添加死锁检测方法

    def find_nodes_by_metadata(self, metadata_key, metadata_value=None):
        """通过元数据查找节点，支持分区优化查询
        
        参数:
            metadata_key: 元数据键
            metadata_value: 可选的元数据值，如果为None则匹配所有具有该键的节点
            
        返回:
            list: 匹配的节点列表
        """
        # 检查分区查询缓存
        cache_key = f"metadata:{metadata_key}:{metadata_value}"
        if self.parallel_query and hasattr(self, 'partition_query_cache'):
            with self.partition_query_cache_lock:
                if cache_key in self.partition_query_cache:
                    cache_entry = self.partition_query_cache[cache_key]
                    if time.time() - cache_entry['timestamp'] < 60:  # 缓存1分钟
                        return cache_entry['result']
        
        # 确定要查询的分区
        query_conditions = {metadata_key: metadata_value} if metadata_value is not None else {}
        if self.partition_config["field"] == metadata_key:
            # 如果查询的正好是分区字段，可以直接定位分区
            partitions = self.get_partition_for_query(query_conditions)
        else:
            # 否则需要查询所有分区
            partitions = list(self.partitions.keys())
        
        # 定义过滤函数
        def metadata_filter(node):
            if node.metadata and metadata_key in node.metadata:
                if metadata_value is None:
                    return True
                return node.metadata[metadata_key] == metadata_value
            return False
        
        # 从分区中获取节点
        results = self._get_nodes_from_partitions(partitions, metadata_filter)
        
        # 更新缓存
        if self.parallel_query and hasattr(self, 'partition_query_cache'):
            with self.partition_query_cache_lock:
                self.partition_query_cache[cache_key] = {
                    'result': results,
                    'timestamp': time.time()
                }
                
                # 限制缓存大小
                if len(self.partition_query_cache) > 100:
                    # 删除最旧的缓存条目
                    oldest_key = min(
                        self.partition_query_cache.keys(),
                        key=lambda k: self.partition_query_cache[k]['timestamp']
                    )
                    del self.partition_query_cache[oldest_key]
        
        return results
    
    def find_nodes_by_tag(self, tag):
        """通过标签查找节点，支持分区优化查询
        
        参数:
            tag: 要查找的标签
            
        返回:
            list: 匹配的节点列表
        """
        # 检查分区查询缓存
        cache_key = f"tag:{tag}"
        if self.parallel_query and hasattr(self, 'partition_query_cache'):
            with self.partition_query_cache_lock:
                if cache_key in self.partition_query_cache:
                    cache_entry = self.partition_query_cache[cache_key]
                    if time.time() - cache_entry['timestamp'] < 60:  # 缓存1分钟
                        return cache_entry['result']
        
        # 所有分区都需要查询
        partitions = list(self.partitions.keys())
        
        # 定义过滤函数
        def tag_filter(node):
            return tag in (node.tags or [])
        
        # 从分区中获取节点
        results = self._get_nodes_from_partitions(partitions, tag_filter)
        
        # 更新缓存
        if self.parallel_query and hasattr(self, 'partition_query_cache'):
            with self.partition_query_cache_lock:
                self.partition_query_cache[cache_key] = {
                    'result': results,
                    'timestamp': time.time()
                }
                
                # 限制缓存大小
                if len(self.partition_query_cache) > 100:
                    # 删除最旧的缓存条目
                    oldest_key = min(
                        self.partition_query_cache.keys(),
                        key=lambda k: self.partition_query_cache[k]['timestamp']
                    )
                    del self.partition_query_cache[oldest_key]
        
        return results
    
    def find_nodes_by_filter(self, filter_func, partitions=None):
        """通过过滤函数查找节点，支持限定分区范围
        
        参数:
            filter_func: 过滤函数，接收一个节点参数，返回布尔值
            partitions: 可选的分区名称列表，如果为None则查询所有分区
            
        返回:
            list: 匹配的节点列表
        """
        # 确定要查询的分区
        if partitions is None:
            partitions = list(self.partitions.keys())
        else:
            # 验证分区名称
            partitions = [p for p in partitions if p in self.partitions]
            if not partitions:
                return []  # 没有有效的分区
        
        # 从分区中获取节点
        return self._get_nodes_from_partitions(partitions, filter_func)
    
    def clear_partition_cache(self):
        """清除分区查询缓存"""
        if self.parallel_query and hasattr(self, 'partition_query_cache'):
            with self.partition_query_cache_lock:
                self.partition_query_cache.clear()
    
    def optimize_data_distribution(self):
        """优化数据分布，重新分配节点到合适的分区"""
        if not self.auto_optimize:
            return False
            
        with self.lock:
            # 获取所有分区统计
            stats = self.get_partition_stats()
            
            # 合并小分区
            small_partitions = []
            for name, stat in stats.items():
                if name != "default" and name != "_summary" and stat.get("current_size", 0) < self.max_partition_size * 0.2:
                    small_partitions.append(name)
            
            if len(small_partitions) >= 2:
                self.merge_partitions(small_partitions)
            
            # 检查是否有不平衡的分区需要分裂
            for name, stat in stats.items():
                if name != "_summary" and stat.get("current_size", 0) > self.max_partition_size:
                    self.split_partition(name)
            
            # 检查分区命中率，可能需要重新组织
            low_hit_partitions = []
            for name, stat in stats.items():
                if name != "default" and name != "_summary" and stat.get("hit_ratio", 1.0) < 0.3:
                    low_hit_partitions.append(name)
            
            if low_hit_partitions:
                # 将低命中率分区中的节点重新分配
                for partition_name in low_hit_partitions:
                    with self.partition_locks[partition_name]:
                        node_keys = list(self.partitions[partition_name])
                        
                    # 分批处理，避免锁定太长时间
                    batch_size = 100
                    for i in range(0, len(node_keys), batch_size):
                        batch = node_keys[i:i+batch_size]
                        for key in batch:
                            node = self.get_node_by_key(key)
                            if node and node.metadata and self.partition_config["field"] in node.metadata:
                                # 重新计算适合的分区
                                field_value = node.metadata[self.partition_config["field"]]
                                if self.partition_config["strategy"] == "field":
                                    new_partition = self.create_dynamic_partition(field_value)
                                    self.assign_node_to_partition(node, new_partition)
        
        return True

    # 基础方法实现
    def _schedule_resize(self):
        """计划扩容任务，直接调用_check_resize方法"""
        self._check_resize()
    
    def remove_node_by_key_base(self, key):
        """通过键删除节点的基本方法，不处理索引和事务"""
        index = self.hf(key)
        with self.bucket_locks[index]:
            bucket = self.buckets[index]
            for i, node in enumerate(bucket):
                if node.key == key:
                    removed = bucket.pop(i)
                    self.count -= 1
                    
                    # 清理索引
                    self._cleanup_node_indices(removed)
                    
                    # 从组合数据结构中删除
                    self._combined_data.delete(key, keep_relationships=False)
                    
                    self._trigger_event('node_removed', node=removed)
                    return removed
            return None
    # 添加被引用但缺失的基础方法
    def update_node_metadata_base(self, node, new_metadata):
        """更新节点的元数据并同步索引的基础实现"""
        if not isinstance(new_metadata, dict):
            raise TypeError("元数据必须是字典类型")
            
        old_metadata = node.metadata.copy() if node.metadata else {}
        
        # 如果节点没有metadata属性，初始化它
        if node.metadata is None:
            node._metadata = {}
            
        # 获取旧元数据的键值对
        old_meta_pairs = set()
        if old_metadata:
            for key, value in old_metadata.items():
                old_meta_pairs.add((key, value))
        
        # 更新节点的元数据
        node._metadata.update(new_metadata)
        
        # 获取新元数据的键值对
        new_meta_pairs = set()
        for key, value in node.metadata.items():
            new_meta_pairs.add((key, value))
            
        # 需要移除的元数据索引
        to_remove = old_meta_pairs - new_meta_pairs
        
        # 需要添加的元数据索引
        to_add = new_meta_pairs - old_meta_pairs
        
        # 移除旧的元数据索引
        for meta_key, meta_val in to_remove:
            self._remove_metadata_index(meta_key, meta_val, node)
        
        # 添加新的元数据索引
        for meta_key, meta_val in to_add:
            self._index_metadata(meta_key, meta_val, node)
        
        # 更新组合数据结构
        self._combined_data.set_node_metadata(node.key, node.metadata)
        
        self._trigger_event('node_updated', node=node, metadata_changed=True)
        
        return node
        
    def update_node_tag_base(self, node, operation='add', tags=None, clear_existing=False):
        """更新节点的标签的基础实现"""
        # 获取节点对象
        if not node:
            raise ValueError(f"找不到节点")
            
        # 确保tags是可迭代的（如果是单个字符串，转换为列表）
        if tags is not None:
            if isinstance(tags, str):
                tags = [tags]
            elif not hasattr(tags, '__iter__'):
                tags = [tags]  # 处理其他单个非迭代对象
            
        # 根据操作类型处理标签
        if operation == 'add' and tags:
            # 添加新标签
            for tag in tags:
                node.add_tag(tag)
                
        elif operation == 'remove' and tags:
            # 移除指定标签
            for tag in tags:
                node.remove_tag(tag)
                
        elif operation == 'set':
            # 如果需要清除现有标签
            if clear_existing:
                # 保存现有标签的副本，以便稍后清除索引
                old_tags = set(node.tags)
                
                # 清空标签集合
                node._tags.clear()
                
                # 移除旧标签的索引
                for tag in old_tags:
                    self._remove_tag_index(tag, node)
            
            # 设置新标签
            if tags:
                for tag in tags:
                    node.add_tag(tag)
                    
        # 触发节点更新事件
        self._trigger_event('node_updated', node=node, tags_changed=True)
        
        return node
        
    def update_node_weight_base(self, node, weight):
        """更新节点的权重的基础实现"""
        # 保存旧权重以便事件通知
        old_weight = node.weight
        
        # 更新权重
        try:
            node.weight = float(weight)
        except (TypeError, ValueError):
            raise ValueError(f"权重必须是一个有效的数值: {weight}")
            
        # 触发节点更新事件
        self._trigger_event('node_updated', node=node, old_weight=old_weight)
        
        return node
    def update_relationship_weight(self, source, target, new_weight, rel_type=None):
        """更新关系权重
        
        Args:
            source: 源节点键或节点对象
            target: 目标节点键或节点对象  
            new_weight: 新的权重值
            rel_type: 关系类型（可选）
            
        Returns:
            bool: 是否更新成功
        """
        # 获取源节点和目标节点
        source_node = self._get_node_from_input(source)
        target_node = self._get_node_from_input(target)
        
        if not source_node or not target_node:
            return False
        
        # 查找并更新关系权重
        if source_node.r:
            for relation in source_node.r:
                # 检查目标节点和关系类型是否匹配
                if (relation['node'] == target_node and 
                    (rel_type is None or relation.get('type') == rel_type)):
                    # 更新权重
                    old_weight = relation.get('weight', 1.0)
                    relation['weight'] = new_weight
                    
                    # 触发同步事件
                    self._trigger_sync_event(
                        SyncOperation.UPDATE, 
                        node=source_node,
                        old_value=old_weight,
                        new_value=new_weight,
                        context={
                            'operation': 'update_relationship_weight',
                            'source': source_node.key,
                            'target': target_node.key,
                            'rel_type': rel_type
                        }
                    )
                    
                    return True
        
        return False
    
    def update_relationship_metadata(self, source, target, metadata, rel_type=None):
        """更新关系元数据
        
        Args:
            source: 源节点键或节点对象
            target: 目标节点键或节点对象
            metadata: 新的元数据字典
            rel_type: 关系类型（可选）
            
        Returns:
            bool: 是否更新成功
        """
        # 获取源节点和目标节点
        source_node = self._get_node_from_input(source)
        target_node = self._get_node_from_input(target)
        
        if not source_node or not target_node:
            return False
        
        # 查找并更新关系元数据
        if source_node.r:
            for relation in source_node.r:
                # 检查目标节点和关系类型是否匹配
                if (relation['node'] == target_node and 
                    (rel_type is None or relation.get('type') == rel_type)):
                    # 更新元数据
                    old_metadata = relation.get('metadata', {})
                    if metadata:
                        relation['metadata'] = metadata.copy()
                    else:
                        relation['metadata'] = {}
                    
                    # 触发同步事件
                    self._trigger_sync_event(
                        SyncOperation.UPDATE, 
                        node=source_node,
                        old_value=old_metadata,
                        new_value=metadata,
                        context={
                            'operation': 'update_relationship_metadata',
                            'source': source_node.key,
                            'target': target_node.key,
                            'rel_type': rel_type
                        }
                    )
                    
                    return True
        
        return False
    
    def get_relationship_info(self, source, target, rel_type=None):
        """获取关系详细信息
        
        Args:
            source: 源节点键或节点对象
            target: 目标节点键或节点对象
            rel_type: 关系类型（可选）
            
        Returns:
            dict: 关系信息，包含weight, metadata, type等
        """
        # 获取源节点和目标节点
        source_node = self._get_node_from_input(source)
        target_node = self._get_node_from_input(target)
        
        if not source_node or not target_node:
            return None
        
        # 查找关系
        if source_node.r:
            for relation in source_node.r:
                # 检查目标节点和关系类型是否匹配
                if (relation['node'] == target_node and 
                    (rel_type is None or relation.get('type') == rel_type)):
                    return {
                        'source': source_node.key,
                        'target': target_node.key,
                        'type': relation.get('type'),
                        'weight': relation.get('weight', 1.0),
                        'metadata': relation.get('metadata', {}),
                        'created_time': relation.get('created_time')
                    }
        
        return None
    def has_relationship(self,source,target):
        """获取节点之间是否有关系链
        Args:
            source: 源节点键或节点对象
            target: 目标节点键或节点对象
            
        Returns:
            bool: 节点之间是否有关系链
        """
        source_node=self._get_node_from_input(source)
        target_node=self._get_node_from_input(target)
        return target_node in source_node.r
    def get_graph_density(self):
        """
        计算图的密度
        
        对于无向图：密度 = 2 * |E| / (|V| * (|V| - 1))
        对于有向图：密度 = |E| / (|V| * (|V| - 1))
        
        其中 |E| 是边的数量，|V| 是节点的数量
        
        Returns:
            dict: 包含密度信息的字典
        """
        num_nodes = len(self)
        if num_nodes <= 1:
            return {
                'density': 0,
                'nodes': num_nodes,
                'edges': 0,
                'max_possible_edges': 0,
                'description': '图中节点数量不足，无法计算密度'
            }
        
        # 计算边的数量
        total_edges = 0
        directed_edges = 0
        undirected_edges = 0
        
        for bucket in self.buckets:
            if bucket is not None:
                current = bucket
                while current is not None:
                    node = current
                    if hasattr(node, 'r') and node.r:
                        for relation in node.r:
                            total_edges += 1
                            # 检查是否为双向边（简单判断：如果目标节点也有指向当前节点的边）
                            target_node = self.get_node_by_key(relation['target'])
                            if target_node and hasattr(target_node, 'r') and target_node.r:
                                reverse_exists = any(
                                    rel['target'] == node.key for rel in target_node.r
                                )
                                if reverse_exists:
                                    undirected_edges += 1
                                else:
                                    directed_edges += 1
                            else:
                                directed_edges += 1
                    current = getattr(current, 'next', None)
        
        # 调整边数计算（避免双向边重复计算）
        actual_undirected_edges = undirected_edges // 2  # 双向边只算一条
        total_unique_edges = actual_undirected_edges + directed_edges
        
        # 计算最大可能的边数
        max_possible_edges = num_nodes * (num_nodes - 1)  # 有向图的最大边数
        max_possible_undirected = max_possible_edges // 2  # 无向图的最大边数
        
        # 判断图的类型并计算密度
        if actual_undirected_edges > directed_edges:
            # 主要是无向图
            density = (2 * actual_undirected_edges) / max_possible_edges if max_possible_edges > 0 else 0
            graph_type = "主要为无向图"
        else:
            # 主要是有向图
            density = total_edges / max_possible_edges if max_possible_edges > 0 else 0
            graph_type = "主要为有向图"
        
        return {
            'density': round(density, 4),
            'nodes': num_nodes,
            'total_edges': total_edges,
            'directed_edges': directed_edges,
            'undirected_edges': actual_undirected_edges,
            'unique_edges': total_unique_edges,
            'max_possible_edges': max_possible_edges,
            'max_possible_undirected': max_possible_undirected,
            'graph_type': graph_type,
            'density_percentage': round(density * 100, 2),
            'description': f'图密度为 {round(density * 100, 2)}%，{graph_type}'
        }
    
    def analyze_node_density(self, node_key=None):
        """
        分析节点的局部密度
        
        Args:
            node_key: 要分析的节点key，如果为None则分析所有节点
            
        Returns:
            dict: 节点密度分析结果
        """
        if node_key:
            # 分析特定节点
            node = self.get_node_by_key(node_key)
            if not node:
                return {'error': f'节点 {node_key} 不存在'}
            
            return self._analyze_single_node_density(node)
        else:
            # 分析所有节点
            all_densities = []
            for bucket in self.buckets:
                if bucket is not None:
                    current = bucket
                    while current is not None:
                        node = current
                        density_info = self._analyze_single_node_density(node)
                        all_densities.append(density_info)
                        current = getattr(current, 'next', None)
            
            if not all_densities:
                return {'error': '没有找到任何节点'}
            
            # 计算统计信息
            densities = [info['local_density'] for info in all_densities]
            
            return {
                'total_nodes': len(all_densities),
                'average_density': round(sum(densities) / len(densities), 4),
                'max_density': max(densities),
                'min_density': min(densities),
                'density_distribution': {
                    'high_density_nodes': len([d for d in densities if d > 0.7]),
                    'medium_density_nodes': len([d for d in densities if 0.3 <= d <= 0.7]),
                    'low_density_nodes': len([d for d in densities if d < 0.3])
                },
                'top_dense_nodes': sorted(all_densities, key=lambda x: x['local_density'], reverse=True)[:10]
            }
    
    def _analyze_single_node_density(self, node):
        """
        分析单个节点的局部密度
        
        Args:
            node: IONNode对象
            
        Returns:
            dict: 单个节点的密度信息
        """
        if not hasattr(node, 'r') or not node.r:
            return {
                'node_key': node.key,
                'local_density': 0,
                'neighbors': 0,
                'possible_connections': 0,
                'actual_connections': 0
            }
        
        # 获取邻居节点
        neighbors = set()
        for relation in node.r:
            neighbors.add(relation['target'])
        
        neighbor_count = len(neighbors)
        if neighbor_count <= 1:
            return {
                'node_key': node.key,
                'local_density': 0,
                'neighbors': neighbor_count,
                'possible_connections': 0,
                'actual_connections': 0
            }
        
        # 计算邻居之间的连接数
        actual_connections = 0
        for neighbor1_key in neighbors:
            neighbor1 = self.get_node_by_key(neighbor1_key)
            if neighbor1 and hasattr(neighbor1, 'r') and neighbor1.r:
                for relation in neighbor1.r:
                    if relation['target'] in neighbors and relation['target'] != neighbor1_key:
                        actual_connections += 1
        
        # 避免重复计算双向边
        actual_connections = actual_connections // 2
        
        # 计算可能的最大连接数（邻居之间的完全图）
        possible_connections = neighbor_count * (neighbor_count - 1) // 2
        
        # 计算局部密度
        local_density = actual_connections / possible_connections if possible_connections > 0 else 0
        
        return {
            'node_key': node.key,
            'local_density': round(local_density, 4),
            'neighbors': neighbor_count,
            'possible_connections': possible_connections,
            'actual_connections': actual_connections,
            'density_percentage': round(local_density * 100, 2)
        }
    
    def get_density_statistics(self):
        """
        获取图的全面密度统计信息
        
        Returns:
            dict: 包含各种密度统计的详细信息
        """
        graph_density = self.get_graph_density()
        node_density = self.analyze_node_density()
        
        # 计算连通性相关的密度指标
        components_info = self._analyze_connected_components()
        
        return {
            'global_density': graph_density,
            'node_density_analysis': node_density,
            'connectivity_analysis': components_info,
            'density_insights': self._generate_density_insights(graph_density, node_density)
        }
    
    def _analyze_connected_components(self):
        """
        分析连通分量
        
        Returns:
            dict: 连通分量信息
        """
        visited = set()
        components = []
        
        for bucket in self.buckets:
            if bucket is not None:
                current = bucket
                while current is not None:
                    node = current
                    if node.key not in visited:
                        component = self._get_connected_component(node, visited)
                        if component:
                            components.append(component)
                    current = getattr(current, 'next', None)
        
        if not components:
            return {'components': 0, 'largest_component_size': 0, 'average_component_size': 0}
        
        component_sizes = [len(comp) for comp in components]
        
        return {
            'components': len(components),
            'largest_component_size': max(component_sizes),
            'smallest_component_size': min(component_sizes),
            'average_component_size': round(sum(component_sizes) / len(component_sizes), 2),
            'component_sizes': component_sizes
        }
    
    def _get_connected_component(self, start_node, visited):
        """
        获取从指定节点开始的连通分量
        
        Args:
            start_node: 起始节点
            visited: 已访问节点集合
            
        Returns:
            list: 连通分量中的节点列表
        """
        component = []
        stack = [start_node]
        
        while stack:
            node = stack.pop()
            if node.key not in visited:
                visited.add(node.key)
                component.append(node.key)
                
                # 添加所有邻居
                if hasattr(node, 'r') and node.r:
                    for relation in node.r:
                        neighbor = self.get_node_by_key(relation['target'])
                        if neighbor and neighbor.key not in visited:
                            stack.append(neighbor)
        
        return component
    
    def _generate_density_insights(self, graph_density, node_density):
        """
        生成密度分析洞察
        
        Args:
            graph_density: 全局密度信息
            node_density: 节点密度信息
            
        Returns:
            list: 洞察列表
        """
        insights = []
        
        density = graph_density['density']
        
        if density < 0.1:
            insights.append("图非常稀疏，大部分节点连接较少")
        elif density < 0.3:
            insights.append("图较为稀疏，适合使用邻接表存储")
        elif density < 0.7:
            insights.append("图密度适中，连接性良好")
        else:
            insights.append("图非常稠密，大部分节点之间都有连接")
        
        if 'average_density' in node_density:
            avg_local = node_density['average_density']
            if avg_local > density * 2:
                insights.append("存在明显的聚类结构，局部密度远高于全局密度")
            elif avg_local < density * 0.5:
                insights.append("图结构较为分散，缺乏明显的聚类")
        
        return insights
    # ==================== 表查找功能 ====================
    


    def _normalize_row_id(self, row_id):
        """将行标识转换为可哈希对象，避免 'unhashable type' 错误"""
        if isinstance(row_id, (list, set)):
            return tuple(row_id)
        if isinstance(row_id, dict):
            # 按键排序后转为元组，保证稳定性
            return tuple(sorted(row_id.items()))
        return row_id

    # ---------- 行索引 ----------

    def _index_row(self, row_id, node):
        """为节点建立行索引"""
        if row_id is None:
            return
        row_id = self._normalize_row_id(row_id)

        with self.row_index_lock:
            self.row_index.setdefault(row_id, [])
            if node not in self.row_index[row_id]:
                self.row_index[row_id].append(node)

    def _remove_row_index(self, row_id, node):
        """移除节点的行索引"""
        if row_id is None:
            return
        row_id = self._normalize_row_id(row_id)

        with self.row_index_lock:
            bucket = self.row_index.get(row_id)
            if bucket and node in bucket:
                bucket.remove(node)
                if not bucket:
                    del self.row_index[row_id]

    def update_node_row(self, node, new_row):
        """更新节点的行标识并同步索引"""
        old_row = node._row
        norm_old = self._normalize_row_id(old_row) if old_row is not None else None
        norm_new = self._normalize_row_id(new_row) if new_row is not None else None

        if norm_old is not None:
            self._remove_row_index(norm_old, node)

        node._row = norm_new

        if norm_new is not None:
            self._index_row(norm_new, node)

        if self.enable_sync:
            self._trigger_sync_event(
                SyncOperation.UPDATE,
                node=node,
                old_value=old_row,
                new_value=norm_new,
                context={'field': 'row'}
            )
        return node

    def find_nodes_by_row(self, row_id):
        """根据行标识查找节点"""
        if row_id is None:
            return []
        row_id = self._normalize_row_id(row_id)

        with self.row_index_lock:
            return list(self.row_index.get(row_id, []))

    # ---------- 表视图 ----------

    def create_table_view(self, table_name, row_column_mapping):
        """
        创建表视图
        Args:
            table_name: 表名
            row_column_mapping: {row_id: {column: node_key}}
        """
        with self.table_index_lock:
            self.table_index.setdefault(table_name, {})
            for row_id, columns in row_column_mapping.items():
                tbl_row = self.table_index[table_name].setdefault(row_id, {})
                for col, key in columns.items():
                    node = self.get_node_by_key(key)
                    if node:
                        self.update_node_row(node, row_id)
                        tbl_row[col] = node

    def get_table_row(self, table_name, row_id):
        """获取指定行"""
        with self.table_index_lock:
            return self.table_index.get(table_name, {}).get(row_id, {})

    def get_table_column(self, table_name, column_name):
        """获取整列"""
        col_data = []
        with self.table_index_lock:
            for row_id, row in self.table_index.get(table_name, {}).items():
                if column_name in row:
                    col_data.append({'row_id': row_id, 'node': row[column_name]})
        return col_data

    def query_table(self, table_name, conditions=None, columns=None, limit=None):
        """
        通用查询
        conditions 可为 dict 或可调用对象；columns=None 返回整行
        """
        results = []
        with self.table_index_lock:
            tbl = self.table_index.get(table_name, {})
            for row_id, row in tbl.items():
                match = True
                if conditions:
                    if callable(conditions):
                        match = conditions(row)
                    elif isinstance(conditions, dict):
                        for c, v in conditions.items():
                            if c not in row or row[c].val != v:
                                match = False
                                break
                if not match:
                    continue

                res_row = {'row_id': row_id}
                if columns:
                    for col in columns:
                        if col in row:
                            res_row[col] = row[col]
                else:
                    res_row.update(row)
                results.append(res_row)

                if limit and len(results) >= limit:
                    break
        return results

    def batch_create_table_nodes(self, table_data, table_name=None):
        """
        批量创建节点并可自动生成表视图
        table_data: [{'row': row_id, 'columns': {col: val}}]
        """
        created, mapping = [], {}
        for item in table_data:
            row_id, cols = item.get('row'), item.get('columns', {})
            if table_name:
                mapping.setdefault(row_id, {})
            for col, val in cols.items():
                key = f"{table_name}_{row_id}_{col}" if table_name else f"{row_id}_{col}"
                node = self.create_node(key=key, val=val,
                                        metadata={'table': table_name, 'column': col},
                                        row=row_id)
                created.append(node)
                if table_name:
                    mapping[row_id][col] = key
        if table_name and mapping:
            self.create_table_view(table_name, mapping)
        return created

    def get_table_stats(self, table_name=None):
        """返回单表或全部表统计信息"""
        if table_name:
            with self.table_index_lock:
                tbl = self.table_index.get(table_name)
                if not tbl:
                    return None
                return {
                    'table_name': table_name,
                    'row_count': len(tbl),
                    'columns': set(col for row in tbl.values() for col in row),
                    'total_cells': sum(len(r) for r in tbl.values())
                }
        stats = {}
        with self.table_index_lock:
            for name in self.table_index:
                stats[name] = self.get_table_stats(name)
        with self.row_index_lock:
            stats['_row_index'] = {
                'total_rows': len(self.row_index),
                'total_nodes_with_rows': sum(len(nodes) for nodes in self.row_index.values())
            }
        return stats

    def optimize_table_indices(self):
        """清理空行 / 空表，保持索引紧凑"""
        with self.row_index_lock:
            for rid in [r for r, nodes in self.row_index.items() if not nodes]:
                del self.row_index[rid]

        with self.table_index_lock:
            for tbl, rows in list(self.table_index.items()):
                for rid in [r for r, row in rows.items() if not row]:
                    del rows[rid]
                if not rows:
                    del self.table_index[tbl]

    # ---------- 高级：JOIN 与 GROUP BY ----------

    def table_join(self, left_table, right_table,
                   join_type="inner",
                   on=None,
                   suffixes=("_x", "_y"),
                   limit=None):
        """
        支持 inner/left/right/outer 四种连接
        on:
            None                  -> 两表共有列
            {"lcol": "rcol"}      -> dict
            [("lcol1","rcol1"),]  -> 列对列表
        """
        def normalize_on(lcols, rcols):
            if on is None:
                return [(c, c) for c in lcols & rcols]
            if isinstance(on, dict):
                return list(on.items())
            return list(on)

        with self.table_index_lock:
            if left_table not in self.table_index or right_table not in self.table_index:
                return []

            l_rows, r_rows = self.table_index[left_table], self.table_index[right_table]
            l_cols = {c for r in l_rows.values() for c in r}
            r_cols = {c for r in r_rows.values() for c in r}
            pairs = normalize_on(l_cols, r_cols)
            if not pairs:
                return []

            # 右表键→行列表
            r_idx = {}
            for r_id, rdat in r_rows.items():
                k = tuple(rdat[p[1]].val if p[1] in rdat else None for p in pairs)
                r_idx.setdefault(k, []).append((r_id, rdat))

            out = []

            def emit(lid, ldat, rid, rdat):
                row = {"row_id_left": lid, "row_id_right": rid}
                for c, n in ldat.items():
                    row[c + suffixes[0] if c in rdat else c] = n
                for c, n in rdat.items():
                    if c in ldat:
                        row[c + suffixes[1]] = n
                    else:
                        row[c] = n
                out.append(row)

            for lid, ldat in l_rows.items():
                k = tuple(ldat[p[0]].val if p[0] in ldat else None for p in pairs)
                matches = r_idx.get(k, [])
                if matches:
                    for rid, rdat in matches:
                        emit(lid, ldat, rid, rdat)
                        if limit and len(out) >= limit:
                            return out
                elif join_type in ("left", "outer"):
                    emit(lid, ldat, None, {})

            if join_type in ("right", "outer"):
                l_keys = {tuple(ldat[p[0]].val if p[0] in ldat else None for p in pairs)
                          for ldat in l_rows.values()}
                for rid, rdat in r_rows.items():
                    k = tuple(rdat[p[1]].val if p[1] in rdat else None for p in pairs)
                    if k not in l_keys:
                        emit(None, {}, rid, rdat)
                        if limit and len(out) >= limit:
                            return out
            return out

    def group_table(self, table_name, group_by,
                    agg_funcs=None,
                    having=None,
                    limit=None):
        """
        GROUP BY + 聚合
        agg_funcs: {"col": "sum" | "avg" | "min" | "max" | "count" | callable}
        """
        if agg_funcs is None:
            agg_funcs = {}

        with self.table_index_lock:
            tbl = self.table_index.get(table_name)
            if not tbl:
                return []

            grouped = {}
            for rid, row in tbl.items():
                key = tuple(row[c].val if c in row else None for c in group_by)
                bucket = grouped.setdefault(key, defaultdict(list))
                for col, node in row.items():
                    bucket[col].append(node.val)

            def apply_agg(values, func):
                if isinstance(func, str):
                    if func == "count":
                        return len(values)
                    if func == "sum":
                        return sum(values)
                    if func == "avg":
                        return sum(values) / len(values) if values else None
                    if func == "min":
                        return min(values) if values else None
                    if func == "max":
                        return max(values) if values else None
                    raise ValueError(func)
                return func(values)

            results = []
            for k, cols in grouped.items():
                row = {group_by[i]: k[i] for i in range(len(group_by))}
                row["count"] = len(next(iter(cols.values())))  # 近似 count(*)
                for col, func in agg_funcs.items():
                    if col in cols:
                        row[f"{col}_{func if isinstance(func,str) else func.__name__}"] = \
                            apply_agg(cols[col], func)
                if having and not having(row):
                    continue
                results.append(row)
                if limit and len(results) >= limit:
                    break
            return results
    def get_node_by_row(self, row_id):
        """根据行标识获取单个节点
        
        Args:
            row_id: 行标识符
            
        Returns:
            IONNode: 找到的第一个节点，如果没有找到则返回None
            
        Note:
            如果有多个节点具有相同的row_id，返回第一个找到的节点
            要获取所有相同row_id的节点，请使用find_nodes_by_row()
        """
        if row_id is None:
            return None
            
        row_id = self._normalize_row_id(row_id)
        
        with self.row_index_lock:
            nodes = self.row_index.get(row_id, [])
            return nodes[0] if nodes else None
    
    def get_nodes_by_row_pattern(self, pattern):
        """根据行标识模式获取节点
        
        Args:
            pattern: 行标识模式（支持通配符 * 和 ?）
            
        Returns:
            List[IONNode]: 匹配模式的所有节点
        """
        import fnmatch
        
        if pattern is None:
            return []
            
        pattern = self._normalize_row_id(pattern)
        matching_nodes = []
        
        with self.row_index_lock:
            for row_id, nodes in self.row_index.items():
                if fnmatch.fnmatch(str(row_id), str(pattern)):
                    matching_nodes.extend(nodes)
        
        return matching_nodes
    
    def get_row_statistics(self):
        """获取行索引统计信息
        
        Returns:
            dict: 包含行索引统计信息的字典
        """
        with self.row_index_lock:
            total_rows = len(self.row_index)
            total_nodes = sum(len(nodes) for nodes in self.row_index.values())
            
            # 计算每行节点数的分布
            nodes_per_row = [len(nodes) for nodes in self.row_index.values()]
            
            stats = {
                'total_rows': total_rows,
                'total_nodes_with_rows': total_nodes,
                'avg_nodes_per_row': total_nodes / total_rows if total_rows > 0 else 0,
                'max_nodes_per_row': max(nodes_per_row) if nodes_per_row else 0,
                'min_nodes_per_row': min(nodes_per_row) if nodes_per_row else 0,
                'rows_with_multiple_nodes': sum(1 for count in nodes_per_row if count > 1),
                'unique_row_ids': total_rows
            }
            
            return stats
    def get_row_id(self, node_or_key):
        """返回节点当前的行 ID（若无则返回 None）
    
        Args:
            node_or_key: 节点对象或节点键
    
        Returns:
            str | None
        """
        node = self._get_node_from_input(node_or_key)
        return node.row if node else None
    def get_or_assign_row_id(self, node_or_key, row_id=None,
                         auto_generate=True, prefix="row_", start_index=1):
        """获取节点行 ID；若不存在则分配一个新的行 ID
    
        Args:
            node_or_key:  节点对象或节点键
            row_id:       指定行 ID；若传入则直接设置为该 ID
            auto_generate:当 row_id 为空且节点无行 ID 时，是否自动生成
            prefix:       自动生成行 ID 的前缀
            start_index:  自动生成行 ID 的起始编号
    
        Returns:
            str | None: 最终的行 ID；如果无法分配返回 None
        """
        node = self._get_node_from_input(node_or_key)
        if not node:
            return None
    
        # 若已有行 ID，直接返回
        if node.row:
            return node.row
    
        # 如果调用方显式给出了行 ID
        if row_id:
            self.update_node_row(node, row_id)
            return row_id
    
        # 自动生成唯一行 ID
        if auto_generate:
            with self.row_index_lock:
                idx = start_index
                while True:
                    candidate = f"{prefix}{idx}"
                    if candidate not in self.row_index:
                        self.update_node_row(node, candidate)
                        return candidate
                    idx += 1
        return None
    # =======================机器学习智能方法===============================
    def enable_ml_learning(self, enabled=True):
        """启用/禁用机器学习"""
        if hasattr(self, 'ml_engine'):
            self.ml_engine.enable_learning(enabled)
    
    def get_smart_cache_suggestions(self, top_k=10):
        """获取智能缓存建议"""
        if hasattr(self, 'ml_engine'):
            return self.ml_engine.predict_cache_candidates(top_k)
        return []
    
    def optimize_partitions_ml(self):
        """基于机器学习优化分区"""
        if hasattr(self, 'ml_engine'):
            suggestions = self.ml_engine.optimize_partitions()
            
            # 应用分区建议
            for partition_name, node_keys in suggestions.items():
                try:
                    self.create_partition(partition_name, f"ML优化分区: {len(node_keys)}个节点")
                    for key in node_keys:
                        node = self.get_node_by_key(key)
                        if node:
                            self.assign_node_to_partition(node, partition_name)
                except Exception as e:
                    print(f"应用分区建议失败: {e}")
            
            return suggestions
        return {}
    
    def detect_access_anomalies(self):
        """检测访问异常"""
        if hasattr(self, 'ml_engine'):
            return self.ml_engine.detect_anomalies()
        return []
    
    def predict_query_time(self, query_type, context=None):
        """预测查询执行时间"""
        if hasattr(self, 'ml_engine'):
            return self.ml_engine.predict_query_performance(query_type, context)
        return None
    
    def smart_astar_search(self, start, goal, **kwargs):
        """智能A*搜索（使用ML优化的启发式）"""
        start_node = self._get_node_from_input(start)
        goal_node = self._get_node_from_input(goal)
        
        if not start_node or not goal_node:
            return None
        
        # 尝试获取ML优化的启发式函数
        smart_heuristic = None
        if hasattr(self, 'ml_engine'):
            smart_heuristic = self.ml_engine.optimize_path_finding(start_node.key, goal_node.key)
        
        # 使用智能启发式或回退到标准A*
        if smart_heuristic:
            return self.astar_search(start, goal, heuristic_func=smart_heuristic, **kwargs)
        else:
            return self.astar_search(start, goal, **kwargs)
    

    def _extract_node_key_from_arg(self, arg):
        """从参数中提取节点键"""
        if isinstance(arg, str):
            return arg
        elif hasattr(arg, 'key'):
            return arg.key
        elif isinstance(arg, (list, tuple)) and len(arg) > 0:
            if isinstance(arg[0], str):
                return arg[0]
            elif hasattr(arg[0], 'key'):
                return arg[0].key
        return None
    
    def _analyze_method_result(self, result, method_name, access_type):
        """分析方法结果"""
        analysis = {
            'type': type(result).__name__,
            'count': 0,
            'size': 0
        }
        
        if result is None:
            analysis['count'] = 0
            analysis['size'] = 0
        elif isinstance(result, (list, tuple, set)):
            analysis['count'] = len(result)
            analysis['size'] = len(str(result))
        elif isinstance(result, dict):
            analysis['count'] = len(result)
            analysis['size'] = len(str(result))
        elif hasattr(result, '__len__'):
            try:
                analysis['count'] = len(result)
                analysis['size'] = len(str(result))
            except:
                analysis['count'] = 1
                analysis['size'] = len(str(result))
        else:
            analysis['count'] = 1 if result else 0
            analysis['size'] = len(str(result))
        
        return analysis
    
    def _record_io_for_training(self, method_name, args, kwargs, result, execution_time, access_type):
        """记录输入输出对用于训练"""
        if not hasattr(self, 'ml_engine'):
            return
        
        # 构建训练数据
        training_data = {
            'method': method_name,
            'access_type': access_type,
            'input_features': self._extract_input_features(args, kwargs),
            'output_features': self._extract_output_features(result),
            'execution_time': execution_time,
            'timestamp': time.time(),
            'context': {
                'node_count': self.count,
                'bucket_count': len(self.buckets),
                'load_factor': self.get_current_load_factor()
            }
        }
        
        # 添加到ML引擎的训练数据
        if not hasattr(self.ml_engine, 'io_training_data'):
            self.ml_engine.io_training_data = []
        
        self.ml_engine.io_training_data.append(training_data)
        
        # 限制训练数据大小，保持最近的1000条记录
        if len(self.ml_engine.io_training_data) > 1000:
            self.ml_engine.io_training_data = self.ml_engine.io_training_data[-1000:]
        
        # 每100条记录触发一次增量训练
        if len(self.ml_engine.io_training_data) % 100 == 0:
            self._trigger_io_training()
    
    def _extract_input_features(self, args, kwargs):
        """从输入参数中提取特征"""
        features = {
            'args_count': len(args),
            'kwargs_count': len(kwargs),
            'has_string_args': any(isinstance(arg, str) for arg in args),
            'has_list_args': any(isinstance(arg, (list, tuple)) for arg in args),
            'has_dict_args': any(isinstance(arg, dict) for arg in args),
            'total_input_size': sum(len(str(arg)) for arg in args) + sum(len(str(v)) for v in kwargs.values())
        }
        
        # 添加特定参数特征
        if args:
            features['first_arg_type'] = type(args[0]).__name__
            features['first_arg_size'] = len(str(args[0]))
        
        return features
    
    def _extract_output_features(self, result):
        """从输出结果中提取特征"""
        features = {
            'result_type': type(result).__name__,
            'result_size': len(str(result)),
            'is_empty': result is None or (hasattr(result, '__len__') and len(result) == 0)
        }
        
        if isinstance(result, (list, tuple, set)):
            features['result_count'] = len(result)
            features['has_nodes'] = any(hasattr(item, 'key') for item in result)
        elif isinstance(result, dict):
            features['result_count'] = len(result)
            features['dict_keys'] = list(result.keys())[:5]  # 只保存前5个键
        elif hasattr(result, 'key'):
            features['result_count'] = 1
            features['is_node'] = True
        else:
            features['result_count'] = 1 if result else 0
        
        return features
    
    def _record_batch_statistics(self, method_name, args, result, execution_time):
        """记录批量操作统计"""
        if not hasattr(self, 'ml_engine'):
            return
        
        batch_stats = {
            'method': method_name,
            'input_size': len(args[0]) if args and hasattr(args[0], '__len__') else 0,
            'output_size': len(result) if hasattr(result, '__len__') else 0,
            'execution_time': execution_time,
            'throughput': (len(args[0]) / execution_time) if args and hasattr(args[0], '__len__') and execution_time > 0 else 0,
            'timestamp': time.time()
        }
        
        if not hasattr(self.ml_engine, 'batch_statistics'):
            self.ml_engine.batch_statistics = []
        
        self.ml_engine.batch_statistics.append(batch_stats)
        
        # 保持最近的500条批量统计
        if len(self.ml_engine.batch_statistics) > 500:
            self.ml_engine.batch_statistics = self.ml_engine.batch_statistics[-500:]

    def _trigger_io_training(self):
        """触发输入输出训练"""
        if not hasattr(self, 'ml_engine') or not hasattr(self.ml_engine, 'io_training_data'):
            return
        
        try:
            # 使用训练数据进行增量学习
            self.ml_engine.train_io_predictor()
        except Exception as e:
            print(f"IO训练失败: {e}")
    def get_ml_stats(self):
        """获取机器学习统计信息"""
        if hasattr(self, 'ml_engine'):
            return self.ml_engine.get_intelligence_stats()
        return {'enabled': False, 'reason': 'ML引擎未初始化'}
    def trigger_ml_learning(self, force=False):
        """手动触发机器学习"""
        if hasattr(self, 'ml_engine'):
            if force:
                return self.ml_engine.incremental_learning()
            else:
                return self.ml_engine.adaptive_learning_schedule()
        return False
    
    def tune_ml_hyperparameters(self):
        """手动触发超参数调优"""
        if hasattr(self, 'ml_engine'):
            return self.ml_engine.auto_hyperparameter_tuning()
        return {}
    
    def get_ml_recommendations(self):
        """获取ML推荐"""
        if not hasattr(self, 'ml_engine'):
            return {}
        
        return {
            'cache_suggestions': self.get_smart_cache_suggestions(10),
            'partition_optimization': self.ml_engine.optimize_partitions(),
            'anomalies': self.detect_access_anomalies(),
            'performance_stats': self.ml_engine.get_intelligence_stats()
        }
    
    def enable_auto_ml_optimization(self, enabled=True):
        """启用/禁用自动ML优化"""
        if hasattr(self, 'ml_engine'):
            self.ml_engine.enable_learning(enabled)
            if enabled:
                # 启动后台自适应学习
                def auto_learning_worker():
                    import threading
                    import time
                    while getattr(self.ml_engine, 'learning_enabled', False):
                        time.sleep(300)  # 每5分钟检查一次
                        try:
                            self.ml_engine.adaptive_learning_schedule()
                        except Exception as e:
                            print(f"自动学习失败: {e}")
                
                learning_thread = threading.Thread(target=auto_learning_worker, daemon=True)
                learning_thread.start()
    def clear_ml_data(self):
            """清除机器学习数据"""
            if hasattr(self, 'ml_engine'):
                self.ml_engine.clear_learning_data()
    
# ===========特殊方法=============
import subprocess
import os
def system(file=None):
    if file is None:
        os.system(f'python3 /Users/$(whoami)/$(cat /usr/local/cpl/cpl_pathset)')
    else:
        subprocess.run(['python3','app_ion.py',file])