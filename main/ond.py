#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 17 00:46:25 2025

@author: wuzhixiang
"""
from bsd import HashTable, NestedBidirectionalMap, RelationshipChain
import pickle
import hashlib
import concurrent.futures
import threading
from collections import deque, Counter
from typing import Any, List, Dict, Optional, Union, Tuple, Set, Callable

def obj_to_number(obj, bits=64):
    """
    将任意可pickle的对象转换为固定位数的数字
    
    参数:
    obj: 任意可pickle的Python对象
    bits: 输出数字的位数，默认为64位
    
    返回:
    int: 转换后的数字，范围在0到2^bits-1之间
    """
    try:
        # 使用pickle序列化对象
        serialized = pickle.dumps(obj)
        
        # 使用SHA-256哈希算法生成摘要
        hash_object = hashlib.sha256(serialized)
        hash_digest = hash_object.digest()
        
        # 将哈希摘要转换为整数
        full_number = int.from_bytes(hash_digest, 'big')
        
        # 截取指定位数的数字
        max_value = 2 ** bits
        return full_number % max_value
    
    except (pickle.PicklingError, TypeError) as e:
        print(f"对象无法被pickle序列化: {e}")
        return None

class RNode:
    """增强的关系节点类"""
    def __init__(self, key, val, metadata=None):
        self._key = key
        self._val = val
        self._r = []  # 关系列表
        self._metadata = metadata or {}
        self._weight = 1.0  # 节点权重
        self._visited = False  # 用于遍历
        self._created_at = None  # 创建时间
        self._updated_at = None  # 更新时间
        self._ond_reference = None  # 引用所属的OND实例，用于元数据更新
        self._tags = set()  # 节点标签
        
    @property
    def key(self):
        """键属性访问器"""
        return self._key
        
    @key.setter
    def key(self, new_key):
        """键属性设置器"""
        if self._ond_reference:
            # 如果有OND引用，通过OND更新键
            self._ond_reference.update_node_key(self, new_key)
        else:
            self._key = new_key
            
    @property
    def val(self):
        """值属性访问器"""
        return self._val
        
    @val.setter
    def val(self, new_val):
        """值属性设置器"""
        if self._ond_reference:
            # 如果有OND引用，通过OND更新值
            self._ond_reference.update_node_value(self, new_val)
        else:
            self._val = new_val
            
    @property
    def metadata(self):
        """元数据属性访问器"""
        return self._metadata
        
    @metadata.setter
    def metadata(self, new_metadata):
        """元数据属性设置器"""
        if self._ond_reference:
            # 如果有OND引用，通过OND更新元数据
            self._ond_reference.update_node_metadata(self, new_metadata)
        else:
            self._metadata = new_metadata or {}
            
    @property
    def r(self):
        """关系列表属性访问器"""
        return self._r
        
    @r.setter
    def r(self, new_relations):
        """关系列表属性设置器"""
        if self._ond_reference:
            # 如果有OND引用，通过OND更新关系
            self._ond_reference.update_node_relations(self, new_relations)
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
        if self._ond_reference:
            self._ond_reference._index_tag(tag, self)
            
    def remove_tag(self, tag):
        """移除标签"""
        if tag in self._tags:
            self._tags.remove(tag)
            if self._ond_reference:
                self._ond_reference._remove_tag_index(tag, self)
        
    def add_relation(self, node, rel_type=None, rel_weight=1.0):
        """添加带类型和权重的关系"""
        relation = {
            'node': node,
            'type': rel_type,
            'weight': rel_weight
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
        
        # 如果有OND引用，同步更新索引
        if self._ond_reference and old_value != value:
            if old_value is not None:
                self._ond_reference._remove_metadata_index(key, old_value, self)
            self._ond_reference._index_metadata(key, value, self)
        
    def __str__(self):
        return f"RNode(key={self._key}, val={self._val}, relations={len(self._r)})"

class ThreadSafeDict:
    """线程安全的字典实现"""
    def __init__(self):
        self._dict = {}
        self._lock = threading.RLock()
        
    def __getstate__(self):
        """自定义序列化行为"""
        state = self.__dict__.copy()
        # 不序列化锁对象
        del state['_lock']
        return state
        
    def __setstate__(self, state):
        """自定义反序列化行为"""
        self.__dict__.update(state)
        # 重新创建锁对象
        self._lock = threading.RLock()
        
    def __getitem__(self, key):
        with self._lock:
            return self._dict[key]
    
    def __setitem__(self, key, value):
        with self._lock:
            self._dict[key] = value
            
    def __delitem__(self, key):
        with self._lock:
            del self._dict[key]
            
    def __contains__(self, key):
        with self._lock:
            return key in self._dict
            
    def get(self, key, default=None):
        with self._lock:
            return self._dict.get(key, default)
            
    def items(self):
        with self._lock:
            return list(self._dict.items())
            
    def keys(self):
        with self._lock:
            return list(self._dict.keys())
            
    def values(self):
        with self._lock:
            return list(self._dict.values())

class BiDirectionalMap:
    """双向映射实现"""
    def __init__(self):
        self._forward = {}
        self._backward = {}
        self._lock = threading.RLock()
        
    def __getstate__(self):
        """自定义序列化行为"""
        state = self.__dict__.copy()
        # 不序列化锁对象
        del state['_lock']
        return state
        
    def __setstate__(self, state):
        """自定义反序列化行为"""
        self.__dict__.update(state)
        # 重新创建锁对象
        self._lock = threading.RLock()
        
    def add(self, key, value):
        with self._lock:
            self._forward[key] = value
            self._backward[value] = key
            
    def get_by_key(self, key):
        with self._lock:
            return self._forward.get(key)
            
    def get_by_value(self, value):
        with self._lock:
            return self._backward.get(value)
            
    def remove_key(self, key):
        with self._lock:
            if key in self._forward:
                value = self._forward[key]
                del self._forward[key]
                if value in self._backward:
                    del self._backward[value]
                    
    def remove_value(self, value):
        with self._lock:
            if value in self._backward:
                key = self._backward[value]
                del self._backward[value]
                if key in self._forward:
                    del self._forward[key]

class OND:
    """增强的对象网络数据库"""
    rnode = RNode
    
    def __init__(self, start=None, size=1024, max_workers=4, load_factor_threshold=0.75):
        '''关系链+元数据+哈希表+双向映射'''
        self.buckets = [[] for _ in range(size)]
        self.size = size
        self.count = 0  # 实际节点数
        self.start = [start] if start else []
        self.bimap = BiDirectionalMap()  # 双向映射
        self.metadata_index = ThreadSafeDict()  # 元数据索引
        self.tag_index = ThreadSafeDict()  # 标签索引
        self.value_type_index = ThreadSafeDict()  # 值类型索引
        self.lock = threading.RLock()  # 全局锁
        self.bucket_locks = [threading.RLock() for _ in range(size)]  # 桶锁
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.load_factor_threshold = load_factor_threshold  # 负载因子阈值
        self.resizing_lock = threading.RLock()  # 扩容专用锁
        self.is_resizing = False  # 是否正在扩容
        
    def hf(self, obj):
        """哈希函数"""
        return obj_to_number(obj) % self.size
        
    def create_node(self, key, val, metadata=None):
        """创建新节点或更新现有节点"""
        # 检查是否需要扩容
        self._check_resize()
        
        index = self.hf(key)
        with self.bucket_locks[index]:
            nodes = self.buckets[index]
            for node in nodes:
                if node.key == key:
                    node.val = val
                    if metadata:
                        # 如果更新元数据，确保更新索引
                        self.update_node_metadata(node, metadata)
                    return node
                    
            new_node = RNode(key, val, metadata=metadata)
            new_node._ond_reference = self  # 设置OND引用
            nodes.append(new_node)
            self.count += 1
            self.bimap.add(key, val)
            
            # 索引元数据
            if metadata:
                for m_key, m_val in metadata.items():
                    self._index_metadata(m_key, m_val, new_node)
                    
            return new_node
            
    def _index_metadata(self, meta_key, meta_val, node):
        """为元数据创建索引"""
        key = f"{meta_key}:{meta_val}"
        
        # 使用get方法检查和初始化，避免KeyError
        nodes = self.metadata_index.get(key)
        if nodes is None:
            self.metadata_index[key] = [node]
        else:
            nodes.append(node)
        
    def find_by_metadata(self, meta_key, meta_val):
        """通过元数据查找节点"""
        key = f"{meta_key}:{meta_val}"
        return self.metadata_index.get(key, [])
        
    def get_node_by_key(self, key):
        """通过键获取节点"""
        index = self.hf(key)
        with self.bucket_locks[index]:
            for node in self.buckets[index]:
                if node.key == key:
                    return node
        return None
        
    def get_node_by_value(self, val):
        """通过值获取节点"""
        key = self.bimap.get_by_value(val)
        if key:
            return self.get_node_by_key(key)
        
        # 如果双向映射未找到，则遍历搜索
        for bucket in self.buckets:
            for node in bucket:
                if node.val == val:
                    return node
        return None
        
    def remove_node_by_key(self, key):
        """通过键删除节点"""
        index = self.hf(key)
        with self.bucket_locks[index]:
            bucket = self.buckets[index]
            for i, node in enumerate(bucket):
                if node.key == key:
                    removed = bucket.pop(i)
                    self.count -= 1
                    self.bimap.remove_key(key)
                    return removed
        return None
        
    def remove_node_by_value(self, val):
        """通过值删除节点"""
        key = self.bimap.get_by_value(val)
        if key:
            return self.remove_node_by_key(key)
            
        # 如果双向映射未找到，则遍历搜索
        for i, bucket in enumerate(self.buckets):
            with self.bucket_locks[i]:
                for j, node in enumerate(bucket):
                    if node.val == val:
                        removed = bucket.pop(j)
                        self.count -= 1
                        return removed
        return None
        
    def add_relationship(self, source, target, rel_type=None, rel_weight=1.0):
        """添加关系 (支持多种输入类型)"""
        source_node = self._get_node_from_input(source)
        target_node = self._get_node_from_input(target)
        
        if source_node and target_node:
            source_node.add_relation(target_node, rel_type, rel_weight)
            return True
        return False
        
    def _get_node_from_input(self, input_data):
        """从各种输入类型获取节点"""
        if isinstance(input_data, RNode):
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
            
    def remove_relationship(self, source, target):
        """移除两个节点间的关系"""
        source_node = self._get_node_from_input(source)
        target_node = self._get_node_from_input(target)
        
        if source_node and target_node:
            # 找到并移除关系
            for i, rel in enumerate(source_node.r):
                if rel['node'] == target_node:
                    source_node.r.pop(i)
                    return True
        return False
        
    def search_path(self, start, end, max_depth=10, rel_type=None):
        """查找从start到end的路径"""
        start_node = self._get_node_from_input(start)
        end_node = self._get_node_from_input(end)
        
        if not start_node or not end_node:
            return None
            
        # 重置所有节点的访问状态
        self._reset_visited()
        
        # 使用BFS查找路径
        queue = deque([(start_node, [start_node])])
        start_node.visited = True
        
        while queue:
            current, path = queue.popleft()
            
            if current == end_node:
                return path
                
            if len(path) >= max_depth:
                continue
                
            relations = current.get_relations(rel_type)
            for next_node in relations:
                if not next_node.visited:
                    next_node.visited = True
                    queue.append((next_node, path + [next_node]))
                    
        return None
        
    def _reset_visited(self):
        """重置所有节点的访问状态"""
        for bucket in self.buckets:
            for node in bucket:
                node.visited = False
                
    def find_related(self, node_input, rel_type=None, max_depth=2):
        """查找与节点相关的所有节点"""
        start_node = self._get_node_from_input(node_input)
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
        return list(result)
        
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
            
        return self.parallel_batch_process(
            node_data_list,
            lambda data: self.create_node(data['key'], data['val'], data.get('metadata'))
        )
        
    def batch_add_relationships(self, relationship_list):
        """批量添加关系"""
        return self.parallel_batch_process(
            relationship_list,
            lambda rel: self.add_relationship(rel['source'], rel['target'], 
                                            rel.get('type'), rel.get('weight', 1.0))
        )
        
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
        
    def get_current_load_factor(self):
        """获取当前负载因子"""
        return self.count / self.size
        
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
                
                # 保存旧桶
                old_buckets = self.buckets
                old_size = self.size
                
                # 创建新桶
                self.buckets = [[] for _ in range(new_size)]
                self.size = new_size
                
                # 更新锁数组大小
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
        
    def get_stats(self):
        """获取数据库状态统计"""
        bucket_loads = [len(bucket) for bucket in self.buckets]
        non_empty_buckets = sum(1 for load in bucket_loads if load > 0)
        
        return {
            'node_count': self.count,
            'bucket_count': self.size,
            'load_factor': self.get_current_load_factor(),
            'max_bucket_load': max(bucket_loads) if bucket_loads else 0,
            'min_bucket_load': min(bucket_loads) if bucket_loads else 0,
            'avg_bucket_load': self.count / self.size if self.size > 0 else 0,
            'non_empty_buckets': non_empty_buckets,
            'bucket_utilization': non_empty_buckets / self.size if self.size > 0 else 0
        }
        
    def __getstate__(self):
        """自定义序列化行为，处理不可序列化的对象"""
        state = self.__dict__.copy()
        
        # 移除所有锁对象和线程池
        del state['lock']
        del state['resizing_lock']
        del state['bucket_locks']
        del state['executor']
        
        # 确保节点中的OND引用不会造成循环引用
        for bucket in state['buckets']:
            for node in bucket:
                if hasattr(node, '_ond_reference'):
                    node._ond_reference = None
        
        return state
        
    def __setstate__(self, state):
        """自定义反序列化行为，重新创建不可序列化的对象"""
        self.__dict__.update(state)
        
        # 重新创建所有锁对象
        self.lock = threading.RLock()
        self.resizing_lock = threading.RLock()
        self.bucket_locks = [threading.RLock() for _ in range(self.size)]
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        # 重新设置节点的OND引用
        for bucket in self.buckets:
            for node in bucket:
                if hasattr(node, '_ond_reference'):
                    node._ond_reference = self
        
    def save_to_file(self, filename):
        """保存数据库到文件"""
        try:
            print("开始保存数据库...")
            with open(filename, 'wb') as f:
                # 使用协议2以兼容更多Python版本
                pickle.dump(self, f, protocol=2)
            print(f"成功保存到: {filename}")
            return True
        except Exception as e:
            print(f"保存失败: {e}")
            return False
            
    @classmethod
    def load_from_file(cls, filename, max_workers=4):
        """从文件加载数据库"""
        try:
            print(f"开始从 {filename} 加载数据库...")
            with open(filename, 'rb') as f:
                ond = pickle.load(f)
                
            # 确保执行器使用指定的max_workers
            if hasattr(ond, 'executor') and ond.executor:
                ond.executor.shutdown(wait=False)
                ond.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
                
            print("数据库加载成功")
            return ond
        except Exception as e:
            print(f"加载失败: {e}")
            return None
        
    def update_node_metadata(self, node, new_metadata):
        """更新节点的元数据并同步索引"""
        if not isinstance(new_metadata, dict):
            raise TypeError("元数据必须是字典类型")
            
        old_metadata = node.metadata.copy() if node.metadata else {}
        
        # 如果节点没有metadata属性，初始化它
        if node.metadata is None:
            node.metadata = {}
            
        # 获取旧元数据的键值对
        old_meta_pairs = set()
        if old_metadata:
            for key, value in old_metadata.items():
                old_meta_pairs.add((key, value))
        
        # 更新节点的元数据
        node.metadata.update(new_metadata)
        
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
            index_key = f"{meta_key}:{meta_val}"
            nodes = self.metadata_index.get(index_key, [])
            if node in nodes:
                nodes.remove(node)
        
        # 添加新的元数据索引
        for meta_key, meta_val in to_add:
            self._index_metadata(meta_key, meta_val, node)
            
        return node
        
    def find_nodes_by_metadata_kv(self, metadata_dict):
        """根据多个元数据键值对查找节点（AND条件）"""
        if not metadata_dict:
            return []
            
        result_sets = []
        for meta_key, meta_val in metadata_dict.items():
            nodes = self.find_by_metadata(meta_key, meta_val)
            if not nodes:
                return []  # 如果任一条件无匹配，则返回空
            result_sets.append(set(nodes))
            
        # 取交集
        if result_sets:
            return list(set.intersection(*result_sets))
        return []
        
    def _index_tag(self, tag, node):
        """为标签创建索引"""
        nodes = self.tag_index.get(tag)
        if nodes is None:
            self.tag_index[tag] = [node]
        else:
            if node not in nodes:
                nodes.append(node)
                
    def _remove_tag_index(self, tag, node):
        """移除标签索引"""
        nodes = self.tag_index.get(tag)
        if nodes and node in nodes:
            nodes.remove(node)
            
    def find_by_tag(self, tag):
        """通过标签查找节点"""
        return self.tag_index.get(tag, [])
    
    def _remove_metadata_index(self, meta_key, meta_val, node):
        """移除元数据索引"""
        key = f"{meta_key}:{meta_val}"
        nodes = self.metadata_index.get(key)
        if nodes and node in nodes:
            nodes.remove(node)
            
    def update_node_key(self, node, new_key):
        """更新节点的键，并同步相关索引"""
        old_key = node._key
        
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
        self.bimap.remove_key(old_key)
        self.bimap.add(new_key, val)
        
        # 添加到新位置
        new_index = self.hf(new_key)
        with self.bucket_locks[new_index]:
            self.buckets[new_index].append(node)
            
        return node
        
    def update_node_value(self, node, new_val):
        """更新节点的值，并同步相关索引"""
        old_val = node._val
        
        # 更新双向映射
        key = node.key
        self.bimap.remove_key(key)
        self.bimap.add(key, new_val)
        
        # 更新值类型索引
        old_type = type(old_val).__name__
        new_type = type(new_val).__name__
        
        if old_type != new_type:
            # 移除旧类型索引
            type_nodes = self.value_type_index.get(old_type, [])
            if node in type_nodes:
                type_nodes.remove(node)
                
            # 添加新类型索引
            new_type_nodes = self.value_type_index.get(new_type)
            if new_type_nodes is None:
                self.value_type_index[new_type] = [node]
            else:
                if node not in new_type_nodes:
                    new_type_nodes.append(node)
        
        # 更新节点值
        node._val = new_val
        
        return node
        
    def update_node_relations(self, node, new_relations):
        """更新节点的关系列表"""
        # 简单地替换关系列表
        node._r = new_relations or []
        return node
        
    def find_by_value_type(self, type_name):
        """通过值的类型查找节点"""
        return self.value_type_index.get(type_name, [])
        
    def get_node_tags(self):
        """获取所有标签"""
        return self.tag_index.keys()
        
    def get_nodes_with_relation_type(self, rel_type):
        """获取具有特定关系类型的所有节点"""
        result = []
        for bucket in self.buckets:
            for node in bucket:
                for rel in node.r:
                    if rel.get('type') == rel_type:
                        result.append(node)
                        break
        return result
        
    def batch_update_node_metadata(self, nodes, metadata_dict):
        """批量更新节点元数据"""
        return self.parallel_batch_process(
            nodes,
            lambda node: self.update_node_metadata(node, metadata_dict)
        )
        
    def batch_add_tags(self, nodes, tags):
        """批量添加标签"""
        def add_tags_to_node(node):
            for tag in tags:
                node.add_tag(tag)
            return node
            
        return self.parallel_batch_process(nodes, add_tags_to_node)
        
    def batch_update_node_weights(self, nodes, weight_value):
        """批量更新节点权重"""
        def update_weight(node):
            node.weight = weight_value
            return node
            
        return self.parallel_batch_process(nodes, update_weight)
        