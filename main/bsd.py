# 引入必要的库
from collections import deque
import heapq
import functools
import hashlib
import base64
import os
import hmac
import json
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# 增加加密所需的常量
ENCRYPTION_VERSION = 1
DEFAULT_ITERATIONS = 100000
KEY_LENGTH = 32
SALT_LENGTH = 16
NONCE_LENGTH = 12
TAG_LENGTH = 16

try:
    # 尝试导入高性能加密库
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    # 如果不可用，使用内置加密
    CRYPTOGRAPHY_AVAILABLE = False

# 哈希表类
class HashTable:
    def __init__(self, initial_capacity=16, load_factor=0.75):
        self.initial_capacity = initial_capacity
        self.load_factor = load_factor
        self.size = 0
        self.buckets = [[] for _ in range(initial_capacity)]

    def _hash_function(self, key):
        """增强的哈希函数：处理不同类型键，减少冲突"""
        h = hash(key)
        # 对哈希值进行扰动，避免高位数据丢失（针对整数/字符串等常见类型）
        h ^= (h >> 30) & 0x7FFFFFFF  # 混合高位和低位
        h ^= (h >> 15) & 0x7FFFFFFF
        return h % len(self.buckets)

    def _get_bucket(self, key):
        return self.buckets[self._hash_function(key)]

    def _find_key_in_bucket(self, bucket, key):
        for i, (k, _) in enumerate(bucket):
            if k == key:
                return i
        return -1

    def _resize(self, new_capacity):
        """扩容时重新哈希所有元素，减少冲突"""
        new_buckets = [[] for _ in range(new_capacity)]
        for bucket in self.buckets:
            for key, value in bucket:
                index = self._hash_function(key)  # 使用新容量计算索引
                new_buckets[index].append((key, value))
        self.buckets = new_buckets

    def insert(self, key, value):
        """严格插入：键存在时抛出异常（防止重复键）"""
        bucket = self._get_bucket(key)
        if self._find_key_in_bucket(bucket, key) != -1:
            raise ValueError(f"Key '{key}' already exists")
        bucket.append((key, value))
        self.size += 1
        # 触发扩容
        if self.size > self.load_factor * len(self.buckets):
            self._resize(len(self.buckets) * 2)

    def update(self, key, value):
        """严格更新：键不存在时抛出异常"""
        bucket = self._get_bucket(key)
        index = self._find_key_in_bucket(bucket, key)
        if index == -1:
            raise KeyError(f"Key '{key}' does not exist")
        bucket[index] = (key, value)

    def put(self, key, value):
        """兼容模式：存在则更新，不存在则插入（类似Python字典）"""
        bucket = self._get_bucket(key)
        index = self._find_key_in_bucket(bucket, key)
        if index != -1:
            bucket[index] = (key, value)  # 更新
        else:
            bucket.append((key, value))  # 插入
            self.size += 1
            if self.size > self.load_factor * len(self.buckets):
                self._resize(len(self.buckets) * 2)

    def get(self, key, default=None):
        """获取键对应的值，不存在时返回默认值"""
        bucket = self._get_bucket(key)
        index = self._find_key_in_bucket(bucket, key)
        return bucket[index][1] if index != -1 else default

    def delete(self, key):
        """删除指定键的键值对"""
        bucket = self._get_bucket(key)
        index = self._find_key_in_bucket(bucket, key)
        if index == -1:
            raise KeyError(f"Key '{key}' not found")

        del bucket[index]
        self.size -= 1
        # 可选：缩容逻辑（这里暂不实现，避免复杂度增加）

    def contains_key(self, key):
        """检查是否存在指定键"""
        return self._find_key_in_bucket(self._get_bucket(key), key) != -1

    def keys(self):
        """返回所有键的列表"""
        return [key for bucket in self.buckets for key, _ in bucket]

    def values(self):
        """返回所有值的列表"""
        return [value for bucket in self.buckets for _, value in bucket]

    def items(self):
        """返回所有键值对的列表"""
        return [(key, value) for bucket in self.buckets for key, value in bucket]

    def is_empty(self):
        """检查哈希表是否为空"""
        return self.size == 0

    def clear(self):
        """清空哈希表"""
        self.buckets = [[] for _ in range(self.initial_capacity)]
        self.size = 0

    def get_capacity(self):
        """获取当前哈希表容量（桶的数量）"""
        return len(self.buckets)

    def get_load_factor(self):
        """获取当前负载因子"""
        return self.load_factor

    def __len__(self):
        """返回存储的键值对数量"""
        return self.size

    def __getitem__(self, key):
        """支持通过索引访问值（h[key]）"""
        value = self.get(key)
        if value is None:
            raise KeyError(f"Key '{key}' not found")
        return value

    def __setitem__(self, key, value):
        """支持通过索引设置值（h[key] = value）"""
        self.put(key, value)

    def __delitem__(self, key):
        """支持通过del删除键（del h[key]）"""
        self.delete(key)

    def __contains__(self, key):
        """支持in操作符（key in h）"""
        return self.contains_key(key)

    def __iter__(self):
        """支持迭代键"""
        return iter(self.keys())

    def __str__(self):
        """返回哈希表的字符串表示"""
        return f"{{{', '.join(f'{k}: {v}' for k, v in self.items())}}}"

    def __repr__(self):
        """返回对象的正式表示"""
        return f"HashTable(capacity={len(self.buckets)}, size={self.size})"

    # 新增：检查键是否可哈希（防止不可哈希类型作为键）
    def _validate_key(self, key):
        try:
            hash(key)
        except TypeError:
            raise TypeError(f"Unsupported key type: {type(key).__name__}")


# 嵌套双向映射类
class NestedBidirectionalMap:
    def __init__(self, initial_data=None):
        self.forward = {}
        self.backward = {}
        self.nested_maps = {}

        if initial_data:
            if not isinstance(initial_data, dict):
                raise TypeError("初始数据必须是字典类型")
            for key, value in initial_data.items():
                self.add(key, value)

    def add(self, key, value):
        if key in self.forward:
            self.remove(key)

        self.forward[key] = value

        if isinstance(value, NestedBidirectionalMap):
            self.nested_maps[key] = value
            collected = self._collect_nested_values(value)
            for item in collected:
                if item not in self.backward:
                    self.backward[item] = set()
                self.backward[item].add(key)
        else:
            try:
                hash(value)
                is_hashable = True
            except TypeError:
                is_hashable = False

            if is_hashable:
                if value not in self.backward:
                    self.backward[value] = set()
                self.backward[value].add(key)

        return self

    def remove(self, key):
        if key not in self.forward:
            return None

        value = self.forward[key]
        del self.forward[key]

        if isinstance(value, NestedBidirectionalMap):
            if key in self.nested_maps:
                del self.nested_maps[key]
            collected = self._collect_nested_values(value)
            for item in collected:
                if item in self.backward:
                    self.backward[item].discard(key)
                    if not self.backward[item]:
                        del self.backward[item]
        else:
            if value in self.backward:
                self.backward[value].discard(key)
                if not self.backward[value]:
                    del self.backward[value]

        return value

    def update(self, key, value):
        if key not in self.forward:
            raise KeyError(f"键 '{key}' 不存在")
        self.add(key, value)
        return self

    def _collect_nested_values(self, nested_map):
        if not isinstance(nested_map, NestedBidirectionalMap):
            return {nested_map}

        result = set()
        visited = set()

        def collect_values(current_map):
            map_id = id(current_map)
            if map_id in visited:
                return
            visited.add(map_id)

            for k, v in current_map.forward.items():
                result.add(k)
                if isinstance(v, NestedBidirectionalMap):
                    collect_values(v)
                else:
                    result.add(v)

        collect_values(nested_map)
        return result

    def get_by_key(self, key, default=None):
        return self.forward.get(key, default)

    def get_by_value(self, value, default=None):
        return self.backward.get(value, default if default is not None else set())

    def has_key(self, key):
        return key in self.forward

    def has_value(self, value):
        if isinstance(value, NestedBidirectionalMap):
            return any(nested_map == value for nested_map in self.nested_maps.values())
        else:
            return value in self.backward

    def keys(self):
        return list(self.forward.keys())

    def values(self):
        return list(self.forward.values())

    def items(self):
        return list(self.forward.items())

    def clear(self):
        self.forward.clear()
        self.backward.clear()
        self.nested_maps.clear()
        return self

    def copy(self):
        new_map = NestedBidirectionalMap()
        for key, value in self.forward.items():
            new_map.add(key, value)
        return new_map

    def deep_copy(self):
        new_map = NestedBidirectionalMap()
        for key, value in self.forward.items():
            if isinstance(value, NestedBidirectionalMap):
                new_map.add(key, value.deep_copy())
            else:
                new_map.add(key, value)
        return new_map

    def merge(self, other):
        if not isinstance(other, NestedBidirectionalMap):
            raise TypeError("只能与NestedBidirectionalMap类型合并")
        for key, value in other.forward.items():
            self.add(key, value)
        return self

    def to_dict(self):
        result = {}
        for key, value in self.forward.items():
            if isinstance(value, NestedBidirectionalMap):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise TypeError("数据必须是字典类型")
        result = cls()
        for key, value in data.items():
            if isinstance(value, dict):
                result.add(key, cls.from_dict(value))
            else:
                result.add(key, value)
        return result

    def find_path_to_value(self, target_value):
        paths = []

        def dfs(current_map, current_path):
            for key, value in current_map.forward.items():
                new_path = current_path + [key]
                if value == target_value:
                    paths.append(new_path)
                elif isinstance(value, NestedBidirectionalMap):
                    dfs(value, new_path)

        dfs(self, [])
        return paths

    def get_nested_value(self, path):
        if not path:
            raise ValueError("路径不能为空")
        current = self
        for i, key in enumerate(path):
            if not isinstance(current, NestedBidirectionalMap):
                raise KeyError(f"路径 {path[:i]} 不指向嵌套映射")
            if key not in current.forward:
                raise KeyError(f"键 '{key}' 不存在于路径 {path[:i]} 的映射中")
            current = current.forward[key]
        return current

    def set_nested_value(self, path, value):
        if not path:
            raise ValueError("路径不能为空")
        if len(path) == 1:
            self.add(path[0], value)
            return self
        current = self
        for i, key in enumerate(path[:-1]):
            if key not in current.forward:
                raise KeyError(f"键 '{key}' 不存在于路径 {path[:i]} 的映射中")
            if not isinstance(current.forward[key], NestedBidirectionalMap):
                raise KeyError(f"路径 {path[:i+1]} 不指向嵌套映射")
            current = current.forward[key]
        current.add(path[-1], value)
        return self

    def __getitem__(self, key):
        if key not in self.forward:
            raise KeyError(f"键 '{key}' 不存在")
        return self.forward[key]

    def __setitem__(self, key, value):
        self.add(key, value)

    def __delitem__(self, key):
        if key not in self.forward:
            raise KeyError(f"键 '{key}' 不存在")
        self.remove(key)

    def __contains__(self, key):
        return key in self.forward

    def __iter__(self):
        return iter(self.forward)

    def __len__(self):
        return len(self.forward)

    def __str__(self):
        items = []
        for key, value in self.forward.items():
            if isinstance(value, NestedBidirectionalMap):
                items.append(f"{key}: {{{str(value)}}}")
            else:
                items.append(f"{key}: {value}")
        return ", ".join(items)

    def __repr__(self):
        return f"NestedBidirectionalMap({self.to_dict()})"

    def __eq__(self, other):
        if not isinstance(other, NestedBidirectionalMap):
            return False
        return self.forward == other.forward

    def __bool__(self):
        return bool(self.forward)


# 关系链类（使用图的邻接表表示）
class RelationshipChain:
    def __init__(self):
        self.graph = {}

    def add_relationship(self, node1, node2):
        if node1 not in self.graph:
            self.graph[node1] = set()
        if node2 not in self.graph:
            self.graph[node2] = set()
        self.graph[node1].add(node2)
        self.graph[node2].add(node1)

    def remove_relationship(self, node1, node2):
        if node1 in self.graph and node2 in self.graph[node1]:
            self.graph[node1].remove(node2)
        if node2 in self.graph and node1 in self.graph[node2]:
            self.graph[node2].remove(node1)

    def get_related_nodes(self, node):
        return self.graph.get(node, set())

    def has_relationship(self, node1, node2):
        return node1 in self.graph and node2 in self.graph[node1]

    def find_path(self, start, end):
        visited = set()
        queue = deque([(start, [start])])

        while queue:
            current, path = queue.popleft()
            if current == end:
                return path
            if current not in visited:
                visited.add(current)
                for neighbor in self.graph.get(current, []):
                    if neighbor not in visited:
                        new_path = path + [neighbor]
                        queue.append((neighbor, new_path))
        return None


# 结合哈希表、嵌套双向映射和关系链的类
class CombinedDataStructure(HashTable, NestedBidirectionalMap, RelationshipChain):
    def __init__(self, initial_capacity=16, load_factor=0.75, initial_data=None):
        HashTable.__init__(self, initial_capacity, load_factor)
        NestedBidirectionalMap.__init__(self, initial_data)
        RelationshipChain.__init__(self)
        self.metadata = {}  # 存储节点和关系的元数据

    def put(self, key, value, create_relationship=False):
        """增强版put方法，可选择是否同时创建关系"""
        # 调用哈希表的put方法
        HashTable.put(self, key, value)
        # 调用嵌套双向映射的add方法
        NestedBidirectionalMap.add(self, key, value)
        
        # 如果指定了创建关系，自动在key和value之间建立关系
        if create_relationship and not isinstance(value, NestedBidirectionalMap):
            try:
                hash(value)  # 检查值是否可哈希
                self.add_relationship(key, value)
            except TypeError:
                pass  # 不可哈希的值不建立关系
        
        return self

    def get(self, key, default=None, include_related=False):
        """增强版get方法，可选择是否返回关联节点"""
        # 优先从哈希表中获取值
        value = HashTable.get(self, key, default)
        if value is default:
            # 如果哈希表中没有，从嵌套双向映射中获取
            value = NestedBidirectionalMap.get_by_key(self, key, default)
        
        # 如果需要包含关联节点
        if include_related and key in self.graph:
            related = self.get_related_nodes(key)
            return {"value": value, "related": related}
            
        return value

    def delete(self, key, keep_relationships=False):
        """增强版delete方法，可选择是否保留关系"""
        # 从哈希表中删除键值对
        HashTable.delete(self, key)
        # 从嵌套双向映射中删除键值对
        NestedBidirectionalMap.remove(self, key)
        
        # 处理关系链
        if not keep_relationships and key in self.graph:
            related_nodes = list(self.graph[key])
            for related_node in related_nodes:
                self.remove_relationship(key, related_node)
            del self.graph[key]
            
        # 清除元数据
        if key in self.metadata:
            del self.metadata[key]
            
        return self

    # 增强关系功能
    def add_relationship_with_metadata(self, node1, node2, metadata=None):
        """创建带有元数据的关系"""
        self.add_relationship(node1, node2)
        rel_key = f"{node1}:{node2}"
        if metadata:
            self.metadata[rel_key] = metadata
        return self

    def get_relationship_metadata(self, node1, node2):
        """获取关系的元数据"""
        rel_key = f"{node1}:{node2}"
        return self.metadata.get(rel_key)

    def set_node_metadata(self, node, metadata):
        """设置节点的元数据"""
        self.metadata[node] = metadata
        return self

    def get_node_metadata(self, node):
        """获取节点的元数据"""
        return self.metadata.get(node)

    # 综合查询方法
    def query_by_value(self, value):
        """通过值查询所有相关信息"""
        # 从双向映射获取映射到此值的所有键
        keys = NestedBidirectionalMap.get_by_value(self, value)
        
        # 从关系链中获取与值直接相关的节点
        related = set()
        if value in self.graph:
            related = self.get_related_nodes(value)
            
        # 返回综合信息
        return {
            "keys": keys,
            "related_nodes": related,
            "paths": [self.find_path(k, value) for k in keys if k != value]
        }

    def find_connected_components(self):
        """查找图中的所有连通分量"""
        visited = set()
        components = []
        
        for node in self.graph:
            if node not in visited:
                component = set()
                queue = deque([node])
                
                while queue:
                    current = queue.popleft()
                    if current not in visited:
                        visited.add(current)
                        component.add(current)
                        for neighbor in self.graph.get(current, []):
                            if neighbor not in visited:
                                queue.append(neighbor)
                
                components.append(component)
                
        return components

    def find_common_connections(self, node1, node2):
        """查找两个节点的共同连接"""
        if node1 not in self.graph or node2 not in self.graph:
            return set()
        return self.graph[node1].intersection(self.graph[node2])
    
    def find_shortest_paths(self, start, end, limit=10):
        """查找从起点到终点的多条最短路径（限制数量）"""
        if start not in self.graph or end not in self.graph:
            return []
            
        # 使用BFS查找最短路径
        visited = {start: None}
        queue = deque([(start, [start])])
        paths = []
        min_length = float('inf')
        
        while queue and len(paths) < limit:
            current, path = queue.popleft()
            
            # 如果已经找到更短的路径，不再考虑更长的路径
            if len(path) > min_length:
                break
                
            if current == end:
                paths.append(path)
                min_length = len(path)
            else:
                for neighbor in self.graph.get(current, []):
                    if neighbor not in visited or len(path) <= len(visited[neighbor]):
                        new_path = path + [neighbor]
                        queue.append((neighbor, new_path))
                        visited[neighbor] = path
        
        return paths
    
    def get_nested_path_value(self, path):
        """获取嵌套路径的值，同时考虑关系"""
        try:
            # 先尝试从嵌套双向映射获取
            return self.get_nested_value(path)
        except (KeyError, ValueError):
            # 如果失败，尝试通过关系链查找路径
            if len(path) >= 2:
                found_path = self.find_path(path[0], path[-1])
                if found_path:
                    return found_path
            return None
    
    def recursive_search(self, value, max_depth=3):
        """递归搜索与值相关的所有节点，限制深度"""
        visited = set()
        result = {}
        
        def dfs(node, depth=0):
            if depth > max_depth or node in visited:
                return
            
            visited.add(node)
            
            # 添加当前节点信息
            if node not in result:
                result[node] = {
                    "direct_value": HashTable.get(self, node, None),
                    "related": set()
                }
            
            # 添加关系信息
            for related in self.get_related_nodes(node):
                result[node]["related"].add(related)
                dfs(related, depth + 1)
        
        # 从值开始搜索
        keys = NestedBidirectionalMap.get_by_value(self, value)
        if keys:
            for key in keys:
                dfs(key)
        
        # 如果值本身是节点，也从它开始搜索
        if value in self.graph:
            dfs(value)
            
        return result

    def __str__(self):
        hash_count = self.size
        bidirectional_count = len(self.forward)
        relationship_count = sum(len(connections) for connections in self.graph.values()) // 2
        
        return (f"CombinedDataStructure(keys={hash_count}, "
                f"bidirectional_pairs={bidirectional_count}, "
                f"relationships={relationship_count})")

    def __repr__(self):
        return f"CombinedDataStructure(capacity={self.get_capacity()}, size={self.size}, relationships={len(self.graph)})"

    def get_by_multiple_keys(self, keys, mode="any", include_related=False):
        """通过多个键查找值和关系
        
        参数:
            keys: 要查找的键的列表或集合
            mode: 查找模式，可选值:
                  - "any": 返回任意键匹配的结果（并集）
                  - "all": 仅返回所有键都匹配的结果（交集）
                  - "each": 返回每个键对应的单独结果
            include_related: 是否包含关联节点
            
        返回:
            根据mode不同返回不同格式的结果:
            - "any"模式: {key: value, ...} 所有匹配键的字典
            - "all"模式: 匹配所有键的共同值的集合
            - "each"模式: {key: {value, related_nodes}, ...} 每个键的结果字典
        """
        if not keys:
            return {} if mode in ["any", "each"] else set()
            
        if mode == "any":
            # 返回任意键匹配的结果（并集）
            result = {}
            for key in keys:
                if self.contains_key(key):
                    value = self.get(key, include_related=include_related)
                    result[key] = value
            return result
            
        elif mode == "all":
            # 返回所有键都匹配的共同值（交集）
            # 先获取第一个键的值
            if not keys:
                return set()
                
            first_key = next(iter(keys))
            if not self.contains_key(first_key):
                return set()
                
            # 获取第一个键的值（可能是直接值或关联节点）
            common_values = set()
            first_result = self.get(first_key, include_related=False)
            
            # 单个值加入集合
            if first_result is not None:
                common_values.add(first_result)
                
            # 如果需要包含关联节点
            if include_related and first_key in self.graph:
                common_values.update(self.get_related_nodes(first_key))
                
            # 求交集
            for key in keys:
                if key == first_key:
                    continue
                    
                if not self.contains_key(key):
                    return set()  # 有键不存在，交集为空
                    
                key_values = set()
                value = self.get(key, include_related=False)
                
                if value is not None:
                    key_values.add(value)
                    
                if include_related and key in self.graph:
                    key_values.update(self.get_related_nodes(key))
                    
                # 求交集
                common_values &= key_values
                
                if not common_values:
                    break  # 交集已为空，提前结束
                    
            return common_values
            
        elif mode == "each":
            # 返回每个键的单独结果
            result = {}
            for key in keys:
                if self.contains_key(key):
                    value = self.get(key, include_related=False)
                    related = set()
                    
                    if include_related and key in self.graph:
                        related = self.get_related_nodes(key)
                        
                    result[key] = {"value": value, "related": related}
                    
            return result
            
        else:
            raise ValueError(f"不支持的模式: {mode}，请使用 'any', 'all' 或 'each'")
            
    def search(self, query, search_in=None, max_results=None, fuzzy=False, case_sensitive=False):
        """全局搜索函数，支持多种搜索模式
        
        参数:
            query: 搜索查询（可以是字符串或任何值）
            search_in: 搜索范围列表，可选值: "keys", "values", "metadata", "relationships"
                      默认为None表示所有范围
            max_results: 最大结果数量
            fuzzy: 是否进行模糊匹配（字符串时有效）
            case_sensitive: 是否区分大小写（字符串时有效）
            
        返回:
            {
                "keys": [匹配的键列表],
                "values": [匹配的值列表],
                "relationships": [匹配的关系列表],
                "metadata": [匹配的元数据列表]
            }
        """
        if search_in is None:
            # 默认搜索所有范围
            search_in = ["keys", "values", "metadata", "relationships"]
            
        # 规范化搜索查询（如果是字符串）
        if isinstance(query, str) and not case_sensitive:
            str_query = query.lower()
        else:
            str_query = query
            
        results = {
            "keys": [],
            "values": [],
            "relationships": [],
            "metadata": []
        }
        
        # 字符串模糊匹配函数
        def fuzzy_match(target, q):
            if not isinstance(target, str) or not isinstance(q, str):
                return False
                
            if not case_sensitive:
                target = target.lower()
                
            if fuzzy:
                # 简单的模糊匹配: 检查查询中的每个字符是否按顺序出现在目标中
                j = 0
                for c in q:
                    j = target.find(c, j)
                    if j == -1:
                        return False
                    j += 1
                return True
            else:
                # 精确子串匹配
                return q in target
        
        # 值匹配函数
        def value_match(target, q):
            if isinstance(target, str) and isinstance(q, str):
                return fuzzy_match(target, q)
            else:
                try:
                    return target == q
                except:
                    return False
        
        # 搜索键
        if "keys" in search_in:
            for key in self.keys():
                if isinstance(key, str):
                    if isinstance(query, str) and fuzzy_match(key, str_query):
                        results["keys"].append(key)
                        if max_results and len(results["keys"]) >= max_results:
                            break
                elif key == query:
                    results["keys"].append(key)
                    if max_results and len(results["keys"]) >= max_results:
                        break
        
        # 搜索值
        if "values" in search_in:
            for key, value in self.items():
                if value_match(value, str_query) or value == query:
                    results["values"].append((key, value))
                    if max_results and len(results["values"]) >= max_results:
                        break
                        
            # 从双向映射中搜索
            if isinstance(query, (str, int, float, bool)) and query in self.backward:
                for key in self.backward[query]:
                    if (key, query) not in results["values"]:
                        results["values"].append((key, query))
                        if max_results and len(results["values"]) >= max_results:
                            break
        
        # 搜索关系
        if "relationships" in search_in:
            # 寻找关系中包含查询的节点
            for node, connections in self.graph.items():
                if node == query or (isinstance(node, str) and isinstance(query, str) and fuzzy_match(node, str_query)):
                    for connected in connections:
                        results["relationships"].append((node, connected))
                        if max_results and len(results["relationships"]) >= max_results:
                            break
                            
                for connected in connections:
                    if connected == query or (isinstance(connected, str) and isinstance(query, str) and fuzzy_match(connected, str_query)):
                        if (node, connected) not in results["relationships"]:
                            results["relationships"].append((node, connected))
                            if max_results and len(results["relationships"]) >= max_results:
                                break
        
        # 搜索元数据
        if "metadata" in search_in:
            for key, meta in self.metadata.items():
                if isinstance(meta, dict):
                    # 搜索字典中的键和值
                    for meta_key, meta_value in meta.items():
                        if meta_key == query or meta_value == query:
                            results["metadata"].append((key, meta))
                            break
                        elif isinstance(meta_key, str) and isinstance(query, str) and fuzzy_match(meta_key, str_query):
                            results["metadata"].append((key, meta))
                            break
                        elif isinstance(meta_value, str) and isinstance(query, str) and fuzzy_match(meta_value, str_query):
                            results["metadata"].append((key, meta))
                            break
                            
                elif meta == query:
                    results["metadata"].append((key, meta))
                elif isinstance(meta, str) and isinstance(query, str) and fuzzy_match(meta, str_query):
                    results["metadata"].append((key, meta))
                    
                if max_results and len(results["metadata"]) >= max_results:
                    break
                    
        return results

# 加密增强版CombinedDataStructure
class EncryptedCombinedDataStructure(CombinedDataStructure):
    def __init__(self, password=None, initial_capacity=16, load_factor=0.75, initial_data=None):
        super().__init__(initial_capacity, load_factor, initial_data)
        self.encryption_enabled = False
        self.encryption_key = None
        self.salt = None
        
        # 如果提供了密码，则启用加密
        if password:
            self.enable_encryption(password)
        
        # 加密相关元数据
        self.encrypted_keys = set()  # 记录哪些键是加密的
        self.key_aliases = {}  # 键别名映射，用于支持加密键的搜索

    def _derive_key(self, password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """从密码派生密钥"""
        if salt is None:
            salt = os.urandom(SALT_LENGTH)
            
        if CRYPTOGRAPHY_AVAILABLE:
            # 使用cryptography库的PBKDF2
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=KEY_LENGTH,
                salt=salt,
                iterations=DEFAULT_ITERATIONS
            )
            key = kdf.derive(password.encode('utf-8'))
        else:
            # 使用hashlib的PBKDF2
            key = hashlib.pbkdf2_hmac(
                'sha256', 
                password.encode('utf-8'), 
                salt, 
                DEFAULT_ITERATIONS, 
                dklen=KEY_LENGTH
            )
            
        return key, salt

    def enable_encryption(self, password: str) -> None:
        """启用加密功能"""
        self.encryption_key, self.salt = self._derive_key(password)
        self.encryption_enabled = True
        
    def disable_encryption(self) -> None:
        """禁用加密功能"""
        # 如果有加密数据，先解密
        if self.encrypted_keys:
            raise ValueError("数据结构包含加密数据，请先解密所有数据")
        
        self.encryption_enabled = False
        self.encryption_key = None
        self.salt = None
        self.encrypted_keys.clear()
        self.key_aliases.clear()

    def change_password(self, old_password: str, new_password: str) -> None:
        """更改加密密码"""
        # 验证旧密码
        old_key, _ = self._derive_key(old_password, self.salt)
        if not hmac.compare_digest(old_key, self.encryption_key):
            raise ValueError("密码不正确")
            
        # 解密所有数据
        encrypted_items = [(k, self.get(k)) for k in self.encrypted_keys.copy()]
        for key, value in encrypted_items:
            decrypted_value = self._decrypt_value(value)
            super().put(key, decrypted_value)
            self.encrypted_keys.remove(key)
            
        # 派生新密钥
        self.encryption_key, self.salt = self._derive_key(new_password)
        
        # 重新加密数据
        for key, value in encrypted_items:
            self.put_encrypted(key, value)

    def _encrypt_value(self, value: Any) -> str:
        """加密值"""
        if not self.encryption_enabled:
            raise ValueError("加密未启用")
            
        # 序列化值
        value_bytes = json.dumps(value).encode('utf-8')
        
        if CRYPTOGRAPHY_AVAILABLE:
            # 使用AES-GCM加密
            aesgcm = AESGCM(self.encryption_key)
            nonce = os.urandom(NONCE_LENGTH)
            ciphertext = aesgcm.encrypt(nonce, value_bytes, None)
            result = nonce + ciphertext
        else:
            # 基本加密方案（不安全，仅作示例）
            # 实际应用中应使用专业加密库
            key_hash = hashlib.sha256(self.encryption_key).digest()
            nonce = os.urandom(NONCE_LENGTH)
            
            # 简单异或加密（不安全）
            ciphertext = bytearray(len(value_bytes))
            for i in range(len(value_bytes)):
                key_byte = key_hash[i % len(key_hash)]
                nonce_byte = nonce[i % len(nonce)]
                ciphertext[i] = value_bytes[i] ^ key_byte ^ nonce_byte
                
            result = nonce + bytes(ciphertext)
            
        # Base64编码以便存储
        return base64.b64encode(result).decode('utf-8')

    def _decrypt_value(self, encrypted_value: str) -> Any:
        """解密值"""
        if not self.encryption_enabled:
            raise ValueError("加密未启用")
            
        # 解码Base64
        encrypted_bytes = base64.b64decode(encrypted_value.encode('utf-8'))
        
        if CRYPTOGRAPHY_AVAILABLE:
            # 使用AES-GCM解密
            nonce = encrypted_bytes[:NONCE_LENGTH]
            ciphertext = encrypted_bytes[NONCE_LENGTH:]
            
            aesgcm = AESGCM(self.encryption_key)
            plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        else:
            # 基本解密方案（与上面的加密对应）
            nonce = encrypted_bytes[:NONCE_LENGTH]
            ciphertext = encrypted_bytes[NONCE_LENGTH:]
            
            key_hash = hashlib.sha256(self.encryption_key).digest()
            plaintext = bytearray(len(ciphertext))
            
            for i in range(len(ciphertext)):
                key_byte = key_hash[i % len(key_hash)]
                nonce_byte = nonce[i % len(nonce)]
                plaintext[i] = ciphertext[i] ^ key_byte ^ nonce_byte
                
            plaintext = bytes(plaintext)
            
        # 反序列化
        return json.loads(plaintext.decode('utf-8'))

    def _hash_key(self, key: Any) -> str:
        """哈希键以生成别名"""
        key_str = str(key)
        return hashlib.sha256(key_str.encode('utf-8')).hexdigest()

    def put_encrypted(self, key: Any, value: Any) -> 'EncryptedCombinedDataStructure':
        """添加加密的键值对"""
        if not self.encryption_enabled:
            raise ValueError("加密未启用")
            
        # 加密值
        encrypted_value = self._encrypt_value(value)
        
        # 如果键是字符串，创建键的哈希别名用于搜索
        if isinstance(key, (str, int, float, bool)):
            key_alias = self._hash_key(key)
            self.key_aliases[key_alias] = key
            
        # 存储加密值
        super().put(key, encrypted_value)
        self.encrypted_keys.add(key)
        
        return self

    def get_encrypted(self, key: Any, default=None, decrypt=True) -> Any:
        """获取加密的值并自动解密"""
        if not self.encryption_enabled:
            raise ValueError("加密未启用")
            
        if key not in self.encrypted_keys:
            return default
            
        encrypted_value = super().get(key, default)
        
        if encrypted_value is default:
            return default
        
        if decrypt:
            try:
                return self._decrypt_value(encrypted_value)
            except Exception as e:
                raise ValueError(f"解密失败: {e}")
        else:
            return encrypted_value

    def put(self, key: Any, value: Any, create_relationship=False, encrypt=False) -> 'EncryptedCombinedDataStructure':
        """增强版put方法，支持加密选项"""
        if encrypt and self.encryption_enabled:
            return self.put_encrypted(key, value)
        else:
            # 如果键之前是加密的，移除加密标记
            if key in self.encrypted_keys:
                self.encrypted_keys.remove(key)
            
            return super().put(key, value, create_relationship)

    def get(self, key: Any, default=None, include_related=False, decrypt=True) -> Any:
        """增强版get方法，支持解密选项"""
        if key in self.encrypted_keys and self.encryption_enabled:
            value = self.get_encrypted(key, default, decrypt)
            
            # 如果需要包含关联节点且有解密值
            if include_related and key in self.graph and decrypt:
                related = self.get_related_nodes(key)
                return {"value": value, "related": related}
                
            return value
        else:
            return super().get(key, default, include_related)

    def search_by_encrypted_key(self, search_term: str) -> List[Any]:
        """搜索加密的键（基于关键词）"""
        if not self.encryption_enabled:
            raise ValueError("加密未启用")
            
        results = []
        search_term_lower = search_term.lower()
        
        # 搜索别名
        for key_alias, original_key in self.key_aliases.items():
            if search_term_lower in str(original_key).lower():
                results.append(original_key)
                
        return results

    def encrypt_existing_data(self, keys: Optional[List[Any]] = None) -> None:
        """加密已存在的数据"""
        if not self.encryption_enabled:
            raise ValueError("加密未启用")
            
        # 确定要加密的键
        if keys is None:
            # 加密所有非加密数据
            keys_to_encrypt = [k for k in self.keys() if k not in self.encrypted_keys]
        else:
            # 加密指定的键
            keys_to_encrypt = [k for k in keys if k in self.keys() and k not in self.encrypted_keys]
            
        # 加密数据
        for key in keys_to_encrypt:
            value = super().get(key)
            if value is not None:
                self.put_encrypted(key, value)

    def decrypt_all_data(self) -> None:
        """解密所有加密数据"""
        if not self.encryption_enabled:
            raise ValueError("加密未启用")
            
        # 解密所有加密数据
        encrypted_keys = list(self.encrypted_keys)
        for key in encrypted_keys:
            encrypted_value = super().get(key)
            if encrypted_value is not None:
                try:
                    decrypted_value = self._decrypt_value(encrypted_value)
                    super().put(key, decrypted_value)
                    self.encrypted_keys.remove(key)
                except Exception as e:
                    print(f"解密键 '{key}' 失败: {e}")

    def export_encrypted_data(self, include_metadata=True) -> Dict[str, Any]:
        """导出加密数据（包括元数据）"""
        if not self.encryption_enabled:
            raise ValueError("加密未启用")
            
        data = {
            "version": ENCRYPTION_VERSION,
            "encrypted_data": {},
            "salt": base64.b64encode(self.salt).decode('utf-8')
        }
        
        # 添加加密数据
        for key in self.encrypted_keys:
            encrypted_value = super().get(key)
            if encrypted_value is not None:
                # 将键序列化为字符串（因为JSON不支持任意类型的键）
                key_str = str(key)
                data["encrypted_data"][key_str] = encrypted_value
                
        # 添加元数据（如果需要）
        if include_metadata:
            data["metadata"] = {}
            for key in self.encrypted_keys:
                if key in self.metadata:
                    # 元数据也需要加密
                    data["metadata"][str(key)] = self._encrypt_value(self.metadata[key])
                    
            # 添加关系数据
            data["relationships"] = {}
            for node in self.encrypted_keys:
                if node in self.graph:
                    # 只包含与加密节点的关系
                    encrypted_relations = [n for n in self.graph[node] if n in self.encrypted_keys]
                    if encrypted_relations:
                        data["relationships"][str(node)] = [str(n) for n in encrypted_relations]
        
        return data

    def import_encrypted_data(self, data: Dict[str, Any], password: str, merge=True) -> None:
        """导入加密数据"""
        if not self.encryption_enabled and password:
            # 如果未启用加密但提供了密码，则启用加密
            self.enable_encryption(password)
        elif not self.encryption_enabled:
            raise ValueError("加密未启用且未提供密码")
            
        # 验证版本
        version = data.get("version")
        if version != ENCRYPTION_VERSION:
            raise ValueError(f"不支持的加密版本: {version}")
            
        # 检查是否提供了盐
        salt_b64 = data.get("salt")
        if not salt_b64:
            raise ValueError("数据缺少盐值")
            
        salt = base64.b64decode(salt_b64.encode('utf-8'))
        
        # 使用提供的盐和密码派生密钥
        key, _ = self._derive_key(password, salt)
        
        # 临时保存当前密钥状态
        old_key, old_salt = self.encryption_key, self.salt
        self.encryption_key, self.salt = key, salt
        
        # 导入加密数据
        try:
            encrypted_data = data.get("encrypted_data", {})
            for key_str, encrypted_value in encrypted_data.items():
                # 尝试将键转回原始类型（基本类型）
                try:
                    # 尝试作为数字解析
                    if key_str.isdigit():
                        key = int(key_str)
                    elif key_str.replace('.', '', 1).isdigit() and key_str.count('.') <= 1:
                        key = float(key_str)
                    elif key_str.lower() in ('true', 'false'):
                        key = key_str.lower() == 'true'
                    else:
                        key = key_str
                except:
                    key = key_str
                
                if merge or key not in self.keys():
                    super().put(key, encrypted_value)
                    self.encrypted_keys.add(key)
                    
                    # 为键创建别名
                    if isinstance(key, (str, int, float, bool)):
                        key_alias = self._hash_key(key)
                        self.key_aliases[key_alias] = key
            
            # 导入元数据
            if "metadata" in data:
                for key_str, encrypted_metadata in data["metadata"].items():
                    try:
                        if key_str.isdigit():
                            key = int(key_str)
                        elif key_str.replace('.', '', 1).isdigit() and key_str.count('.') <= 1:
                            key = float(key_str)
                        else:
                            key = key_str
                    except:
                        key = key_str
                        
                    if key in self.encrypted_keys:
                        try:
                            metadata = self._decrypt_value(encrypted_metadata)
                            self.metadata[key] = metadata
                        except:
                            pass
                            
            # 导入关系
            if "relationships" in data:
                for node_str, relations in data["relationships"].items():
                    try:
                        if node_str.isdigit():
                            node = int(node_str)
                        elif node_str.replace('.', '', 1).isdigit() and node_str.count('.') <= 1:
                            node = float(node_str)
                        else:
                            node = node_str
                    except:
                        node = node_str
                        
                    if node in self.encrypted_keys:
                        for rel_str in relations:
                            try:
                                if rel_str.isdigit():
                                    rel = int(rel_str)
                                elif rel_str.replace('.', '', 1).isdigit() and rel_str.count('.') <= 1:
                                    rel = float(rel_str)
                                else:
                                    rel = rel_str
                            except:
                                rel = rel_str
                                
                            if rel in self.encrypted_keys:
                                self.add_relationship(node, rel)
                                
        except Exception as e:
            # 恢复原始密钥状态
            self.encryption_key, self.salt = old_key, old_salt
            raise ValueError(f"导入加密数据失败: {e}")
            
        # 如果密码不同，使用新密码
        if not hmac.compare_digest(old_key, key):
            self.encryption_key, self.salt = key, salt

    def get_encryption_info(self) -> Dict[str, Any]:
        """获取加密相关信息"""
        if not self.encryption_enabled:
            return {"encryption_enabled": False}
            
        return {
            "encryption_enabled": True,
            "encrypted_keys_count": len(self.encrypted_keys),
            "encryption_version": ENCRYPTION_VERSION,
            "using_cryptography_lib": CRYPTOGRAPHY_AVAILABLE
        }

    # 批量加密/解密
    def batch_encrypt(self, keys_values: Dict[Any, Any]) -> None:
        """批量加密多个键值对"""
        if not self.encryption_enabled:
            raise ValueError("加密未启用")
            
        for key, value in keys_values.items():
            self.put_encrypted(key, value)

    def batch_decrypt(self, keys: List[Any]) -> Dict[Any, Any]:
        """批量解密多个键，返回解密后的键值对"""
        if not self.encryption_enabled:
            raise ValueError("加密未启用")
            
        result = {}
        for key in keys:
            if key in self.encrypted_keys:
                try:
                    result[key] = self.get_encrypted(key)
                except Exception as e:
                    print(f"解密键 '{key}' 失败: {e}")
                    
        return result

    # 哈希盐旋转（安全实践）
    def rotate_salt(self, password: str) -> None:
        """旋转加密盐（增强安全性）"""
        if not self.encryption_enabled:
            raise ValueError("加密未启用")
            
        # 验证密码
        old_key, _ = self._derive_key(password, self.salt)
        if not hmac.compare_digest(old_key, self.encryption_key):
            raise ValueError("密码不正确")
            
        # 重新加密所有数据
        all_encrypted_items = [(k, self.get_encrypted(k)) for k in self.encrypted_keys.copy()]
        
        # 生成新的盐和密钥
        new_key, new_salt = self._derive_key(password)
        self.encryption_key, self.salt = new_key, new_salt
        
        # 用新密钥重新加密所有数据
        self.encrypted_keys.clear()
        for key, value in all_encrypted_items:
            self.put_encrypted(key, value)

# 使用组合模式的改进版数据结构
class ICombinedDataStructure:
    """
    改进版组合数据结构，使用组合而非继承，解决多重继承冲突，
    增强错误处理、搜索性能、元数据清理和循环检测。
    """
    
    class OperationError(Exception):
        """操作错误基类"""
        pass
        
    class KeyError(OperationError):
        """键不存在错误"""
        pass
        
    class ValueError(OperationError):
        """值错误"""
        pass
        
    class RelationshipError(OperationError):
        """关系操作错误"""
        pass
        
    class SearchError(OperationError):
        """搜索错误"""
        pass
    
    def __init__(self, initial_capacity=16, load_factor=0.75, initial_data=None):
        """初始化组合数据结构"""
        # 使用组合而非继承
        self._hash_table = HashTable(initial_capacity, load_factor)
        self._bidir_map = NestedBidirectionalMap(initial_data)
        self._rel_chain = RelationshipChain()
        
        # 元数据存储
        self._metadata = {}
        
        # 关系元数据使用更精确的结构
        self._relationship_metadata = {}
        
        # 缓存和索引结构
        self._value_index = {}  # 值到键的反向索引
        self._search_index = {}  # 搜索索引
        
        # 如果提供了初始数据，填充哈希表
        if initial_data:
            for key, value in initial_data.items():
                self.put(key, value)
    
    def put(self, key, value, create_relationship=False, metadata=None):
        """
        存储键值对，可选择是否创建关系和元数据
        
        参数:
            key: 键
            value: 值
            create_relationship: 是否在键和值之间创建关系（如果值是可哈希的）
            metadata: 与键关联的元数据
            
        返回:
            self 实例（链式调用）
            
        异常:
            ValueError: 如果键或值不可哈希
        """
        try:
            # 校验键的可哈希性
            hash(key)
        except Exception as e:
            raise self.ValueError(f"键不可哈希: {e}")
        
        # 存储值
        self._hash_table.put(key, value)
        self._bidir_map.add(key, value)
        
        # 更新反向索引
        try:
            hash(value)
            if value not in self._value_index:
                self._value_index[value] = set()
            self._value_index[value].add(key)
        except:
            # 值不可哈希，不添加到索引
            pass
        
        # 如果指定了创建关系，且值可哈希且不是复杂的嵌套结构
        if create_relationship:
            try:
                hash(value)
                if not isinstance(value, NestedBidirectionalMap):
                    self.add_relationship(key, value)
            except:
                # 值不可哈希，不创建关系
                pass
        
        # 如果提供了元数据，存储元数据
        if metadata is not None:
            self._metadata[key] = metadata
            
        # 更新搜索索引
        self._update_search_index(key, value)
            
        return self
    
    def _update_search_index(self, key, value):
        """更新搜索索引"""
        # 为字符串类型的键和值创建索引
        if isinstance(key, str):
            # 将键转为小写并拆分单词，用于更快的搜索
            lower_key = key.lower()
            words = lower_key.split()
            
            # 将每个单词加入索引
            for word in words:
                if word not in self._search_index:
                    self._search_index[word] = {"keys": set(), "values": set()}
                self._search_index[word]["keys"].add(key)
        
        # 为字符串类型的值创建索引
        if isinstance(value, str):
            lower_value = value.lower()
            words = lower_value.split()
            
            for word in words:
                if word not in self._search_index:
                    self._search_index[word] = {"keys": set(), "values": set()}
                self._search_index[word]["values"].add(value)
    
    def get(self, key, default=None, include_related=False):
        """
        获取键对应的值
        
        参数:
            key: 键
            default: 如果键不存在返回的默认值
            include_related: 是否包含关联节点
            
        返回:
            如果include_related为False: 值或默认值
            如果include_related为True: {"value": 值, "related": 关联节点集合}
        """
        try:
            # 优先从哈希表获取
            value = self._hash_table.get(key, default)
            
            # 如果找不到，尝试从双向映射获取
            if value is default:
                value = self._bidir_map.get_by_key(key, default)
                
            # 是否包含关联节点
            if include_related and key in self._rel_chain.graph:
                related = self._rel_chain.get_related_nodes(key)
                return {
                    "value": value,
                    "related": related,
                    "metadata": self._metadata.get(key)
                }
                
            return value
                
        except Exception as e:
            # 处理任何异常
            return default
    
    def delete(self, key, keep_relationships=False):
        """
        删除键值对
        
        参数:
            key: 要删除的键
            keep_relationships: 是否保留关系
            
        返回:
            self 实例（链式调用）
            
        异常:
            KeyError: 如果键不存在
        """
        # 检查键是否存在
        if not self._hash_table.contains_key(key):
            raise self.KeyError(f"键不存在: {key}")
            
        # 获取值用于更新索引
        value = self._hash_table.get(key)
        
        # 从哈希表删除
        self._hash_table.delete(key)
        
        # 从双向映射删除
        self._bidir_map.remove(key)
        
        # 更新反向索引
        try:
            if value in self._value_index and key in self._value_index[value]:
                self._value_index[value].remove(key)
                if not self._value_index[value]:
                    del self._value_index[value]
        except:
            pass
        
        # 处理关系
        if not keep_relationships and key in self._rel_chain.graph:
            # 备份相关节点以便清理元数据
            related_nodes = list(self._rel_chain.graph[key])
            
            # 移除所有关系
            for related_node in related_nodes:
                self._rel_chain.remove_relationship(key, related_node)
                
                # 清理关系元数据
                rel_key_1 = f"{key}:{related_node}"
                rel_key_2 = f"{related_node}:{key}"
                
                if rel_key_1 in self._relationship_metadata:
                    del self._relationship_metadata[rel_key_1]
                if rel_key_2 in self._relationship_metadata:
                    del self._relationship_metadata[rel_key_2]
            
            # 删除节点
            if key in self._rel_chain.graph:
                del self._rel_chain.graph[key]
        
        # 清理元数据
        if key in self._metadata:
            del self._metadata[key]
            
        # 清理搜索索引
        self._clean_search_index(key, value)
            
        return self
    
    def _clean_search_index(self, key, value):
        """清理搜索索引中的条目"""
        if isinstance(key, str):
            lower_key = key.lower()
            words = lower_key.split()
            
            for word in words:
                if word in self._search_index and "keys" in self._search_index[word]:
                    self._search_index[word]["keys"].discard(key)
                    if not self._search_index[word]["keys"] and not self._search_index[word].get("values"):
                        del self._search_index[word]
        
        if isinstance(value, str):
            lower_value = value.lower()
            words = lower_value.split()
            
            for word in words:
                if word in self._search_index and "values" in self._search_index[word]:
                    self._search_index[word]["values"].discard(value)
                    if not self._search_index[word]["values"] and not self._search_index[word].get("keys"):
                        del self._search_index[word]
    
    def add_relationship(self, node1, node2):
        """
        在两个节点之间添加关系
        
        参数:
            node1: 第一个节点
            node2: 第二个节点
            
        返回:
            self 实例（链式调用）
            
        异常:
            ValueError: 如果节点不可哈希
        """
        try:
            hash(node1)
            hash(node2)
        except Exception as e:
            raise self.ValueError(f"节点不可哈希: {e}")
            
        self._rel_chain.add_relationship(node1, node2)
        return self
    
    def add_relationship_with_metadata(self, node1, node2, metadata=None):
        """
        在两个节点之间添加带元数据的关系
        
        参数:
            node1: 第一个节点
            node2: 第二个节点
            metadata: 关系元数据
            
        返回:
            self 实例（链式调用）
        """
        # 验证节点存在或可哈希
        try:
            hash(node1)
            hash(node2)
        except Exception as e:
            raise self.ValueError(f"节点不可哈希: {e}")
        
        # 添加关系
        self._rel_chain.add_relationship(node1, node2)
        
        # 存储元数据
        if metadata is not None:
            rel_key = f"{node1}:{node2}"
            self._relationship_metadata[rel_key] = metadata
            
        return self
    
    def get_relationship_metadata(self, node1, node2):
        """
        获取关系元数据
        
        参数:
            node1: 第一个节点
            node2: 第二个节点
            
        返回:
            关系元数据或None
        """
        # 尝试两种顺序的键
        rel_key_1 = f"{node1}:{node2}"
        rel_key_2 = f"{node2}:{node1}"
        
        return self._relationship_metadata.get(rel_key_1) or self._relationship_metadata.get(rel_key_2)
    
    def remove_relationship(self, node1, node2):
        """
        移除两个节点之间的关系
        
        参数:
            node1: 第一个节点
            node2: 第二个节点
            
        返回:
            self 实例（链式调用）
        """
        self._rel_chain.remove_relationship(node1, node2)
        
        # 清理元数据
        rel_key_1 = f"{node1}:{node2}"
        rel_key_2 = f"{node2}:{node1}"
        
        if rel_key_1 in self._relationship_metadata:
            del self._relationship_metadata[rel_key_1]
        if rel_key_2 in self._relationship_metadata:
            del self._relationship_metadata[rel_key_2]
            
        return self
    
    def set_node_metadata(self, node, metadata):
        """
        设置节点元数据
        
        参数:
            node: 节点
            metadata: 元数据
            
        返回:
            self 实例（链式调用）
        """
        self._metadata[node] = metadata
        return self
    
    def get_node_metadata(self, node):
        """
        获取节点元数据
        
        参数:
            node: 节点
            
        返回:
            节点元数据或None
        """
        return self._metadata.get(node)
    
    def query_by_value(self, value):
        """
        通过值查询相关信息
        
        参数:
            value: 要查询的值
            
        返回:
            包含键、关联节点和路径的字典
        """
        result = {
            "keys": set(),
            "related_nodes": set(),
            "paths": []
        }
        
        # 使用反向索引快速获取键
        if value in self._value_index:
            result["keys"] = self._value_index[value]
        else:
            # 如果不在索引中，尝试使用双向映射
            keys = self._bidir_map.get_by_value(value)
            if keys:
                result["keys"] = keys
        
        # 获取关联节点
        if value in self._rel_chain.graph:
            result["related_nodes"] = self._rel_chain.get_related_nodes(value)
            
        # 查找路径
        for key in result["keys"]:
            if key != value:  # 避免自我路径
                path = self._rel_chain.find_path(key, value)
                if path:
                    result["paths"].append(path)
                    
        return result
    
    def find_connected_components(self):
        """
        查找图中的所有连通分量
        
        返回:
            连通分量列表，每个分量是节点集合
        """
        # 由于RelationshipChain没有这个方法，我们直接在这里实现
        visited = set()
        components = []
        
        for node in self._rel_chain.graph:
            if node not in visited:
                component = set()
                queue = deque([node])
                
                while queue:
                    current = queue.popleft()
                    if current not in visited:
                        visited.add(current)
                        component.add(current)
                        for neighbor in self._rel_chain.graph.get(current, []):
                            if neighbor not in visited:
                                queue.append(neighbor)
                
                components.append(component)
                
        return components
    
    def find_common_connections(self, node1, node2):
        """
        查找两个节点的共同连接
        
        参数:
            node1: 第一个节点
            node2: 第二个节点
            
        返回:
            共同连接节点集合
        """
        if node1 not in self._rel_chain.graph or node2 not in self._rel_chain.graph:
            return set()
        return self._rel_chain.graph[node1].intersection(self._rel_chain.graph[node2])
    
    def find_shortest_paths(self, start, end, limit=10):
        """
        查找从起点到终点的多条最短路径
        
        参数:
            start: 起点
            end: 终点
            limit: 结果数量限制
            
        返回:
            路径列表
        """
        if start not in self._rel_chain.graph or end not in self._rel_chain.graph:
            return []
            
        # 使用BFS查找最短路径
        visited = {start: None}
        queue = deque([(start, [start])])
        paths = []
        min_length = float('inf')
        
        while queue and len(paths) < limit:
            current, path = queue.popleft()
            
            # 如果路径长度超过已知最短路径，不再考虑
            if len(path) > min_length:
                break
                
            if current == end:
                paths.append(path)
                min_length = len(path)
            else:
                for neighbor in self._rel_chain.graph.get(current, []):
                    if neighbor not in visited or len(path) <= len(visited[neighbor]):
                        new_path = path + [neighbor]
                        queue.append((neighbor, new_path))
                        visited[neighbor] = path
        
        return paths
    
    def recursive_search(self, value, max_depth=3):
        """
        递归搜索与值相关的所有节点，带循环检测
        
        参数:
            value: 搜索值
            max_depth: 最大搜索深度
            
        返回:
            搜索结果字典
        """
        visited = set()
        result = {}
        
        def dfs(node, depth=0, path=None):
            if path is None:
                path = []
                
            # 检测循环
            if node in path:
                # 发现循环，标记但不继续递归
                if node in result:
                    result[node]["is_cycle"] = True
                return
                
            # 检查深度和访问状态
            if depth > max_depth or node in visited:
                return
            
            # 标记为已访问
            visited.add(node)
            path = path + [node]
            
            # 添加节点信息
            if node not in result:
                result[node] = {
                    "direct_value": self._hash_table.get(node, None),
                    "related": set(),
                    "metadata": self._metadata.get(node),
                    "depth": depth,
                    "is_cycle": False
                }
            
            # 添加关系信息
            if node in self._rel_chain.graph:
                for related in self._rel_chain.graph[node]:
                    # 添加关系信息
                    result[node]["related"].add(related)
                    # 添加关系元数据
                    rel_meta = self.get_relationship_metadata(node, related)
                    if rel_meta:
                        if "relationship_metadata" not in result[node]:
                            result[node]["relationship_metadata"] = {}
                        result[node]["relationship_metadata"][str(related)] = rel_meta
                    
                    # 递归，但传递路径用于循环检测
                    dfs(related, depth + 1, path)
        
        # 从值开始搜索
        if value in self._value_index:
            for key in self._value_index[value]:
                dfs(key)
        elif value in self._bidir_map.backward:
            for key in self._bidir_map.backward[value]:
                dfs(key)
        
        # 如果值本身是节点，也从它开始搜索
        if value in self._rel_chain.graph:
            dfs(value)
            
        return result
    
    def get_by_multiple_keys(self, keys, mode="any", include_related=False):
        """
        通过多个键查找值和关系
        
        参数:
            keys: 要查找的键的列表或集合
            mode: 查找模式：'any'(任意)、'all'(全部)或'each'(每个)
            include_related: 是否包含关联节点
            
        返回:
            根据mode不同返回不同格式的结果
        """
        # 参数验证
        if not keys:
            return {} if mode in ["any", "each"] else set()
            
        valid_modes = ["any", "all", "each"]
        if mode not in valid_modes:
            raise self.ValueError(f"不支持的模式: {mode}，请使用 {', '.join(valid_modes)}")
            
        # 验证键是否存在
        existing_keys = []
        for key in keys:
            if self._hash_table.contains_key(key):
                existing_keys.append(key)
                
        if not existing_keys:
            return {} if mode in ["any", "each"] else set()
            
        # 根据不同模式执行操作
        if mode == "any":
            # 返回任意键匹配的结果（并集）
            result = {}
            for key in existing_keys:
                if include_related and key in self._rel_chain.graph:
                    result[key] = {
                        "value": self._hash_table.get(key),
                        "related": self._rel_chain.get_related_nodes(key),
                        "metadata": self._metadata.get(key)
                    }
                else:
                    result[key] = self._hash_table.get(key)
            return result
            
        elif mode == "all":
            # 返回所有键都匹配的共同值（交集）
            # 获取第一个键的值和关联节点
            first_key = existing_keys[0]
            common_values = set()
            
            # 添加直接值
            first_value = self._hash_table.get(first_key)
            if first_value is not None:
                try:
                    hash(first_value)
                    common_values.add(first_value)
                except:
                    pass  # 值不可哈希，不添加
                
            # 添加关联节点
            if include_related and first_key in self._rel_chain.graph:
                common_values.update(self._rel_chain.get_related_nodes(first_key))
                
            # 计算交集
            for key in existing_keys[1:]:
                current_values = set()
                
                # 添加直接值
                value = self._hash_table.get(key)
                if value is not None:
                    try:
                        hash(value)
                        current_values.add(value)
                    except:
                        pass
                
                # 添加关联节点
                if include_related and key in self._rel_chain.graph:
                    current_values.update(self._rel_chain.get_related_nodes(key))
                
                # 求交集
                common_values &= current_values
                
                if not common_values:
                    break  # 交集为空，提前返回
                    
            return common_values
            
        elif mode == "each":
            # 返回每个键的单独结果
            result = {}
            for key in existing_keys:
                value_info = {"value": self._hash_table.get(key)}
                
                if key in self._metadata:
                    value_info["metadata"] = self._metadata[key]
                    
                if include_related and key in self._rel_chain.graph:
                    value_info["related"] = self._rel_chain.get_related_nodes(key)
                    
                result[key] = value_info
                    
            return result
    
    def search(self, query, search_in=None, max_results=None, fuzzy=False, case_sensitive=False):
        """
        全局搜索函数，支持高性能索引搜索
        
        参数:
            query: 搜索查询
            search_in: 搜索范围
            max_results: 最大结果数量
            fuzzy: 是否进行模糊匹配
            case_sensitive: 是否区分大小写
            
        返回:
            搜索结果字典
        """
        if search_in is None:
            search_in = ["keys", "values", "metadata", "relationships"]
            
        # 初始化结果
        results = {
            "keys": [],
            "values": [],
            "relationships": [],
            "metadata": []
        }
        
        # 如果query是字符串，优化搜索
        if isinstance(query, str):
            return self._search_string(query, search_in, max_results, fuzzy, case_sensitive)
        else:
            # 非字符串查询，使用精确匹配
            return self._search_exact(query, search_in, max_results)
    
    def _search_string(self, query, search_in, max_results, fuzzy, case_sensitive):
        """字符串搜索优化实现"""
        results = {
            "keys": [],
            "values": [],
            "relationships": [],
            "metadata": []
        }
        
        # 规范化查询
        if not case_sensitive:
            query = query.lower()
            
        # 拆分查询词
        words = query.split()
        
        # 搜索键
        if "keys" in search_in:
            # 使用索引快速查找
            matching_keys = set()
            
            if not fuzzy:
                # 精确匹配，使用索引
                for word in words:
                    if word in self._search_index and "keys" in self._search_index[word]:
                        # 第一个词，初始化结果集
                        if not matching_keys:
                            matching_keys = self._search_index[word]["keys"].copy()
                        else:
                            # 取交集，所有词都要匹配
                            matching_keys &= self._search_index[word]["keys"]
                            
                        if not matching_keys:
                            break  # 已经没有匹配项，提前退出
            else:
                # 模糊匹配，需要全扫描
                for key in self._hash_table.keys():
                    if isinstance(key, str):
                        key_str = key if case_sensitive else key.lower()
                        # 检查是否是子串匹配
                        if query in key_str:
                            matching_keys.add(key)
                            
            # 将匹配的键添加到结果
            results["keys"] = list(matching_keys)
            if max_results and len(results["keys"]) > max_results:
                results["keys"] = results["keys"][:max_results]
        
        # 搜索值
        if "values" in search_in:
            matching_values = set()
            
            if not fuzzy:
                # 使用索引查找
                for word in words:
                    if word in self._search_index and "values" in self._search_index[word]:
                        new_matching = set()
                        for value in self._search_index[word]["values"]:
                            # 找到值对应的键
                            if value in self._value_index:
                                for key in self._value_index[value]:
                                    new_matching.add((key, value))
                                    
                        if not matching_values:
                            matching_values = new_matching
                        else:
                            matching_values &= new_matching
                            
                        if not matching_values:
                            break
            else:
                # 模糊匹配
                for key, value in self._hash_table.items():
                    if isinstance(value, str):
                        value_str = value if case_sensitive else value.lower()
                        if query in value_str:
                            matching_values.add((key, value))
            
            # 添加到结果
            results["values"] = list(matching_values)
            if max_results and len(results["values"]) > max_results:
                results["values"] = results["values"][:max_results]
        
        # 搜索关系
        if "relationships" in search_in:
            matching_relationships = []
            
            # 遍历所有节点关系
            for node, connections in self._rel_chain.graph.items():
                node_str = str(node)
                node_matches = False
                
                if isinstance(node, str):
                    node_lower = node if case_sensitive else node.lower()
                    if fuzzy:
                        node_matches = query in node_lower
                    else:
                        node_matches = all(word in node_lower for word in words)
                
                if node_matches:
                    # 节点匹配，添加所有关系
                    for conn in connections:
                        matching_relationships.append((node, conn))
                        
                        if max_results and len(matching_relationships) >= max_results:
                            break
                else:
                    # 检查连接节点是否匹配
                    for conn in connections:
                        conn_matches = False
                        
                        if isinstance(conn, str):
                            conn_lower = conn if case_sensitive else conn.lower()
                            if fuzzy:
                                conn_matches = query in conn_lower
                            else:
                                conn_matches = all(word in conn_lower for word in words)
                                
                        if conn_matches:
                            matching_relationships.append((node, conn))
                            
                            if max_results and len(matching_relationships) >= max_results:
                                break
                                
                if max_results and len(matching_relationships) >= max_results:
                    break
                    
            results["relationships"] = matching_relationships
        
        # 搜索元数据
        if "metadata" in search_in:
            matching_metadata = []
            
            for key, meta in self._metadata.items():
                # 检查元数据或键是否匹配
                meta_matches = False
                
                # 如果元数据是字典
                if isinstance(meta, dict):
                    for meta_key, meta_value in meta.items():
                        meta_key_matches = False
                        meta_value_matches = False
                        
                        if isinstance(meta_key, str):
                            meta_key_lower = meta_key if case_sensitive else meta_key.lower()
                            if fuzzy:
                                meta_key_matches = query in meta_key_lower
                            else:
                                meta_key_matches = all(word in meta_key_lower for word in words)
                                
                        if isinstance(meta_value, str):
                            meta_value_lower = meta_value if case_sensitive else meta_value.lower()
                            if fuzzy:
                                meta_value_matches = query in meta_value_lower
                            else:
                                meta_value_matches = all(word in meta_value_lower for word in words)
                                
                        if meta_key_matches or meta_value_matches:
                            meta_matches = True
                            break
                # 如果元数据是字符串
                elif isinstance(meta, str):
                    meta_lower = meta if case_sensitive else meta.lower()
                    if fuzzy:
                        meta_matches = query in meta_lower
                    else:
                        meta_matches = all(word in meta_lower for word in words)
                        
                if meta_matches:
                    matching_metadata.append((key, meta))
                    
                    if max_results and len(matching_metadata) >= max_results:
                        break
                        
            results["metadata"] = matching_metadata
            
        return results
    
    def _search_exact(self, query, search_in, max_results):
        """非字符串精确搜索实现"""
        results = {
            "keys": [],
            "values": [],
            "relationships": [],
            "metadata": []
        }
        
        # 搜索键
        if "keys" in search_in:
            if query in self._hash_table.keys():
                results["keys"].append(query)
                
        # 搜索值
        if "values" in search_in:
            # 使用反向索引更快查找
            if query in self._value_index:
                for key in self._value_index[query]:
                    results["values"].append((key, query))
                    
                    if max_results and len(results["values"]) >= max_results:
                        break
                        
        # 搜索关系
        if "relationships" in search_in:
            if query in self._rel_chain.graph:
                for conn in self._rel_chain.graph[query]:
                    results["relationships"].append((query, conn))
                    
                    if max_results and len(results["relationships"]) >= max_results:
                        break
                        
            # 还需要检查连接到该查询的节点
            for node, connections in self._rel_chain.graph.items():
                if query in connections:
                    results["relationships"].append((node, query))
                    
                    if max_results and len(results["relationships"]) >= max_results:
                        break
                        
        # 搜索元数据
        if "metadata" in search_in:
            for key, meta in self._metadata.items():
                if meta == query or (isinstance(meta, dict) and (query in meta or query in meta.values())):
                    results["metadata"].append((key, meta))
                    
                    if max_results and len(results["metadata"]) >= max_results:
                        break
                        
        return results
    
    # 下面是属性访问和辅助方法
    
    def contains_key(self, key):
        """检查是否存在指定键"""
        return self._hash_table.contains_key(key)
        
    def keys(self):
        """返回所有键的列表"""
        return self._hash_table.keys()
        
    def values(self):
        """返回所有值的列表"""
        return self._hash_table.values()
        
    def items(self):
        """返回所有键值对的列表"""
        return self._hash_table.items()
        
    def is_empty(self):
        """检查数据结构是否为空"""
        return self._hash_table.is_empty()
        
    def size(self):
        """返回键值对数量"""
        return len(self._hash_table)
        
    def clear(self):
        """清空数据结构"""
        self._hash_table.clear()
        self._bidir_map.clear()
        self._rel_chain.graph.clear()
        self._metadata.clear()
        self._relationship_metadata.clear()
        self._value_index.clear()
        self._search_index.clear()
        return self
    
    def get_statistics(self):
        """获取数据结构统计信息"""
        return {
            "keys_count": len(self._hash_table),
            "values_count": len(self._value_index),
            "relationships_count": sum(len(connections) for connections in self._rel_chain.graph.values()) // 2,
            "nodes_count": len(self._rel_chain.graph),
            "metadata_count": len(self._metadata),
            "relationship_metadata_count": len(self._relationship_metadata),
            "indexed_words_count": len(self._search_index)
        }
    
    # 实现标准Python接口
    
    def __getitem__(self, key):
        """支持dict风格的访问: data[key]"""
        value = self.get(key)
        if value is None:
            raise self.KeyError(f"键不存在: {key}")
        return value
        
    def __setitem__(self, key, value):
        """支持dict风格的赋值: data[key] = value"""
        return self.put(key, value)
        
    def __delitem__(self, key):
        """支持dict风格的删除: del data[key]"""
        return self.delete(key)
        
    def __contains__(self, key):
        """支持成员判断: key in data"""
        return self.contains_key(key)
        
    def __len__(self):
        """支持len()函数"""
        return len(self._hash_table)
        
    def __iter__(self):
        """支持迭代"""
        return iter(self._hash_table)
        
    def __str__(self):
        """字符串表示"""
        stats = self.get_statistics()
        return (f"ImprovedCombinedDataStructure(keys={stats['keys_count']}, "
                f"relationships={stats['relationships_count']}, "
                f"nodes={stats['nodes_count']})")
                
    def __repr__(self):
        """代码表示"""
        return f"ICombinedDataStructure(size={len(self)})"
        
    # 导出和导入功能
    
    def to_dict(self):
        """
        将数据结构导出为字典
        
        返回:
            dict: 包含所有数据的字典
        """
        result = {
            "data": {},
            "relationships": [],
            "metadata": {},
            "relationship_metadata": {}
        }
        
        # 导出键值对
        for key, value in self._hash_table.items():
            # 尝试将键转换为可序列化形式
            try:
                key_str = str(key)
                # 对于可序列化的值，直接添加
                try:
                    json.dumps(value)  # 测试值是否可序列化
                    result["data"][key_str] = value
                except (TypeError, OverflowError):
                    # 值不可序列化，转换为字符串
                    result["data"][key_str] = str(value)
            except:
                # 如果键无法转换为字符串，跳过
                continue
        
        # 导出关系
        for node1, connections in self._rel_chain.graph.items():
            node1_str = str(node1)
            for node2 in connections:
                node2_str = str(node2)
                # 避免重复添加关系
                if not any(r[0] == node2_str and r[1] == node1_str for r in result["relationships"]):
                    result["relationships"].append([node1_str, node2_str])
        
        # 导出元数据
        for key, meta in self._metadata.items():
            try:
                key_str = str(key)
                # 尝试直接导出元数据
                try:
                    json.dumps(meta)  # 测试元数据是否可序列化
                    result["metadata"][key_str] = meta
                except (TypeError, OverflowError):
                    # 元数据不可序列化，转换为字符串
                    result["metadata"][key_str] = str(meta)
            except:
                continue
        
        # 导出关系元数据
        for rel_key, meta in self._relationship_metadata.items():
            try:
                # 尝试直接导出元数据
                try:
                    json.dumps(meta)  # 测试元数据是否可序列化
                    result["relationship_metadata"][rel_key] = meta
                except (TypeError, OverflowError):
                    # 元数据不可序列化，转换为字符串
                    result["relationship_metadata"][rel_key] = str(meta)
            except:
                continue
                
        return result
    
    def from_dict(self, data_dict, clear_existing=False):
        """
        从字典导入数据
        
        参数:
            data_dict: 包含数据的字典
            clear_existing: 是否在导入前清空现有数据
            
        返回:
            self 实例
            
        异常:
            ValueError: 如果数据格式无效
        """
        if not isinstance(data_dict, dict):
            raise self.ValueError("导入数据必须是字典")
            
        # 验证必需的键
        required_keys = ["data"]
        for key in required_keys:
            if key not in data_dict:
                raise self.ValueError(f"导入数据缺少必需的键: {key}")
                
        # 清空现有数据
        if clear_existing:
            self.clear()
            
        # 导入键值对
        for key_str, value in data_dict.get("data", {}).items():
            try:
                self.put(key_str, value)
            except Exception as e:
                print(f"导入键值对出错: {key_str} -> {value}, 错误: {e}")
        
        # 导入关系
        for rel in data_dict.get("relationships", []):
            if isinstance(rel, list) and len(rel) == 2:
                try:
                    self.add_relationship(rel[0], rel[1])
                except Exception as e:
                    print(f"导入关系出错: {rel[0]} - {rel[1]}, 错误: {e}")
        
        # 导入元数据
        for key_str, meta in data_dict.get("metadata", {}).items():
            try:
                if key_str in self:
                    self.set_node_metadata(key_str, meta)
            except Exception as e:
                print(f"导入元数据出错: {key_str}, 错误: {e}")
        
        # 导入关系元数据
        for rel_key, meta in data_dict.get("relationship_metadata", {}).items():
            try:
                # 尝试拆分关系键
                parts = rel_key.split(":")
                if len(parts) == 2:
                    node1, node2 = parts[0], parts[1]
                    # 检查关系是否存在
                    if node1 in self._rel_chain.graph and node2 in self._rel_chain.graph[node1]:
                        self.add_relationship_with_metadata(node1, node2, meta)
            except Exception as e:
                print(f"导入关系元数据出错: {rel_key}, 错误: {e}")
                
        return self
    
    def export_to_json(self, file_path):
        """
        将数据导出为JSON文件
        
        参数:
            file_path: JSON文件路径
            
        返回:
            bool: 成功则为True
            
        异常:
            Exception: 如果导出失败
        """
        try:
            data = self.to_dict()
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            raise Exception(f"导出JSON失败: {e}")
    
    @classmethod
    def import_from_json(cls, file_path, initial_capacity=16, load_factor=0.75):
        """
        从JSON文件导入数据
        
        参数:
            file_path: JSON文件路径
            initial_capacity: 哈希表初始容量
            load_factor: 哈希表负载因子
            
        返回:
            新的实例
            
        异常:
            Exception: 如果导入失败
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            instance = cls(initial_capacity, load_factor)
            instance.from_dict(data)
            return instance
        except Exception as e:
            raise Exception(f"导入JSON失败: {e}")
    
    # 批量操作
    
    def batch_put(self, key_value_pairs, create_relationships=False, metadata_dict=None):
        """
        批量添加键值对
        
        参数:
            key_value_pairs: 键值对字典
            create_relationships: 是否创建关系
            metadata_dict: 元数据字典 {key: metadata}
            
        返回:
            self 实例
        """
        if metadata_dict is None:
            metadata_dict = {}
            
        for key, value in key_value_pairs.items():
            metadata = metadata_dict.get(key)
            self.put(key, value, create_relationship=create_relationships, metadata=metadata)
            
        return self
    
    def batch_delete(self, keys, keep_relationships=False):
        """
        批量删除键值对
        
        参数:
            keys: 要删除的键列表
            keep_relationships: 是否保留关系
            
        返回:
            self 实例
        """
        for key in keys:
            try:
                self.delete(key, keep_relationships=keep_relationships)
            except:
                # 忽略不存在的键
                pass
            
        return self
    
    def batch_add_relationships(self, relationship_pairs, metadata_dict=None):
        """
        批量添加关系
        
        参数:
            relationship_pairs: 关系对列表 [(node1, node2), ...]
            metadata_dict: 元数据字典 {(node1, node2): metadata}
            
        返回:
            self 实例
        """
        if metadata_dict is None:
            metadata_dict = {}
            
        for node1, node2 in relationship_pairs:
            metadata = metadata_dict.get((node1, node2)) or metadata_dict.get((node2, node1))
            if metadata:
                self.add_relationship_with_metadata(node1, node2, metadata)
            else:
                self.add_relationship(node1, node2)
                
        return self
        
    # 加密功能方法
    
    def setup_encryption(self, password, salt=None, iterations=100000):
        """
        设置加密功能
        
        参数:
            password: 加密密码
            salt: 盐值，如果为None则自动生成
            iterations: PBKDF2迭代次数
            
        返回:
            self 实例
        """
        if not salt:
            salt = os.urandom(16)  # 生成16字节的盐
        
        # 使用PBKDF2生成密钥
        key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, iterations, dklen=32)
        
        # 存储加密配置
        self._encryption = {
            'enabled': True,
            'key': key,
            'salt': salt,
            'iterations': iterations,
            'encrypted_keys': set(),
            'algorithm': 'AES-GCM' if CRYPTOGRAPHY_AVAILABLE else 'XOR'
        }
        
        return self
    
    def disable_encryption(self):
        """
        禁用加密功能
        
        返回:
            self 实例
            
        异常:
            ValueError: 如果有未解密的数据
        """
        if hasattr(self, '_encryption') and self._encryption.get('enabled'):
            # 检查是否有未解密的数据
            if len(self._encryption.get('encrypted_keys', set())) > 0:
                raise ValueError("有未解密的数据，请先调用decrypt_all_data()解密所有数据")
            
            # 禁用加密
            self._encryption['enabled'] = False
        
        return self
    
    def is_encryption_enabled(self):
        """
        检查加密是否已启用
        
        返回:
            bool: 如果加密已启用则为True
        """
        return hasattr(self, '_encryption') and self._encryption.get('enabled', False)
    
    def _encrypt_value(self, value):
        """
        加密值
        
        参数:
            value: 要加密的值
            
        返回:
            bytes: 加密后的数据
            
        异常:
            ValueError: 如果加密未启用
        """
        if not self.is_encryption_enabled():
            raise ValueError("加密未启用")
        
        # 序列化值
        value_bytes = json.dumps(value).encode('utf-8')
        
        # 使用AES-GCM加密(如果可用)
        if CRYPTOGRAPHY_AVAILABLE:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            aesgcm = AESGCM(self._encryption['key'])
            nonce = os.urandom(12)
            ciphertext = aesgcm.encrypt(nonce, value_bytes, None)
            encrypted_data = nonce + ciphertext
        else:
            # 使用基本的XOR加密(不够安全，仅作示例)
            key = self._encryption['key']
            nonce = os.urandom(12)
            encrypted_data = bytearray(len(value_bytes) + 12)
            encrypted_data[:12] = nonce
            
            for i in range(len(value_bytes)):
                encrypted_data[i + 12] = value_bytes[i] ^ key[i % len(key)] ^ nonce[i % 12]
        
        # 返回Base64编码的加密数据
        return base64.b64encode(encrypted_data).decode('utf-8')
    
    def _decrypt_value(self, encrypted_data):
        """
        解密值
        
        参数:
            encrypted_data: 加密的数据(Base64编码的字符串)
            
        返回:
            解密后的值
            
        异常:
            ValueError: 如果加密未启用或解密失败
        """
        if not self.is_encryption_enabled():
            raise ValueError("加密未启用")
        
        # 解码Base64
        data = base64.b64decode(encrypted_data.encode('utf-8'))
        
        try:
            # 使用AES-GCM解密(如果可用)
            if CRYPTOGRAPHY_AVAILABLE:
                from cryptography.hazmat.primitives.ciphers.aead import AESGCM
                nonce = data[:12]
                ciphertext = data[12:]
                aesgcm = AESGCM(self._encryption['key'])
                plaintext = aesgcm.decrypt(nonce, ciphertext, None)
            else:
                # 使用基本的XOR解密
                key = self._encryption['key']
                nonce = data[:12]
                ciphertext = data[12:]
                plaintext = bytearray(len(ciphertext))
                
                for i in range(len(ciphertext)):
                    plaintext[i] = ciphertext[i] ^ key[i % len(key)] ^ nonce[i % 12]
                
                plaintext = bytes(plaintext)
            
            # 反序列化数据
            return json.loads(plaintext.decode('utf-8'))
        except Exception as e:
            raise ValueError(f"解密失败: {e}")
    
    def put_encrypted(self, key, value, create_relationship=False, metadata=None):
        """
        添加加密的键值对
        
        参数:
            key: 键
            value: 值
            create_relationship: 是否创建关系
            metadata: 元数据
            
        返回:
            self 实例
            
        异常:
            ValueError: 如果加密未启用
        """
        if not self.is_encryption_enabled():
            raise ValueError("加密未启用")
        
        # 加密值
        encrypted_value = self._encrypt_value(value)
        
        # 存储加密值
        self.put(key, encrypted_value, create_relationship, metadata)
        
        # 记录该键存储了加密数据
        self._encryption['encrypted_keys'].add(key)
        
        return self
    
    def get_encrypted(self, key, default=None, include_related=False, auto_decrypt=True):
        """
        获取加密的值
        
        参数:
            key: 键
            default: 如果键不存在返回的默认值
            include_related: 是否包含关联节点
            auto_decrypt: 是否自动解密
            
        返回:
            解密后的值或加密形式
            
        异常:
            ValueError: 如果加密未启用
        """
        if not self.is_encryption_enabled():
            raise ValueError("加密未启用")
        
        # 检查键是否存在
        if not self.contains_key(key) or key not in self._encryption['encrypted_keys']:
            return default
        
        # 获取加密值
        encrypted_value = self.get(key, default, include_related)
        
        # 如果不需要自动解密，直接返回
        if not auto_decrypt:
            return encrypted_value
        
        # 如果包含关联节点，需要特殊处理
        if include_related and isinstance(encrypted_value, dict) and 'value' in encrypted_value:
            try:
                decrypted_value = self._decrypt_value(encrypted_value['value'])
                result = {
                    'value': decrypted_value,
                    'related': encrypted_value.get('related', set()),
                    'metadata': encrypted_value.get('metadata')
                }
                return result
            except Exception as e:
                raise ValueError(f"解密失败: {e}")
        
        # 否则直接解密
        try:
            return self._decrypt_value(encrypted_value)
        except Exception as e:
            raise ValueError(f"解密失败: {e}")
    
    def update_encrypted(self, key, value):
        """
        更新加密的值
        
        参数:
            key: 键
            value: 新值
            
        返回:
            self 实例
            
        异常:
            ValueError: 如果加密未启用或键不存在
            KeyError: 如果键不存在
        """
        if not self.is_encryption_enabled():
            raise ValueError("加密未启用")
        
        if not self.contains_key(key):
            raise KeyError(f"键不存在: {key}")
        
        # 如果键存储了加密数据，更新加密数据
        if key in self._encryption['encrypted_keys']:
            encrypted_value = self._encrypt_value(value)
            self.put(key, encrypted_value)
        else:
            # 否则直接添加为加密数据
            self.put_encrypted(key, value)
        
        return self
    
    def encrypt_existing(self, keys=None):
        """
        加密现有数据
        
        参数:
            keys: 要加密的键列表，如果为None则加密所有未加密数据
            
        返回:
            self 实例
        """
        if not self.is_encryption_enabled():
            raise ValueError("加密未启用")
        
        # 确定要加密的键
        if keys is None:
            all_keys = set(self.keys())
            keys = all_keys - self._encryption['encrypted_keys']
        
        # 加密指定的键
        for key in keys:
            if key not in self._encryption['encrypted_keys'] and self.contains_key(key):
                value = self.get(key)
                encrypted_value = self._encrypt_value(value)
                self.put(key, encrypted_value)
                self._encryption['encrypted_keys'].add(key)
        
        return self
    
    def decrypt_all(self):
        """
        解密所有加密数据
        
        返回:
            self 实例
        """
        if not self.is_encryption_enabled():
            raise ValueError("加密未启用")
        
        # 获取加密键的快照
        encrypted_keys = self._encryption['encrypted_keys'].copy()
        
        # 解密所有数据
        for key in encrypted_keys:
            if self.contains_key(key):
                try:
                    encrypted_value = self.get(key)
                    decrypted_value = self._decrypt_value(encrypted_value)
                    self.put(key, decrypted_value)
                    self._encryption['encrypted_keys'].remove(key)
                except Exception as e:
                    print(f"解密键 '{key}' 失败: {e}")
        
        return self
    
    def is_encrypted(self, key):
        """
        检查键是否存储了加密数据
        
        参数:
            key: 键
            
        返回:
            bool: 如果键存储了加密数据则为True
        """
        return (
            self.is_encryption_enabled() and 
            key in self._encryption.get('encrypted_keys', set())
        )
    
    def get_encryption_info(self):
        """
        获取加密信息
        
        返回:
            dict: 加密信息
        """
        if not hasattr(self, '_encryption'):
            return {'enabled': False}
        
        return {
            'enabled': self._encryption.get('enabled', False),
            'algorithm': self._encryption.get('algorithm', 'None'),
            'encrypted_keys_count': len(self._encryption.get('encrypted_keys', set())),
            'iterations': self._encryption.get('iterations', 0)
        }
    
    def export_encrypted(self, include_relationships=True, include_metadata=True):
        """
        导出加密数据
        
        参数:
            include_relationships: 是否包含关系
            include_metadata: 是否包含元数据
            
        返回:
            dict: 加密数据
            
        异常:
            ValueError: 如果加密未启用
        """
        if not self.is_encryption_enabled():
            raise ValueError("加密未启用")
        
        result = {
            'version': 1,
            'encrypted_data': {},
            'salt': base64.b64encode(self._encryption['salt']).decode('utf-8'),
            'iterations': self._encryption['iterations'],
            'algorithm': self._encryption['algorithm']
        }
        
        # 导出加密数据
        for key in self._encryption['encrypted_keys']:
            if self.contains_key(key):
                # 将键转换为字符串
                key_str = str(key)
                encrypted_value = self.get(key)
                result['encrypted_data'][key_str] = encrypted_value
        
        # 导出关系
        if include_relationships:
            result['relationships'] = []
            encrypted_keys_str = {str(k) for k in self._encryption['encrypted_keys']}
            
            for node1 in self._encryption['encrypted_keys']:
                if node1 in self._rel_chain.graph:
                    for node2 in self._rel_chain.graph[node1]:
                        if node2 in self._encryption['encrypted_keys']:
                            result['relationships'].append([str(node1), str(node2)])
        
        # 导出元数据
        if include_metadata:
            result['metadata'] = {}
            for key in self._encryption['encrypted_keys']:
                if key in self._metadata:
                    meta = self._metadata[key]
                    # 加密元数据
                    encrypted_meta = self._encrypt_value(meta)
                    result['metadata'][str(key)] = encrypted_meta
        
        return result
    
    def import_encrypted(self, data, password, merge=True):
        """
        导入加密数据
        
        参数:
            data: 加密数据
            password: 密码
            merge: 是否合并现有数据
            
        返回:
            self 实例
            
        异常:
            ValueError: 如果数据格式无效
        """
        # 验证数据格式
        if not isinstance(data, dict) or 'encrypted_data' not in data or 'salt' not in data:
            raise ValueError("无效的加密数据格式")
        
        # 解码盐值
        salt = base64.b64decode(data['salt'].encode('utf-8'))
        
        # 设置加密
        iterations = data.get('iterations', 100000)
        self.setup_encryption(password, salt, iterations)
        
        # 清除现有数据
        if not merge:
            self.clear()
            self._encryption['encrypted_keys'].clear()
        
        # 导入加密数据
        for key_str, encrypted_value in data['encrypted_data'].items():
            # 尝试转换键类型
            key = self._parse_key(key_str)
            self.put(key, encrypted_value)
            self._encryption['encrypted_keys'].add(key)
        
        # 导入关系
        if 'relationships' in data:
            for rel in data['relationships']:
                if isinstance(rel, list) and len(rel) == 2:
                    node1 = self._parse_key(rel[0])
                    node2 = self._parse_key(rel[1])
                    self.add_relationship(node1, node2)
        
        # 导入元数据
        if 'metadata' in data:
            for key_str, encrypted_meta in data['metadata'].items():
                key = self._parse_key(key_str)
                if key in self._encryption['encrypted_keys']:
                    # 解密元数据并存储
                    try:
                        meta = self._decrypt_value(encrypted_meta)
                        self.set_node_metadata(key, meta)
                    except Exception as e:
                        print(f"解密元数据失败: {key_str} - {e}")
        
        return self
    
    def _parse_key(self, key_str):
        """
        尝试将字符串键解析为适当的类型
        
        参数:
            key_str: 键字符串
            
        返回:
            解析后的键
        """
        # 尝试将键解析为数字或布尔值
        try:
            # 整数
            if key_str.isdigit():
                return int(key_str)
            # 浮点数
            elif key_str.replace('.', '', 1).isdigit() and key_str.count('.') == 1:
                return float(key_str)
            # 布尔值
            elif key_str.lower() == 'true':
                return True
            elif key_str.lower() == 'false':
                return False
            # 默认为字符串
            return key_str
        except:
            return key_str

    def get_nested_value(self, path):
        """
        获取嵌套路径的值，处理各种数据结构
        
        参数:
            path: 路径列表，如果为空则返回整个数据结构
            
        返回:
            查找到的值或None（如果路径无效）
        """
        # 如果路径为空，返回整个数据
        if not path:
            return self
            
        # 使用数据锁保护访问
        try:
            current = self
            
            for i, key in enumerate(path):
                if hasattr(current, 'get_by_key'):
                    # 支持get_by_key方法的数据结构
                    current = current.get_by_key(key)
                elif isinstance(current, dict):
                    # 字典类型
                    if key in current:
                        current = current[key]
                    else:
                        return None
                elif isinstance(current, (list, tuple)) and isinstance(key, int):
                    # 列表或元组类型，键是整数索引
                    if 0 <= key < len(current):
                        current = current[key]
                    else:
                        return None
                else:
                    # 无法访问嵌套值
                    return None
                
                # 如果中间值是None，提前返回
                if current is None:
                    return None
            
            # 处理最终结果（对象可能有专门的处理方法）
            return current
            
        except Exception as e:
            # 发生异常时返回None
            return None