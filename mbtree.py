#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-way B-Tree Implementation with Expression Query Support and Async I/O

This module implements a custom multi-way tree data structure similar to B-trees,
with optimized heap operations, improved tree node management, advanced
expression-based query functionality, and asynchronous I/O support for disk operations.

Created on Fri Jun 20 22:25:53 2025
@author: wuzhixiang
"""

import bisect
import heapq
import re
import operator as op
import asyncio
import json
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, AsyncIterator, Awaitable
from enum import Enum
from abc import ABC, abstractmethod
import time

try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False
    print("Warning: aiofiles not available. Some async I/O features will be limited.")

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False


class SplitStrategy(Enum):
    """
    Enumeration of different node splitting strategies.
    """
    EVEN = "even"           # 均匀分裂：尽可能平均分配
    LEFT_HEAVY = "left_heavy"   # 左偏分裂：左节点保留更多数据
    RIGHT_HEAVY = "right_heavy" # 右偏分裂：右节点保留更多数据
    ADAPTIVE = "adaptive"   # 自适应分裂：根据访问模式调整


class MergeStrategy(Enum):
    """
    Enumeration of different node merging strategies.
    """
    IMMEDIATE = "immediate"     # 立即合并：容量不足时立即合并
    LAZY = "lazy"              # 延迟合并：延迟合并直到严重不足
    THRESHOLD = "threshold"    # 阈值合并：基于利用率阈值合并


class StorageStrategy(Enum):
    """
    Enumeration of different storage strategies for async I/O.
    """
    MEMORY_ONLY = "memory_only"        # 仅内存存储
    WRITE_THROUGH = "write_through"    # 写透模式：同步写入磁盘
    WRITE_BACK = "write_back"          # 写回模式：异步批量写入
    LAZY_WRITE = "lazy_write"          # 延迟写入：定期异步写入


class SerializationFormat(Enum):
    """
    Enumeration of serialization formats for disk storage.
    """
    JSON = "json"              # JSON格式（可读性好）
    PICKLE = "pickle"          # Pickle格式（性能好）
    MSGPACK = "msgpack"        # MessagePack格式（紧凑）


class AsyncIOConfig:
    """
    Configuration for asynchronous I/O operations.
    """
    
    def __init__(
        self,
        storage_strategy: StorageStrategy = StorageStrategy.MEMORY_ONLY,
        serialization_format: SerializationFormat = SerializationFormat.PICKLE,
        storage_path: Optional[str] = None,
        max_concurrent_operations: int = 10,
        max_workers: int = 4,  # 新增：限制异步模式的工作线程数
        write_batch_size: int = 100,
        write_interval_seconds: float = 5.0,
        enable_compression: bool = True,
        backup_enabled: bool = False,
        backup_interval_seconds: float = 300.0,
        max_memory_cache_size: int = 1000000,  # 1MB
        enable_write_ahead_log: bool = False
    ):
        """
        Initialize async I/O configuration.
        
        Args:
            storage_strategy: Storage strategy to use
            serialization_format: Format for serialization
            storage_path: Path for disk storage
            max_concurrent_operations: Maximum concurrent async operations
            max_workers: Maximum number of worker threads for async operations
            write_batch_size: Number of operations to batch together
            write_interval_seconds: Interval for periodic writes
            enable_compression: Enable data compression
            backup_enabled: Enable automatic backups
            backup_interval_seconds: Interval for backups
            max_memory_cache_size: Maximum memory cache size in bytes
            enable_write_ahead_log: Enable write-ahead logging
        """
        self.storage_strategy = storage_strategy
        self.serialization_format = serialization_format
        self.storage_path = Path(storage_path) if storage_path else Path("./mbtree_data")
        self.max_concurrent_operations = max_concurrent_operations
        self.max_workers = max_workers
        self.write_batch_size = write_batch_size
        self.write_interval_seconds = write_interval_seconds
        self.enable_compression = enable_compression
        self.backup_enabled = backup_enabled
        self.backup_interval_seconds = backup_interval_seconds
        self.max_memory_cache_size = max_memory_cache_size
        self.enable_write_ahead_log = enable_write_ahead_log
        
        # Create storage directory if it doesn't exist
        if self.storage_strategy != StorageStrategy.MEMORY_ONLY:
            self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def __repr__(self) -> str:
        return (f"AsyncIOConfig(strategy={self.storage_strategy.value}, "
                f"format={self.serialization_format.value}, "
                f"path={self.storage_path})")


class QueryOperator(Enum):
    """
    Enumeration of query operators for expression-based queries.
    """
    # Comparison operators
    EQ = "=="          # Equal
    NE = "!="          # Not equal
    LT = "<"           # Less than
    LE = "<="          # Less than or equal
    GT = ">"           # Greater than
    GE = ">="          # Greater than or equal
    
    # String operators
    CONTAINS = "contains"      # String contains
    STARTSWITH = "startswith"  # String starts with
    ENDSWITH = "endswith"      # String ends with
    MATCHES = "matches"        # Regex match
    
    # Range operators
    BETWEEN = "between"        # Value between two bounds
    IN = "in"                 # Value in list
    NOT_IN = "not_in"         # Value not in list
    
    # Logical operators
    AND = "and"
    OR = "or"
    NOT = "not"
    
    # Special operators
    EXISTS = "exists"         # Key exists
    TYPE_IS = "type_is"       # Value type check


class QueryExpression(ABC):
    """
    Abstract base class for query expressions.
    """
    
    @abstractmethod
    def evaluate(self, key: Any, value: Any) -> bool:
        """
        Evaluate the expression against a key-value pair.
        
        Args:
            key: The key to evaluate
            value: The value to evaluate
            
        Returns:
            True if the expression matches, False otherwise
        """
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """Return string representation of the expression."""
        pass


class ComparisonExpression(QueryExpression):
    """
    Expression for comparison operations (==, !=, <, <=, >, >=).
    """
    
    def __init__(self, field: str, operator: QueryOperator, value: Any):
        """
        Initialize comparison expression.
        
        Args:
            field: Field to compare ('key' or 'value')
            operator: Comparison operator
            value: Value to compare against
        """
        self.field = field
        self.operator = operator
        self.value = value
        
        # Map operators to functions
        self.op_funcs = {
            QueryOperator.EQ: op.eq,
            QueryOperator.NE: op.ne,
            QueryOperator.LT: op.lt,
            QueryOperator.LE: op.le,
            QueryOperator.GT: op.gt,
            QueryOperator.GE: op.ge,
        }
    
    def evaluate(self, key: Any, value: Any) -> bool:
        """Evaluate comparison expression."""
        target = key if self.field == 'key' else value
        
        try:
            return self.op_funcs[self.operator](target, self.value)
        except (TypeError, ValueError):
            return False
    
    def __str__(self) -> str:
        return f"{self.field} {self.operator.value} {self.value}"


class StringExpression(QueryExpression):
    """
    Expression for string operations (contains, startswith, endswith, matches).
    """
    
    def __init__(self, field: str, operator: QueryOperator, pattern: str):
        """
        Initialize string expression.
        
        Args:
            field: Field to operate on ('key' or 'value')
            operator: String operator
            pattern: Pattern to match
        """
        self.field = field
        self.operator = operator
        self.pattern = pattern
        
        # Compile regex pattern if needed
        if operator == QueryOperator.MATCHES:
            try:
                self.compiled_pattern = re.compile(pattern)
            except re.error:
                raise ValueError(f"Invalid regex pattern: {pattern}")
    
    def evaluate(self, key: Any, value: Any) -> bool:
        """Evaluate string expression."""
        target = key if self.field == 'key' else value
        
        # Convert to string if not already
        if not isinstance(target, str):
            target = str(target)
        
        try:
            if self.operator == QueryOperator.CONTAINS:
                return self.pattern in target
            elif self.operator == QueryOperator.STARTSWITH:
                return target.startswith(self.pattern)
            elif self.operator == QueryOperator.ENDSWITH:
                return target.endswith(self.pattern)
            elif self.operator == QueryOperator.MATCHES:
                return bool(self.compiled_pattern.search(target))
            else:
                return False
        except (TypeError, AttributeError):
            return False
    
    def __str__(self) -> str:
        return f"{self.field} {self.operator.value} '{self.pattern}'"


class RangeExpression(QueryExpression):
    """
    Expression for range operations (between, in, not_in).
    """
    
    def __init__(self, field: str, operator: QueryOperator, values: Union[List[Any], Tuple[Any, Any]]):
        """
        Initialize range expression.
        
        Args:
            field: Field to operate on ('key' or 'value')
            operator: Range operator
            values: List of values or tuple for between operation
        """
        self.field = field
        self.operator = operator
        self.values = values
        
        if operator == QueryOperator.BETWEEN and len(values) != 2:
            raise ValueError("BETWEEN operator requires exactly 2 values")
    
    def evaluate(self, key: Any, value: Any) -> bool:
        """Evaluate range expression."""
        target = key if self.field == 'key' else value
        
        try:
            if self.operator == QueryOperator.BETWEEN:
                min_val, max_val = self.values
                return min_val <= target <= max_val
            elif self.operator == QueryOperator.IN:
                return target in self.values
            elif self.operator == QueryOperator.NOT_IN:
                return target not in self.values
            else:
                return False
        except (TypeError, ValueError):
            return False
    
    def __str__(self) -> str:
        if self.operator == QueryOperator.BETWEEN:
            return f"{self.field} between {self.values[0]} and {self.values[1]}"
        else:
            return f"{self.field} {self.operator.value} {self.values}"


class LogicalExpression(QueryExpression):
    """
    Expression for logical operations (and, or, not).
    """
    
    def __init__(self, operator: QueryOperator, *expressions: QueryExpression):
        """
        Initialize logical expression.
        
        Args:
            operator: Logical operator
            expressions: Sub-expressions to combine
        """
        self.operator = operator
        self.expressions = expressions
        
        if operator == QueryOperator.NOT and len(expressions) != 1:
            raise ValueError("NOT operator requires exactly 1 expression")
        elif operator in [QueryOperator.AND, QueryOperator.OR] and len(expressions) < 2:
            raise ValueError(f"{operator.value.upper()} operator requires at least 2 expressions")
    
    def evaluate(self, key: Any, value: Any) -> bool:
        """Evaluate logical expression."""
        if self.operator == QueryOperator.AND:
            return all(expr.evaluate(key, value) for expr in self.expressions)
        elif self.operator == QueryOperator.OR:
            return any(expr.evaluate(key, value) for expr in self.expressions)
        elif self.operator == QueryOperator.NOT:
            return not self.expressions[0].evaluate(key, value)
        else:
            return False
    
    def __str__(self) -> str:
        if self.operator == QueryOperator.NOT:
            return f"not ({self.expressions[0]})"
        else:
            op_str = f" {self.operator.value} "
            return f"({op_str.join(str(expr) for expr in self.expressions)})"


class SpecialExpression(QueryExpression):
    """
    Expression for special operations (exists, type_is).
    """
    
    def __init__(self, operator: QueryOperator, field: str = 'key', type_name: Optional[str] = None):
        """
        Initialize special expression.
        
        Args:
            operator: Special operator
            field: Field to operate on ('key' or 'value')
            type_name: Type name for type_is operator
        """
        self.operator = operator
        self.field = field
        self.type_name = type_name
        
        if operator == QueryOperator.TYPE_IS and not type_name:
            raise ValueError("TYPE_IS operator requires type_name")
    
    def evaluate(self, key: Any, value: Any) -> bool:
        """Evaluate special expression."""
        if self.operator == QueryOperator.EXISTS:
            # For EXISTS, we just check if we have a valid key-value pair
            return key is not None
        elif self.operator == QueryOperator.TYPE_IS:
            target = key if self.field == 'key' else value
            target_type = type(target).__name__
            return target_type == self.type_name
        else:
            return False
    
    def __str__(self) -> str:
        if self.operator == QueryOperator.EXISTS:
            return f"{self.field} exists"
        elif self.operator == QueryOperator.TYPE_IS:
            return f"{self.field} type_is {self.type_name}"
        else:
            return str(self.operator.value)


class QueryBuilder:
    """
    Builder class for constructing query expressions with a fluent interface.
    """
    
    def __init__(self):
        """Initialize query builder."""
        self.expressions = []
    
    # Comparison methods
    def key_eq(self, value: Any) -> 'QueryBuilder':
        """Add key equals condition."""
        self.expressions.append(ComparisonExpression('key', QueryOperator.EQ, value))
        return self
    
    def key_ne(self, value: Any) -> 'QueryBuilder':
        """Add key not equals condition."""
        self.expressions.append(ComparisonExpression('key', QueryOperator.NE, value))
        return self
    
    def key_lt(self, value: Any) -> 'QueryBuilder':
        """Add key less than condition."""
        self.expressions.append(ComparisonExpression('key', QueryOperator.LT, value))
        return self
    
    def key_le(self, value: Any) -> 'QueryBuilder':
        """Add key less than or equal condition."""
        self.expressions.append(ComparisonExpression('key', QueryOperator.LE, value))
        return self
    
    def key_gt(self, value: Any) -> 'QueryBuilder':
        """Add key greater than condition."""
        self.expressions.append(ComparisonExpression('key', QueryOperator.GT, value))
        return self
    
    def key_ge(self, value: Any) -> 'QueryBuilder':
        """Add key greater than or equal condition."""
        self.expressions.append(ComparisonExpression('key', QueryOperator.GE, value))
        return self
    
    def value_eq(self, value: Any) -> 'QueryBuilder':
        """Add value equals condition."""
        self.expressions.append(ComparisonExpression('value', QueryOperator.EQ, value))
        return self
    
    def value_ne(self, value: Any) -> 'QueryBuilder':
        """Add value not equals condition."""
        self.expressions.append(ComparisonExpression('value', QueryOperator.NE, value))
        return self
    
    def value_lt(self, value: Any) -> 'QueryBuilder':
        """Add value less than condition."""
        self.expressions.append(ComparisonExpression('value', QueryOperator.LT, value))
        return self
    
    def value_le(self, value: Any) -> 'QueryBuilder':
        """Add value less than or equal condition."""
        self.expressions.append(ComparisonExpression('value', QueryOperator.LE, value))
        return self
    
    def value_gt(self, value: Any) -> 'QueryBuilder':
        """Add value greater than condition."""
        self.expressions.append(ComparisonExpression('value', QueryOperator.GT, value))
        return self
    
    def value_ge(self, value: Any) -> 'QueryBuilder':
        """Add value greater than or equal condition."""
        self.expressions.append(ComparisonExpression('value', QueryOperator.GE, value))
        return self
    
    # String methods
    def key_contains(self, pattern: str) -> 'QueryBuilder':
        """Add key contains condition."""
        self.expressions.append(StringExpression('key', QueryOperator.CONTAINS, pattern))
        return self
    
    def key_startswith(self, pattern: str) -> 'QueryBuilder':
        """Add key starts with condition."""
        self.expressions.append(StringExpression('key', QueryOperator.STARTSWITH, pattern))
        return self
    
    def key_endswith(self, pattern: str) -> 'QueryBuilder':
        """Add key ends with condition."""
        self.expressions.append(StringExpression('key', QueryOperator.ENDSWITH, pattern))
        return self
    
    def key_matches(self, pattern: str) -> 'QueryBuilder':
        """Add key regex match condition."""
        self.expressions.append(StringExpression('key', QueryOperator.MATCHES, pattern))
        return self
    
    def value_contains(self, pattern: str) -> 'QueryBuilder':
        """Add value contains condition."""
        self.expressions.append(StringExpression('value', QueryOperator.CONTAINS, pattern))
        return self
    
    def value_startswith(self, pattern: str) -> 'QueryBuilder':
        """Add value starts with condition."""
        self.expressions.append(StringExpression('value', QueryOperator.STARTSWITH, pattern))
        return self
    
    def value_endswith(self, pattern: str) -> 'QueryBuilder':
        """Add value ends with condition."""
        self.expressions.append(StringExpression('value', QueryOperator.ENDSWITH, pattern))
        return self
    
    def value_matches(self, pattern: str) -> 'QueryBuilder':
        """Add value regex match condition."""
        self.expressions.append(StringExpression('value', QueryOperator.MATCHES, pattern))
        return self
    
    # Range methods
    def key_between(self, min_val: Any, max_val: Any) -> 'QueryBuilder':
        """Add key between condition."""
        self.expressions.append(RangeExpression('key', QueryOperator.BETWEEN, (min_val, max_val)))
        return self
    
    def key_in(self, values: List[Any]) -> 'QueryBuilder':
        """Add key in list condition."""
        self.expressions.append(RangeExpression('key', QueryOperator.IN, values))
        return self
    
    def key_not_in(self, values: List[Any]) -> 'QueryBuilder':
        """Add key not in list condition."""
        self.expressions.append(RangeExpression('key', QueryOperator.NOT_IN, values))
        return self
    
    def value_between(self, min_val: Any, max_val: Any) -> 'QueryBuilder':
        """Add value between condition."""
        self.expressions.append(RangeExpression('value', QueryOperator.BETWEEN, (min_val, max_val)))
        return self
    
    def value_in(self, values: List[Any]) -> 'QueryBuilder':
        """Add value in list condition."""
        self.expressions.append(RangeExpression('value', QueryOperator.IN, values))
        return self
    
    def value_not_in(self, values: List[Any]) -> 'QueryBuilder':
        """Add value not in list condition."""
        self.expressions.append(RangeExpression('value', QueryOperator.NOT_IN, values))
        return self
    
    # Special methods
    def key_exists(self) -> 'QueryBuilder':
        """Add key exists condition."""
        self.expressions.append(SpecialExpression(QueryOperator.EXISTS, 'key'))
        return self
    
    def key_type_is(self, type_name: str) -> 'QueryBuilder':
        """Add key type check condition."""
        self.expressions.append(SpecialExpression(QueryOperator.TYPE_IS, 'key', type_name))
        return self
    
    def value_type_is(self, type_name: str) -> 'QueryBuilder':
        """Add value type check condition."""
        self.expressions.append(SpecialExpression(QueryOperator.TYPE_IS, 'value', type_name))
        return self
    
    # Logical combinators
    def and_(self, *builders: 'QueryBuilder') -> 'QueryBuilder':
        """Combine expressions with AND."""
        all_expressions = self.expressions[:]
        for builder in builders:
            all_expressions.extend(builder.expressions)
        
        new_builder = QueryBuilder()
        if len(all_expressions) > 1:
            new_builder.expressions = [LogicalExpression(QueryOperator.AND, *all_expressions)]
        else:
            new_builder.expressions = all_expressions
        return new_builder
    
    def or_(self, *builders: 'QueryBuilder') -> 'QueryBuilder':
        """Combine expressions with OR."""
        all_expressions = self.expressions[:]
        for builder in builders:
            all_expressions.extend(builder.expressions)
        
        new_builder = QueryBuilder()
        if len(all_expressions) > 1:
            new_builder.expressions = [LogicalExpression(QueryOperator.OR, *all_expressions)]
        else:
            new_builder.expressions = all_expressions
        return new_builder
    
    def not_(self) -> 'QueryBuilder':
        """Negate the current expression."""
        new_builder = QueryBuilder()
        if len(self.expressions) == 1:
            new_builder.expressions = [LogicalExpression(QueryOperator.NOT, self.expressions[0])]
        elif len(self.expressions) > 1:
            combined = LogicalExpression(QueryOperator.AND, *self.expressions)
            new_builder.expressions = [LogicalExpression(QueryOperator.NOT, combined)]
        return new_builder
    
    def build(self) -> Optional[QueryExpression]:
        """Build the final query expression."""
        if not self.expressions:
            return None
        elif len(self.expressions) == 1:
            return self.expressions[0]
        else:
            return LogicalExpression(QueryOperator.AND, *self.expressions)


class QueryParser:
    """
    Parser for string-based query expressions.
    
    Supports syntax like:
    - "key > 10"
    - "value contains 'hello'"
    - "key between 5 and 15"
    - "(key > 10) and (value startswith 'test')"
    """
    
    def __init__(self):
        """Initialize query parser."""
        self.tokens = []
        self.position = 0
    
    def parse(self, query_string: str) -> Optional[QueryExpression]:
        """
        Parse a query string into a QueryExpression.
        
        Args:
            query_string: String representation of the query
            
        Returns:
            Parsed QueryExpression or None if parsing fails
        """
        try:
            self.tokens = self._tokenize(query_string)
            self.position = 0
            return self._parse_expression()
        except Exception as e:
            raise ValueError(f"Failed to parse query: {query_string}. Error: {str(e)}")
    
    def _tokenize(self, query_string: str) -> List[str]:
        """
        Enhanced tokenize method that properly handles quoted strings.
        Supports both single quotes (') and double quotes (") for string literals.
        This prevents injection attacks and allows querying text containing keywords.
        """
        import re
        
        tokens = []
        i = 0
        current_token = ""
        
        while i < len(query_string):
            char = query_string[i]
            
            # Handle quoted strings (both single and double quotes)
            if char in ['"', "'"]:
                quote_char = char
                # If we have accumulated a token, add it first
                if current_token.strip():
                    tokens.append(current_token.strip())
                    current_token = ""
                
                # Start collecting the quoted string
                i += 1  # Skip opening quote
                quoted_string = ""
                
                while i < len(query_string):
                    if query_string[i] == quote_char:
                        # Check for escaped quotes
                        if i + 1 < len(query_string) and query_string[i + 1] == quote_char:
                            # Double quote means escaped quote
                            quoted_string += quote_char
                            i += 2  # Skip both quotes
                        else:
                            # End of quoted string
                            break
                    elif query_string[i] == '\\' and i + 1 < len(query_string):
                        # Handle escape sequences
                        next_char = query_string[i + 1]
                        if next_char == 'n':
                            quoted_string += '\n'
                        elif next_char == 't':
                            quoted_string += '\t'
                        elif next_char == 'r':
                            quoted_string += '\r'
                        elif next_char == '\\':
                            quoted_string += '\\'
                        elif next_char in ['"', "'"]:
                            quoted_string += next_char
                        else:
                            # Unknown escape, keep as is
                            quoted_string += query_string[i:i+2]
                        i += 2
                    else:
                        quoted_string += query_string[i]
                        i += 1
                
                # Add the quoted string as a single token with special marker
                tokens.append(f"__QUOTED__{quoted_string}")
                
            # Handle special characters that need spacing
            elif char in '()[],:':
                # Add current token if any
                if current_token.strip():
                    tokens.append(current_token.strip())
                    current_token = ""
                # Add the special character
                tokens.append(char)
                
            # Handle whitespace
            elif char.isspace():
                # Add current token if any
                if current_token.strip():
                    tokens.append(current_token.strip())
                    current_token = ""
                    
            # Regular character
            else:
                current_token += char
                
            i += 1
        
        # Add final token if any
        if current_token.strip():
            tokens.append(current_token.strip())
        
        # Filter out empty tokens
        tokens = [token for token in tokens if token]
        
        return tokens
    
    def _parse_expression(self) -> Optional[QueryExpression]:
        """Parse a complete expression."""
        return self._parse_or_expression()
    
    def _parse_or_expression(self) -> Optional[QueryExpression]:
        """Parse OR expressions."""
        left = self._parse_and_expression()
        
        while self._current_token() == 'or':
            self._consume('or')
            right = self._parse_and_expression()
            left = LogicalExpression(QueryOperator.OR, left, right)
        
        return left
    
    def _parse_and_expression(self) -> Optional[QueryExpression]:
        """Parse AND expressions."""
        left = self._parse_not_expression()
        
        while self._current_token() == 'and':
            self._consume('and')
            right = self._parse_not_expression()
            left = LogicalExpression(QueryOperator.AND, left, right)
        
        return left
    
    def _parse_not_expression(self) -> Optional[QueryExpression]:
        """Parse NOT expressions."""
        if self._current_token() == 'not':
            self._consume('not')
            expr = self._parse_primary_expression()
            return LogicalExpression(QueryOperator.NOT, expr)
        
        return self._parse_primary_expression()
    
    def _parse_primary_expression(self) -> Optional[QueryExpression]:
        """Parse primary expressions (comparisons, ranges, etc.)."""
        if self._current_token() == '(':
            self._consume('(')
            expr = self._parse_expression()
            self._consume(')')
            return expr
        
        # Parse field operator value pattern
        field = self._consume_field()
        operator_str = self._consume_operator()
        
        # Handle different operator types
        if operator_str in ['==', '!=', '<', '<=', '>', '>=']:
            value = self._consume_value()
            op = QueryOperator(operator_str)
            return ComparisonExpression(field, op, value)
        
        elif operator_str in ['contains', 'startswith', 'endswith', 'matches']:
            pattern = self._consume_value()
            op = QueryOperator(operator_str)
            return StringExpression(field, op, str(pattern))
        
        elif operator_str == 'between':
            min_val = self._consume_value()
            self._consume('and')
            max_val = self._consume_value()
            return RangeExpression(field, QueryOperator.BETWEEN, (min_val, max_val))
        
        elif operator_str in ['in', 'not_in']:
            # Expect a list of values
            values = self._consume_list()
            op = QueryOperator(operator_str)
            return RangeExpression(field, op, values)
        
        elif operator_str == 'exists':
            return SpecialExpression(QueryOperator.EXISTS, field)
        
        elif operator_str == 'type_is':
            type_name = self._consume_value()
            return SpecialExpression(QueryOperator.TYPE_IS, field, str(type_name))
        
        else:
            raise ValueError(f"Unknown operator: {operator_str}")
    
    def _current_token(self) -> Optional[str]:
        """Get current token without consuming it."""
        if self.position < len(self.tokens):
            return self.tokens[self.position]
        return None
    
    def _consume(self, expected: Optional[str] = None) -> str:
        """Consume and return the current token."""
        if self.position >= len(self.tokens):
            raise ValueError("Unexpected end of query")
        
        token = self.tokens[self.position]
        self.position += 1
        
        if expected and token != expected:
            raise ValueError(f"Expected '{expected}', got '{token}'")
        
        return token
    
    def _consume_field(self) -> str:
        """Consume a field name (key or value)."""
        token = self._consume()
        if token not in ['key', 'value']:
            raise ValueError(f"Expected 'key' or 'value', got '{token}'")
        return token
    
    def _consume_operator(self) -> str:
        """Consume an operator token."""
        return self._consume()
    
    def _consume_value(self) -> Any:
        """
        Consume and convert a value token.
        Handles quoted strings, boolean values, None, and numeric conversions.
        """
        token = self._consume()
        
        # Handle quoted strings
        if token.startswith("__QUOTED__"):
            # Remove the special marker and return the raw string
            return token[10:]  # Remove "__QUOTED__" prefix
        
        # Handle boolean values
        if token.lower() in ['true', 'false']:
            return token.lower() == 'true'
        
        # Handle None/null values
        if token.lower() in ['none', 'null']:
            return None
        
        # Try numeric conversion
        try:
            # Check for float
            if '.' in token or 'e' in token.lower():
                return float(token)
            else:
                return int(token)
        except ValueError:
            # Return as string if conversion fails
            return token
    
    def _consume_list(self) -> List[Any]:
        """
        Consume a list of values in brackets.
        Handles quoted strings and various value types.
        """
        self._consume('[')
        values = []
        
        while self._current_token() != ']':
            if self._current_token() is None:
                raise ValueError("Unexpected end of query while parsing list")
            
            values.append(self._consume_value())
            
            # Check for comma or end of list
            if self._current_token() == ',':
                self._consume(',')
            elif self._current_token() != ']':
                raise ValueError(f"Expected ',' or ']' in list, got '{self._current_token()}'")
        
        self._consume(']')
        return values


class TreeConfig:
    """
    Configuration class for tree parameters.
    """
    
    def __init__(
        self,
        max_capacity: int = 64,
        min_capacity: Optional[int] = None,
        split_strategy: SplitStrategy = SplitStrategy.EVEN,
        merge_strategy: MergeStrategy = MergeStrategy.THRESHOLD,
        merge_threshold: float = 0.25,
        enable_compression: bool = False,
        cache_size: int = 1000,
        bulk_load_threshold: int = 100,
        async_io_config: Optional[AsyncIOConfig] = None,
        enable_async_mode: bool = False  # 新增：是否启用异步模式
    ):
        """
        Initialize tree configuration.
        
        Args:
            max_capacity: Maximum number of items per node
            min_capacity: Minimum number of items per node (auto-calculated if None)
            split_strategy: Strategy for splitting nodes
            merge_strategy: Strategy for merging nodes
            merge_threshold: Threshold for triggering merges (0.0 to 1.0)
            enable_compression: Enable node compression
            cache_size: Size of the node cache
            bulk_load_threshold: Threshold for bulk loading operations
            async_io_config: Configuration for async I/O operations
            enable_async_mode: Whether to enable async mode for the tree
        """
        self.max_capacity = max_capacity
        self.min_capacity = min_capacity if min_capacity is not None else max_capacity // 4
        self.split_strategy = split_strategy
        self.merge_strategy = merge_strategy
        self.merge_threshold = merge_threshold
        self.enable_compression = enable_compression
        self.cache_size = cache_size
        self.bulk_load_threshold = bulk_load_threshold
        self.async_io_config = async_io_config
        self.enable_async_mode = enable_async_mode
        
        # Auto-enable async mode if async_io_config is provided
        if async_io_config is not None:
            self.enable_async_mode = True
        
        # Validation
        if self.max_capacity <= 0:
            raise ValueError("max_capacity must be positive")
        if self.min_capacity < 0:
            raise ValueError("min_capacity must be non-negative")
        if self.min_capacity >= self.max_capacity:
            raise ValueError("min_capacity must be less than max_capacity")
        if not 0.0 <= self.merge_threshold <= 1.0:
            raise ValueError("merge_threshold must be between 0.0 and 1.0")
    
    def __repr__(self) -> str:
        return (f"TreeConfig(max={self.max_capacity}, min={self.min_capacity}, "
                f"split={self.split_strategy.value}, merge={self.merge_strategy.value}, "
                f"async_io={self.async_io_config.storage_strategy.value})")


class AsyncStorageManager:
    """
    Manages asynchronous I/O operations for tree persistence.
    """
    
    def __init__(self, config: AsyncIOConfig):
        """Initialize async storage manager."""
        self.config = config
        self.running = False
        self.write_queue = asyncio.Queue()
        self.write_lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        
        # Initialize WAL if enabled
        self.wal_path = None
        if config.enable_write_ahead_log:
            self.wal_path = config.storage_path / "wal.log"
        
        # Background tasks
        self.write_task = None
        self.backup_task = None
        
        # Statistics
        self.stats = {
            'writes_completed': 0,
            'reads_completed': 0,
            'write_errors': 0,
            'read_errors': 0,
            'bytes_written': 0,
            'bytes_read': 0,
            'last_write_time': 0,
            'last_backup_time': 0
        }
    
    async def start(self):
        """Start the async storage manager."""
        if self.running:
            return
        
        self.running = True
        
        # Start background tasks
        if self.config.storage_strategy in [StorageStrategy.WRITE_BACK, StorageStrategy.LAZY_WRITE]:
            task = asyncio.create_task(self._periodic_write_task())
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
        
        if self.config.backup_enabled:
            task = asyncio.create_task(self._periodic_backup_task())
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
    
    async def stop(self):
        """Stop the storage manager and clean up resources."""
        if not self.running:
            return
            
        self.running = False
        
        # Cancel background tasks
        if self.write_task and not self.write_task.done():
            self.write_task.cancel()
            try:
                await self.write_task
            except asyncio.CancelledError:
                pass
        
        if self.backup_task and not self.backup_task.done():
            self.backup_task.cancel()
            try:
                await self.backup_task
            except asyncio.CancelledError:
                pass
        
        # Flush any remaining writes
        await self._flush_write_batch()
        
        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True)
    
    async def serialize_data(self, data: Any) -> bytes:
        """
        Serialize data according to configuration.
        
        Args:
            data: Data to serialize
            
        Returns:
            Serialized bytes
        """
        def _serialize():
            if self.config.serialization_format == SerializationFormat.JSON:
                # Convert to JSON-serializable format
                json_data = self._convert_to_json_serializable(data)
                serialized = json.dumps(json_data, ensure_ascii=False).encode('utf-8')
            elif self.config.serialization_format == SerializationFormat.PICKLE:
                serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            else:  # MSGPACK
                try:
                    import msgpack
                    serialized = msgpack.packb(data, use_bin_type=True)
                except ImportError:
                    # Fallback to pickle if msgpack not available
                    serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Apply compression if enabled
            if self.config.enable_compression:
                import gzip
                serialized = gzip.compress(serialized)
            
            return serialized
        
        # Run serialization in thread pool to avoid blocking
        return await asyncio.get_event_loop().run_in_executor(self.executor, _serialize)
    
    async def deserialize_data(self, data: bytes) -> Any:
        """
        Deserialize data according to configuration.
        
        Args:
            data: Bytes to deserialize
            
        Returns:
            Deserialized data
        """
        def _deserialize():
            # Decompress if needed
            if self.config.enable_compression:
                import gzip
                data_to_deserialize = gzip.decompress(data)
            else:
                data_to_deserialize = data
            
            if self.config.serialization_format == SerializationFormat.JSON:
                json_data = json.loads(data_to_deserialize.decode('utf-8'))
                return self._convert_from_json_serializable(json_data)
            elif self.config.serialization_format == SerializationFormat.PICKLE:
                return pickle.loads(data_to_deserialize)
            else:  # MSGPACK
                try:
                    import msgpack
                    return msgpack.unpackb(data_to_deserialize, raw=False)
                except ImportError:
                    return pickle.loads(data_to_deserialize)
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, _deserialize)
    
    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {str(k): self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return {"__set__": list(obj)}
        elif hasattr(obj, '__dict__'):
            return {"__class__": obj.__class__.__name__, "__dict__": obj.__dict__}
        else:
            return obj
    
    def _convert_from_json_serializable(self, obj: Any) -> Any:
        """Convert from JSON-serializable format back to original."""
        if isinstance(obj, dict):
            if "__set__" in obj:
                return set(obj["__set__"])
            elif "__class__" in obj:
                # Simple class reconstruction - limited functionality
                return obj["__dict__"]
            else:
                return {k: self._convert_from_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_from_json_serializable(item) for item in obj]
        else:
            return obj
    
    async def write_data(self, key: str, data: Any, immediate: bool = False) -> None:
        """
        Write data asynchronously.
        
        Args:
            key: Storage key
            data: Data to write
            immediate: Whether to write immediately or batch
        """
        if self.config.storage_strategy == StorageStrategy.MEMORY_ONLY:
            return
        
        write_operation = {
            'key': key,
            'data': data,
            'timestamp': asyncio.get_event_loop().time(),
            'operation': 'write'
        }
        
        # Write-ahead logging
        if self.wal_path:
            await self._write_to_wal(write_operation)
        
        if immediate or self.config.storage_strategy == StorageStrategy.WRITE_THROUGH:
            await self._perform_write(write_operation)
        else:
            await self.write_queue.put(write_operation)
    
    async def read_data(self, key: str) -> Optional[Any]:
        """
        Read data asynchronously.
        
        Args:
            key: Storage key
            
        Returns:
            Data if found, None otherwise
        """
        if self.config.storage_strategy == StorageStrategy.MEMORY_ONLY:
            return None
        
        try:
            file_path = self.config.storage_path / f"{key}.{self.config.serialization_format.value}"
            
            if not file_path.exists():
                return None
            
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(file_path, 'rb') as f:
                    data_bytes = await f.read()
            else:
                # Fallback to synchronous file operations
                def _sync_read():
                    with open(file_path, 'rb') as f:
                        return f.read()
                
                data_bytes = await asyncio.get_event_loop().run_in_executor(self.executor, _sync_read)
            
            data = await self.deserialize_data(data_bytes)
            
            self.stats['reads_completed'] += 1
            self.stats['bytes_read'] += len(data_bytes)
            
            return data
            
        except Exception as e:
            self.stats['read_errors'] += 1
            print(f"Error reading data for key {key}: {e}")
            return None
    
    async def _perform_write(self, operation: Dict[str, Any]) -> None:
        """Perform actual write operation."""
        try:
            key = operation['key']
            data = operation['data']
            
            # Serialize data
            serialized_data = await self.serialize_data(data)
            
            # Write to file
            file_path = self.config.storage_path / f"{key}.{self.config.serialization_format.value}"
            
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(file_path, 'wb') as f:
                    await f.write(serialized_data)
            else:
                # Fallback to synchronous file operations
                def _sync_write():
                    with open(file_path, 'wb') as f:
                        f.write(serialized_data)
                
                await asyncio.get_event_loop().run_in_executor(self.executor, _sync_write)
            
            self.stats['writes_completed'] += 1
            self.stats['bytes_written'] += len(serialized_data)
            self.stats['last_write_time'] = asyncio.get_event_loop().time()
            
        except Exception as e:
            self.stats['write_errors'] += 1
            print(f"Error writing data for key {operation['key']}: {e}")
    
    async def _write_to_wal(self, operation: Dict[str, Any]) -> None:
        """Write operation to write-ahead log."""
        if not self.wal_path:
            return
        
        try:
            wal_entry = {
                'timestamp': operation['timestamp'],
                'operation': operation['operation'],
                'key': operation['key'],
                'data_hash': hash(str(operation['data']))  # Simple hash for integrity
            }
            
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(self.wal_path, 'a') as f:
                    await f.write(json.dumps(wal_entry) + '\n')
            else:
                # Fallback to synchronous file operations
                def _sync_wal_write():
                    with open(self.wal_path, 'a') as f:
                        f.write(json.dumps(wal_entry) + '\n')
                
                await asyncio.get_event_loop().run_in_executor(self.executor, _sync_wal_write)
                
        except Exception as e:
            print(f"Error writing to WAL: {e}")
    
    async def _periodic_write_task(self) -> None:
        """Background task for periodic writes."""
        while self.running:
            try:
                await asyncio.sleep(self.config.write_interval_seconds)
                await self._flush_write_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in periodic write task: {e}")
    
    async def _flush_write_batch(self) -> None:
        """Flush pending write operations."""
        async with self.write_lock:
            # Collect pending operations
            pending_operations = []
            while not self.write_queue.empty():
                try:
                    operation = self.write_queue.get_nowait()
                    pending_operations.append(operation)
                    if len(pending_operations) >= self.config.write_batch_size:
                        break
                except asyncio.QueueEmpty:
                    break
            
            # Perform batch writes
            if pending_operations:
                write_tasks = [self._perform_write(op) for op in pending_operations]
                await asyncio.gather(*write_tasks, return_exceptions=True)
    
    async def _periodic_backup_task(self) -> None:
        """Background task for periodic backups."""
        while self.running:
            try:
                await asyncio.sleep(self.config.backup_interval_seconds)
                await self._create_backup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in periodic backup task: {e}")
    
    async def _create_backup(self) -> None:
        """Create a backup of the current data."""
        try:
            import shutil
            import datetime
            
            backup_dir = self.config.storage_path / "backups"
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"backup_{timestamp}"
            
            def _copy_files():
                shutil.copytree(self.config.storage_path, backup_path, 
                              ignore=shutil.ignore_patterns("backups", "*.log"))
            
            await asyncio.get_event_loop().run_in_executor(self.executor, _copy_files)
            
            self.stats['last_backup_time'] = asyncio.get_event_loop().time()
            print(f"Backup created: {backup_path}")
            
        except Exception as e:
            print(f"Error creating backup: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage manager statistics."""
        return self.stats.copy()


class NodeStats:
    """
    Statistics tracking for tree nodes.
    """
    
    def __init__(self):
        self.access_count = 0
        self.split_count = 0
        self.merge_count = 0
        self.last_access_time = 0
        self.hot_keys = set()  # 热点键集合
    
    def record_access(self, key: Any = None) -> None:
        """Record an access to this node."""
        import time
        self.access_count += 1
        self.last_access_time = time.time()
        if key is not None:
            self.hot_keys.add(key)
            # 限制热点键数量
            if len(self.hot_keys) > 10:
                self.hot_keys.pop()
    
    def record_split(self) -> None:
        """Record a split operation."""
        self.split_count += 1
    
    def record_merge(self) -> None:
        """Record a merge operation."""
        self.merge_count += 1
    
    @property
    def is_hot(self) -> bool:
        """Check if this is a frequently accessed node."""
        return self.access_count > 100


class OptimizedHeap:
    """
    An optimized heap implementation that maintains sorted order efficiently.
    
    This replaces the original Heap class with better performance characteristics.
    """
    
    def __init__(self, initial_list: Optional[List] = None):
        """Initialize the heap with an optional initial list."""
        self.items = initial_list.copy() if initial_list else []
        heapq.heapify(self.items)
    
    def push(self, item: Any) -> None:
        """Add an item to the heap."""
        heapq.heappush(self.items, item)
    
    def pop(self) -> Any:
        """Remove and return the smallest item from the heap."""
        if not self.items:
            raise IndexError("pop from empty heap")
        return heapq.heappop(self.items)
    
    def peek(self) -> Any:
        """Return the smallest item without removing it."""
        if not self.items:
            raise IndexError("peek from empty heap")
        return self.items[0]
    
    def is_empty(self) -> bool:
        """Check if the heap is empty."""
        return len(self.items) == 0
    
    def __len__(self) -> int:
        """Return the number of items in the heap."""
        return len(self.items)
    
    def __iter__(self):
        """Iterate through heap items in sorted order."""
        # Create a copy to avoid modifying the original heap
        temp_items = self.items.copy()
        while temp_items:
            yield heapq.heappop(temp_items)
    
    def __repr__(self) -> str:
        """Return string representation of the heap."""
        return f"OptimizedHeap({self.items})"


class Node:
    """
    A node class for use in LinkedListHeap.
    
    Supports comparison operations for heap ordering.
    """
    
    def __init__(self, value: Any):
        self.value = value
        self.next: Optional['Node'] = None

    def __lt__(self, other: 'Node') -> bool:
        """Compare nodes based on their values."""
        return self.value < other.value
    
    def __eq__(self, other: 'Node') -> bool:
        """Check equality based on values."""
        return self.value == other.value
    
    def __repr__(self) -> str:
        """Return string representation of the node."""
        return f"Node({self.value})"


class LinkedListHeap:
    """
    A heap implementation that maintains a linked list structure.
    
    Combines heap operations with linked list functionality.
    """
    
    def __init__(self):
        """Initialize an empty LinkedListHeap."""
        self.head: Optional[Node] = None
        self.heap: List[Node] = []

    def push(self, value: Any) -> None:
        """Add a value to the heap."""
        new_node = Node(value)
        heapq.heappush(self.heap, new_node)
        self._update_head()

    def pop(self) -> Any:
        """Remove and return the smallest value from the heap."""
        if not self.heap:
            raise IndexError("pop from empty heap")
        node = heapq.heappop(self.heap)
        self._update_head()
        return node.value

    def peek(self) -> Any:
        """Return the smallest value without removing it."""
        if not self.heap:
            raise IndexError("peek from empty heap")
        return self.heap[0].value

    def _update_head(self) -> None:
        """Update the head pointer to the smallest element."""
        self.head = self.heap[0] if self.heap else None

    def is_empty(self) -> bool:
        """Check if the heap is empty."""
        return len(self.heap) == 0

    def __len__(self) -> int:
        """Return the number of elements in the heap."""
        return len(self.heap)

    def __str__(self) -> str:
        """Return string representation of the heap."""
        if not self.heap:
            return "LinkedListHeap: []"
        
        elements = []
        temp_heap = self.heap.copy()
        while temp_heap:
            node = heapq.heappop(temp_heap)
            elements.append(str(node.value))
        return "LinkedListHeap: [" + ", ".join(elements) + "]"


class TreeNode:
    """
    A node in the multi-way tree structure.
    
    Can be either a leaf node (containing data) or an internal node (containing children).
    Enhanced with statistics tracking and configuration support.
    """
    
    def __init__(self, tree: 'Tree', key: Any, *, is_leaf: bool = False, 
                 parent: Optional['TreeNode'] = None):
        """
        Initialize a tree node.
        
        Args:
            tree: Reference to the parent tree
            key: The key identifying this node
            is_leaf: Whether this is a leaf node
            parent: Parent node reference
        """
        self.tree = tree
        self.parent = parent
        self.key = key
        self.is_leaf = is_leaf
        self.stats = NodeStats()
        
        if is_leaf:
            self.data: Dict[Any, Any] = {}
            # Note: Removed heap storage as it's not essential for tree operations
            # and causes comparison issues with TreeNode objects
        else:
            self.children: Dict[Any, 'TreeNode'] = {}
    
    def get(self, path: List[Any]) -> Optional['TreeNode']:
        """
        Navigate through the tree using a path.
        
        Args:
            path: List of keys/special commands to navigate
            
        Returns:
            The target node or None if not found
        """
        current = self
        
        for step in path:
            if current is None or not isinstance(current, TreeNode):
                return None
            
            if step == '/':
                current = self.tree.root
            elif step == '__parent__':
                current = current.parent
            else:
                if current.is_leaf:
                    # For leaf nodes, get data
                    current = current.data.get(step)
                else:
                    # For internal nodes, get children
                    current = current.children.get(step)
        
        return current
    
    def move(self, target_path: Union[List[Any], 'TreeNode']) -> bool:
        """
        Move this node to a new parent.
        
        Args:
            target_path: Path to new parent or parent node directly
            
        Returns:
            True if move was successful, False otherwise
        """
        if self.parent is None:
            return False
            
        # Remove from current parent
        if self.key in self.parent.children:
            del self.parent.children[self.key]
        
        # Find new parent
        if isinstance(target_path, TreeNode):
            new_parent = target_path
        else:
            new_parent = self.get(target_path)
        
        if new_parent is None or new_parent.is_leaf:
            return False
        
        # Add to new parent
        self.parent = new_parent
        new_parent.children[self.key] = self
        return True
    
    def move_data(self, data_key: Any, target_leaf: Union['TreeNode', List[Any]]) -> bool:
        """
        Move data from this leaf to another leaf.
        
        Args:
            data_key: Key of data to move
            target_leaf: Target leaf node or path to it
            
        Returns:
            True if move was successful, False otherwise
        """
        if not self.is_leaf:
            return False
            
        if data_key not in self.data:
            raise KeyError(f'Key not found: {data_key}')
        
        # Resolve target leaf
        if isinstance(target_leaf, TreeNode):
            target = target_leaf
        else:
            target = self.get(target_leaf)
        
        if target is None or not target.is_leaf:
            return False
        
        # Move the data
        data = self.data.pop(data_key)
        target.data[data_key] = data
        return True
            
    @property
    def mean_coverage(self) -> float:
        """
        Calculate the mean of keys in this node.
        
        Returns:
            Mean value of keys, or 0 if empty
        """
        try:
            import numpy as np
            if self.is_leaf:
                keys = list(self.data.keys())
            else:
                keys = list(self.children.keys())
            
            return float(np.mean(keys)) if keys else 0.0
        except ImportError:
            # Fallback if numpy is not available
            if self.is_leaf:
                keys = list(self.data.keys())
            else:
                keys = list(self.children.keys())
            
            return sum(keys) / len(keys) if keys else 0.0
    
    @property
    def coverage(self) -> Union[Dict[Any, Any], Dict[Any, 'TreeNode']]:
        """Get the data or children dictionary."""
        return self.data if self.is_leaf else self.children
    
    @property
    def utilization(self) -> float:
        """Calculate the utilization ratio of this node."""
        return len(self.coverage) / self.tree.config.max_capacity
    
    @property
    def is_underutilized(self) -> bool:
        """Check if node is underutilized based on configuration."""
        if self.tree.config.merge_strategy == MergeStrategy.THRESHOLD:
            return self.utilization < self.tree.config.merge_threshold
        elif self.tree.config.merge_strategy == MergeStrategy.IMMEDIATE:
            return len(self.coverage) < self.tree.config.min_capacity
        else:  # LAZY
            return len(self.coverage) < self.tree.config.min_capacity // 2
    
    def access(self, key: Any = None) -> None:
        """Record access to this node for statistics."""
        self.stats.record_access(key)
    
    def should_compress(self) -> bool:
        """Determine if this node should be compressed."""
        return (self.tree.config.enable_compression and 
                len(self.coverage) > self.tree.config.max_capacity * 0.8 and
                not self.stats.is_hot)
    
    def get_split_point(self, items: List[tuple]) -> int:
        """
        Determine the optimal split point based on strategy.
        
        Args:
            items: List of (key, value) tuples to split
            
        Returns:
            Index to split at
        """
        length = len(items)
        strategy = self.tree.config.split_strategy
        
        if strategy == SplitStrategy.EVEN:
            return length // 2
        elif strategy == SplitStrategy.LEFT_HEAVY:
            return int(length * 0.6)  # 60% to left
        elif strategy == SplitStrategy.RIGHT_HEAVY:
            return int(length * 0.4)  # 40% to left
        elif strategy == SplitStrategy.ADAPTIVE:
            # 基于访问模式调整
            if self.stats.is_hot:
                # 热点节点倾向于均匀分裂
                return length // 2
            else:
                # 冷节点倾向于左偏分裂
                return int(length * 0.6)
        else:
            return length // 2
    
    def __repr__(self) -> str:
        """Return string representation of the node."""
        node_type = "Leaf" if self.is_leaf else "Internal"
        size = len(self.coverage)
        util = f"{self.utilization:.1%}"
        return f"TreeNode({node_type}, key={self.key}, size={size}, util={util})"
    
    def __lt__(self, other: 'TreeNode') -> bool:
        """Compare nodes based on their keys for heap operations."""
        if not isinstance(other, TreeNode):
            return NotImplemented
        return self.key < other.key
    
    def __eq__(self, other: 'TreeNode') -> bool:
        """Check equality based on keys."""
        if not isinstance(other, TreeNode):
            return NotImplemented
        return self.key == other.key
    
    def __hash__(self) -> int:
        """Make TreeNode hashable."""
        return hash((id(self.tree), self.key, self.is_leaf))


class Tree:
    """
    A multi-way tree implementation similar to B-trees.
    
    Enhanced with configurable parameters, optimized algorithms for large-scale data,
    and asynchronous I/O support for disk operations.
    Supports efficient insertion, splitting, and rebalancing operations.
    """
    
    def __new__(cls, config: Optional[TreeConfig] = None):
        """
        Create a new Tree instance, automatically wrapping with AsyncTreeWrapper if async mode is enabled.
        
        Args:
            config: Tree configuration
            
        Returns:
            Tree instance or AsyncTreeWrapper if async mode is enabled
        """
        # Create the actual tree instance
        instance = super().__new__(cls)
        
        # If async mode is enabled in config, wrap with AsyncTreeWrapper
        if config and config.enable_async_mode:
            # Initialize the tree first
            instance.__init__(config)
            # Return wrapped version
            return AsyncTreeWrapper(instance)
        
        return instance
    
    def __init__(self, config: Optional[TreeConfig] = None):
        """
        Initialize the tree with given configuration.
        
        Args:
            config: Tree configuration
        """
        self.config = config or TreeConfig()
        self.root = TreeNode(self, key=None, is_leaf=True)
        self._node_cache = {}
        self._bulk_buffer = {}
        
        # Initialize async storage manager only if async_io_config is provided
        self.storage_manager = None
        if self.config.async_io_config is not None:
            self.storage_manager = AsyncStorageManager(self.config.async_io_config)
        
        # Statistics tracking
        self._stats = {
            'operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'splits': 0,
            'merges': 0,
            'redistributions': 0,
            'bulk_operations': 0,
        }
    
    def insert(self, key: int, value: Any) -> None:
        """
        Insert a key-value pair into the tree.
        Enhanced with smart insertion strategy and bulk loading support.
        
        Args:
            key: The key to insert
            value: The value to associate with the key
        """
        # Smart insertion strategy: insert directly on first insertion
        if len(self._bulk_buffer) == 0 and self.size() == 0:
            self._single_insert(key, value)
            return
        
        # Use bulk loading for subsequent insertions
        if len(self._bulk_buffer) < self.config.bulk_load_threshold:
            self._bulk_buffer[key] = value  # Store as key-value pair in dict
        else:
            # Flush buffer and insert new item
            self._bulk_insert()
            self._single_insert(key, value)
        
        self._stats['operations'] += 1
    
    async def async_insert(self, key: int, value: Any, persist: bool = True) -> None:
        """
        Asynchronously insert a key-value pair.
        
        Args:
            key: Key to insert
            value: Value to insert
            persist: Whether to persist to disk
        """
        # Perform in-memory insertion
        self.insert(key, value)
        
        # Persist to disk if requested and storage manager is available
        if persist and self.storage_manager is not None and self.config.async_io_config.storage_strategy != StorageStrategy.MEMORY_ONLY:
            await self.storage_manager.write_data(f"data_{key}", {'key': key, 'value': value})
    
    async def async_insert_batch(self, items: List[Tuple[Any, Any]], persist: bool = True) -> None:
        """
        Asynchronously insert multiple key-value pairs.
        
        Args:
            items: List of (key, value) tuples to insert
            persist: Whether to persist to disk
        """
        # Perform in-memory insertions
        for key, value in items:
            self.insert(key, value)
        
        # Persist to disk if requested and storage manager is available
        if persist and self.storage_manager is not None and self.config.async_io_config.storage_strategy != StorageStrategy.MEMORY_ONLY:
            write_tasks = []
            for key, value in items:
                task = self.storage_manager.write_data(f"data_{key}", {'key': key, 'value': value})
                write_tasks.append(task)
            
            # Execute writes concurrently
            await asyncio.gather(*write_tasks, return_exceptions=True)
    
    async def async_search(self, key: Any, check_disk: bool = True) -> Optional[Any]:
        """
        Asynchronously search for a key in the tree.
        
        Args:
            key: The key to search for
            check_disk: Whether to check disk if not found in memory
            
        Returns:
            The value if found, None otherwise
        """
        # First check memory
        result = self.search(key)
        if result is not None:
            return result
        
        # Check disk if enabled, storage manager is available, and not found in memory
        if (check_disk and self.storage_manager is not None and 
            self.config.async_io_config.storage_strategy != StorageStrategy.MEMORY_ONLY):
            disk_data = await self.storage_manager.read_data(f"data_{key}")
            if disk_data:
                return disk_data.get('value')
        
        return None
    
    async def async_query(self, expression: Union[QueryExpression, str, QueryBuilder], 
                         check_disk: bool = True) -> List[Tuple[Any, Any]]:
        """
        Asynchronously perform a query on the tree.
        
        Args:
            expression: Query expression
            check_disk: Whether to include disk data in results
            
        Returns:
            List of (key, value) tuples matching the query
        """
        # Get memory results
        memory_results = self.query(expression)
        
        if (not check_disk or self.storage_manager is None or 
            self.config.async_io_config.storage_strategy == StorageStrategy.MEMORY_ONLY):
            return memory_results
        
        # TODO: Implement disk query - would need to scan disk files
        # For now, return memory results
        return memory_results
    
    async def async_delete(self, key: Any, persist: bool = True) -> bool:
        """
        Asynchronously delete a key from the tree.
        
        Args:
            key: The key to delete
            persist: Whether to delete from disk
            
        Returns:
            True if deleted, False if not found
        """
        # Delete from memory
        memory_deleted = self.delete(key)
        
        # Delete from disk if requested and storage manager is available
        if (persist and self.storage_manager is not None and 
            self.config.async_io_config.storage_strategy != StorageStrategy.MEMORY_ONLY):
            try:
                # Remove from disk storage
                file_path = self.storage_manager.config.storage_path / f"data_{key}"
                if file_path.exists():
                    file_path.unlink()
            except Exception:
                pass  # Ignore disk deletion errors
        
        return memory_deleted
    
    async def async_save_tree(self, filepath: str) -> None:
        """
        Asynchronously save the entire tree to disk.
        
        Args:
            filepath: Path to save the tree
        """
        if self.storage_manager is None:
            raise RuntimeError("Storage manager not available. Enable async_io_config to use this feature.")
            
        tree_data = {
            'config': {
                'max_capacity': self.config.max_capacity,
                'min_capacity': self.config.min_capacity,
                'split_strategy': self.config.split_strategy.value,
                'merge_strategy': self.config.merge_strategy.value,
                'merge_threshold': self.config.merge_threshold,
                'enable_compression': self.config.enable_compression,
                'cache_size': self.config.cache_size,
                'bulk_load_threshold': self.config.bulk_load_threshold
            },
            'data': self.inorder_traversal(),
            'stats': self._stats
        }
        
        await self.storage_manager.write_data(f"tree_backup_{filepath}", tree_data, immediate=True)
    
    async def async_load_tree(self, filepath: str) -> None:
        """
        Asynchronously load a tree from disk.
        
        Args:
            filepath: Path to load the tree from
        """
        if self.storage_manager is None:
            raise RuntimeError("Storage manager not available. Enable async_io_config to use this feature.")
            
        tree_data = await self.storage_manager.read_data(f"tree_backup_{filepath}")
        if not tree_data:
            raise FileNotFoundError(f"Tree backup not found: {filepath}")
        
        # Restore data
        for key, value in tree_data['data']:
            self.insert(key, value)
        
        # Restore stats
        if 'stats' in tree_data:
            self._stats.update(tree_data['stats'])
    
    async def async_backup(self, backup_name: Optional[str] = None) -> str:
        """
        Create an asynchronous backup of the tree.
        
        Args:
            backup_name: Optional name for the backup
            
        Returns:
            The backup identifier
        """
        if backup_name is None:
            import time
            backup_name = f"backup_{int(time.time())}"
        
        await self.async_save_tree(backup_name)
        return backup_name
    
    def start_async_operations(self) -> None:
        """Start async operations (storage manager)."""
        if self.storage_manager is not None:
            # Create a task to start the storage manager
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.storage_manager.start())
            except RuntimeError:
                # No event loop running, will start manually when needed
                pass

    def stop_async_operations(self) -> None:
        """Stop async operations (storage manager)."""
        if self.storage_manager is not None:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.storage_manager.stop())
            except RuntimeError:
                # No event loop running
                pass
    
    def query(self, expression: Union[QueryExpression, str, QueryBuilder]) -> List[Tuple[Any, Any]]:
        """
        Query the tree using a query expression.
        Enhanced to include bulk buffer data and handle duplicates properly.
        
        Args:
            expression: Query expression (QueryExpression, string, or QueryBuilder)
            
        Returns:
            List of (key, value) tuples matching the query
        """
        # Convert input to QueryExpression
        if isinstance(expression, str):
            parser = QueryParser()
            query_expr = parser.parse(expression)
        elif isinstance(expression, QueryBuilder):
            query_expr = expression.build()
        elif isinstance(expression, QueryExpression):
            query_expr = expression
        else:
            raise ValueError("Invalid query expression type")
        
        if query_expr is None:
            return []
        
        # Use a dictionary to track unique results (key as identifier)
        result_dict = {}
        
        # First search in bulk buffer
        for key, value in self._bulk_buffer.items():
            if query_expr.evaluate(key, value):
                result_dict[key] = (key, value)
        
        # Then search in tree
        results_list = []
        self._query_helper(self.root, query_expr, results_list)
        
        # Add tree results to dict (automatically handles duplicates by key)
        for key, value in results_list:
            result_dict[key] = (key, value)
        
        # Convert back to list and return
        return list(result_dict.values())
    
    def _query_helper(self, node: TreeNode, expression: QueryExpression, results: List[Tuple[Any, Any]]) -> None:
        """
        Helper method for recursive querying.
        
        Args:
            node: Current node to search
            expression: Query expression to evaluate
            results: List to collect matching results
        """
        if node.is_leaf:
            # Evaluate expression against all data in leaf
            for key, value in node.data.items():
                if expression.evaluate(key, value):
                    results.append((key, value))
                    node.access(key)  # Record access for statistics
        else:
            # Recursively search children
            for child in node.children.values():
                self._query_helper(child, expression, results)
    
    def query_keys(self, expression: Union[QueryExpression, str, QueryBuilder]) -> List[Any]:
        """
        Query the tree and return only keys.
        
        Args:
            expression: Query expression
            
        Returns:
            List of keys matching the query
        """
        results = self.query(expression)
        return [key for key, _ in results]
    
    def query_values(self, expression: Union[QueryExpression, str, QueryBuilder]) -> List[Any]:
        """
        Query the tree and return only values.
        
        Args:
            expression: Query expression
            
        Returns:
            List of values matching the query
        """
        results = self.query(expression)
        return [value for _, value in results]
    
    def query_count(self, expression: Union[QueryExpression, str, QueryBuilder]) -> int:
        """
        Count items matching a query without returning them.
        
        Args:
            expression: Query expression
            
        Returns:
            Number of matching items
        """
        return len(self.query(expression))
    
    def query_exists(self, expression: Union[QueryExpression, str, QueryBuilder]) -> bool:
        """
        Check if any items match a query.
        Enhanced to include bulk buffer data.
        
        Args:
            expression: Query expression
            
        Returns:
            True if any items match, False otherwise
        """
        # Convert input to QueryExpression
        if isinstance(expression, str):
            parser = QueryParser()
            query_expr = parser.parse(expression)
        elif isinstance(expression, QueryBuilder):
            query_expr = expression.build()
        elif isinstance(expression, QueryExpression):
            query_expr = expression
        else:
            raise ValueError("Invalid query expression type")
        
        if query_expr is None:
            return False
        
        # First check bulk buffer
        for key, value in self._bulk_buffer.items():
            if query_expr.evaluate(key, value):
                return True
        
        # Then check tree
        return self._query_exists_helper(self.root, query_expr)
    
    def _query_exists_helper(self, node: TreeNode, expression: QueryExpression) -> bool:
        """
        Helper method to check if any items match the query.
        
        Args:
            node: Current node to search
            expression: Query expression to evaluate
            
        Returns:
            True if any matching item is found
        """
        if node.is_leaf:
            # Check if any data in leaf matches
            for key, value in node.data.items():
                if expression.evaluate(key, value):
                    return True
        else:
            # Recursively search children
            for child in node.children.values():
                if self._query_exists_helper(child, expression):
                    return True
        
        return False
    
    def query_first(self, expression: Union[QueryExpression, str, QueryBuilder]) -> Optional[Tuple[Any, Any]]:
        """
        Get the first item matching a query.
        Enhanced to include bulk buffer data.
        
        Args:
            expression: Query expression
            
        Returns:
            First matching (key, value) tuple or None if no match
        """
        # Convert input to QueryExpression
        if isinstance(expression, str):
            parser = QueryParser()
            query_expr = parser.parse(expression)
        elif isinstance(expression, QueryBuilder):
            query_expr = expression.build()
        elif isinstance(expression, QueryExpression):
            query_expr = expression
        else:
            raise ValueError("Invalid query expression type")
        
        if query_expr is None:
            return None
        
        # First check bulk buffer
        for key, value in self._bulk_buffer.items():
            if query_expr.evaluate(key, value):
                return (key, value)
        
        # Then check tree
        return self._query_first_helper(self.root, query_expr)
    
    def _query_first_helper(self, node: TreeNode, expression: QueryExpression) -> Optional[Tuple[Any, Any]]:
        """
        Helper method to find the first matching item.
        
        Args:
            node: Current node to search
            expression: Query expression to evaluate
            
        Returns:
            First matching (key, value) tuple or None
        """
        if node.is_leaf:
            # Check data in leaf
            for key, value in node.data.items():
                if expression.evaluate(key, value):
                    return (key, value)
        else:
            # Recursively search children
            for child in node.children.values():
                result = self._query_first_helper(child, expression)
                if result is not None:
                    return result
        
        return None
    
    def range_query(self, min_key: Any, max_key: Any, include_min: bool = True, include_max: bool = True) -> List[Tuple[Any, Any]]:
        """
        Perform an optimized range query on keys.
        
        Args:
            min_key: Minimum key value
            max_key: Maximum key value
            include_min: Whether to include the minimum key
            include_max: Whether to include the maximum key
            
        Returns:
            List of (key, value) tuples in the range
        """
        if include_min and include_max:
            op = QueryOperator.BETWEEN
            values = (min_key, max_key)
        else:
            # Build complex expression for exclusive bounds
            builder = QueryBuilder()
            if include_min:
                builder.key_ge(min_key)
            else:
                builder.key_gt(min_key)
            
            if include_max:
                builder.key_le(max_key)
            else:
                builder.key_lt(max_key)
            
            return self.query(builder)
        
        range_expr = RangeExpression('key', op, values)
        return self.query(range_expr)
    
    def prefix_query(self, prefix: str, field: str = 'value') -> List[Tuple[Any, Any]]:
        """
        Find all items where the specified field starts with a prefix.
        
        Args:
            prefix: Prefix to search for
            field: Field to search ('key' or 'value')
            
        Returns:
            List of matching (key, value) tuples
        """
        expr = StringExpression(field, QueryOperator.STARTSWITH, prefix)
        return self.query(expr)
    
    def regex_query(self, pattern: str, field: str = 'value') -> List[Tuple[Any, Any]]:
        """
        Find all items where the specified field matches a regex pattern.
        
        Args:
            pattern: Regular expression pattern
            field: Field to search ('key' or 'value')
            
        Returns:
            List of matching (key, value) tuples
        """
        expr = StringExpression(field, QueryOperator.MATCHES, pattern)
        return self.query(expr)
    
    def type_query(self, type_name: str, field: str = 'value') -> List[Tuple[Any, Any]]:
        """
        Find all items where the specified field is of a certain type.
        
        Args:
            type_name: Name of the type to search for
            field: Field to check ('key' or 'value')
            
        Returns:
            List of matching (key, value) tuples
        """
        expr = SpecialExpression(QueryOperator.TYPE_IS, field, type_name)
        return self.query(expr)
    
    def aggregate_query(self, expression: Union[QueryExpression, str, QueryBuilder], 
                       aggregation: str = 'count') -> Union[int, float, Any]:
        """
        Perform aggregation on query results.
        
        Args:
            expression: Query expression
            aggregation: Type of aggregation ('count', 'sum', 'avg', 'min', 'max')
            
        Returns:
            Aggregated result
        """
        results = self.query(expression)
        
        if not results:
            return 0 if aggregation == 'count' else None
        
        if aggregation == 'count':
            return len(results)
        
        # Extract values for numeric aggregations
        values = [value for _, value in results]
        numeric_values = []
        
        for value in values:
            try:
                if isinstance(value, (int, float)):
                    numeric_values.append(value)
                else:
                    numeric_values.append(float(value))
            except (ValueError, TypeError):
                continue
        
        if not numeric_values:
            return None
        
        if aggregation == 'sum':
            return sum(numeric_values)
        elif aggregation == 'avg':
            return sum(numeric_values) / len(numeric_values)
        elif aggregation == 'min':
            return min(numeric_values)
        elif aggregation == 'max':
            return max(numeric_values)
        else:
            raise ValueError(f"Unknown aggregation type: {aggregation}")
    
    def group_by_query(self, expression: Union[QueryExpression, str, QueryBuilder],
                      group_field: str = 'key') -> Dict[Any, List[Tuple[Any, Any]]]:
        """
        Group query results by a field.
        
        Args:
            expression: Query expression
            group_field: Field to group by ('key' or 'value')
            
        Returns:
            Dictionary mapping group values to lists of (key, value) tuples
        """
        results = self.query(expression)
        groups = {}
        
        for key, value in results:
            group_key = key if group_field == 'key' else value
            
            # Handle unhashable types by converting to string representation
            try:
                # Test if the group_key is hashable by trying to use it as a dict key
                test_dict = {group_key: None}
            except TypeError:
                # If unhashable, convert to string representation
                group_key = f"<{type(group_key).__name__}>:{str(group_key)}"
            
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append((key, value))
        
        return groups
    
    def __repr__(self) -> str:
        """Return string representation of the tree."""
        return f"Tree(max_capacity={self.config.max_capacity}, root={self.root})"
    
    def delete(self, key: Any) -> bool:
        """
        Delete a key from the tree.
        Enhanced to also check bulk buffer.
        
        Args:
            key: The key to delete
            
        Returns:
            True if key was found and deleted, False otherwise
        """
        # First check and remove from bulk buffer if present
        if key in self._bulk_buffer:
            del self._bulk_buffer[key]
            return True
        
        # Find the leaf containing the key
        current = self.root
        
        while not current.is_leaf:
            if not current.children:
                return False
            
            # Find the appropriate child
            child_keys = sorted(current.children.keys())
            target_child = None
            
            for child_key in child_keys:
                if key <= child_key:
                    target_child = current.children[child_key]
                    break
            
            if target_child is None:
                target_child = current.children[child_keys[-1]]
            
            current = target_child
        
        # Try to delete from leaf
        if key in current.data:
            del current.data[key]
            self._check_underflow(current)
            return True
        
        return False
    
    def _check_underflow(self, node: TreeNode) -> None:
        """
        Check if a node has too few items and handle merging/redistribution.
        Enhanced with configurable merge strategies.
        
        Args:
            node: The node to check
        """
        if node.parent is None:
            return
        
        # Check if node is underutilized based on strategy
        if not node.is_underutilized:
            return
        
        # Try to borrow from siblings based on merge strategy
        if self.config.merge_strategy != MergeStrategy.IMMEDIATE:
            if self._try_borrow(node):
                return
        
        # If borrowing fails or immediate merge is configured, merge with a sibling
        self._merge_with_sibling(node)
    
    def _try_borrow(self, node: TreeNode) -> bool:
        """
        Try to borrow items from sibling nodes with enhanced strategy.
        
        Args:
            node: The node that needs more items
            
        Returns:
            True if borrowing was successful, False otherwise
        """
        if not node.parent:
            return False
        
        # Find siblings with extra items
        best_sibling = None
        max_excess = 0
        
        for sibling in node.parent.children.values():
            if sibling == node:
                continue
                
            excess = len(sibling.coverage) - self.config.min_capacity
            if excess > max_excess:
                max_excess = excess
                best_sibling = sibling
        
        if best_sibling is None or max_excess <= 0:
            return False
        
        # Calculate optimal number of items to borrow
        total_items = len(node.coverage) + len(best_sibling.coverage)
        target_per_node = total_items // 2
        items_to_borrow = target_per_node - len(node.coverage)
        items_to_borrow = min(items_to_borrow, max_excess)
        
        if items_to_borrow <= 0:
            return False
        
        # Perform borrowing
        if node.is_leaf and best_sibling.is_leaf:
            # Borrow data items
            items = list(best_sibling.data.items())
            items.sort()
            
            for _ in range(items_to_borrow):
                if not items:
                    break
                key, value = items.pop(0)  # Take from beginning for better distribution
                del best_sibling.data[key]
                node.data[key] = value
            
            return True
            
        elif not node.is_leaf and not best_sibling.is_leaf:
            # Borrow children
            items = list(best_sibling.children.items())
            items.sort()
            
            for _ in range(items_to_borrow):
                if not items:
                    break
                key, child = items.pop(0)
                del best_sibling.children[key]
                node.children[key] = child
                child.parent = node
            
            return True
        
        return False
    
    def _merge_with_sibling(self, node: TreeNode) -> None:
        """
        Merge a node with one of its siblings using enhanced strategy.
        
        Args:
            node: The node to merge
        """
        if not node.parent or len(node.parent.children) <= 1:
            return
        
        # Find the best sibling to merge with (smallest one for efficiency)
        best_sibling = None
        min_size = float('inf')
        
        for sibling in node.parent.children.values():
            if sibling != node and len(sibling.coverage) < min_size:
                min_size = len(sibling.coverage)
                best_sibling = sibling
        
        if best_sibling is None:
            return
        
        # Check if merged size would exceed capacity
        merged_size = len(node.coverage) + len(best_sibling.coverage)
        if merged_size > self.config.max_capacity:
            # Try redistribution instead
            if self._try_redistribute_between(node, best_sibling):
                return
        
        node.stats.record_merge()
        best_sibling.stats.record_merge()
        self._stats['total_merges'] += 1
        
        # Merge the nodes
        if node.is_leaf and best_sibling.is_leaf:
            # Merge data
            best_sibling.data.update(node.data)
        elif not node.is_leaf and not best_sibling.is_leaf:
            # Merge children
            best_sibling.children.update(node.children)
            for child in node.children.values():
                child.parent = best_sibling
        
        # Remove the merged node from parent
        del node.parent.children[node.key]
        
        # Remove from cache if present
        if node.key in self._node_cache:
            del self._node_cache[node.key]
        
        # Check if parent needs attention
        if node.parent:
            self._check_underflow(node.parent)
    
    def _try_redistribute_between(self, node1: TreeNode, node2: TreeNode) -> bool:
        """
        Try to redistribute items between two specific nodes.
        
        Args:
            node1: First node
            node2: Second node
            
        Returns:
            True if redistribution was successful
        """
        total_items = len(node1.coverage) + len(node2.coverage)
        target_per_node = total_items // 2
        
        if node1.is_leaf and node2.is_leaf:
            # Redistribute data
            all_items = list(node1.data.items()) + list(node2.data.items())
            all_items.sort()
            
            node1.data.clear()
            node2.data.clear()
            
            node1.data.update(dict(all_items[:target_per_node]))
            node2.data.update(dict(all_items[target_per_node:]))
            
            return True
            
        elif not node1.is_leaf and not node2.is_leaf:
            # Redistribute children
            all_items = list(node1.children.items()) + list(node2.children.items())
            all_items.sort()
            
            node1.children.clear()
            node2.children.clear()
            
            for key, child in all_items[:target_per_node]:
                node1.children[key] = child
                child.parent = node1
            
            for key, child in all_items[target_per_node:]:
                node2.children[key] = child
                child.parent = node2
            
            return True
        
        return False
    
    def inorder_traversal(self) -> List[tuple]:
        """
        Perform inorder traversal of the tree.
        Enhanced to include bulk buffer data and avoid duplicates.
        
        Returns:
            List of (key, value) tuples in sorted order (without duplicates)
        """
        # Use a dictionary to avoid duplicates by key
        result_dict = {}
        
        # First collect from tree
        tree_result = []
        self._inorder_helper(self.root, tree_result)
        for key, value in tree_result:
            result_dict[key] = value
        
        # Then add bulk buffer data (this will overwrite tree data if same key exists)
        for key, value in self._bulk_buffer.items():
            result_dict[key] = value
        
        # Convert back to sorted list
        return sorted(result_dict.items(), key=lambda x: x[0])
    
    def _inorder_helper(self, node: TreeNode, result: List[tuple]) -> None:
        """
        Helper method for inorder traversal.
        
        Args:
            node: Current node
            result: List to store results
        """
        if node.is_leaf:
            # Add all data items from leaf
            for key, value in sorted(node.data.items()):
                result.append((key, value))
        else:
            # Recursively traverse children
            for child_key in sorted(node.children.keys()):
                self._inorder_helper(node.children[child_key], result)
    
    def get_all_keys(self) -> List[Any]:
        """
        Get all keys in the tree in sorted order.
        
        Returns:
            List of all keys
        """
        return [key for key, _ in self.inorder_traversal()]
    
    def get_all_values(self) -> List[Any]:
        """
        Get all values in the tree in key-sorted order.
        
        Returns:
            List of all values
        """
        return [value for _, value in self.inorder_traversal()]
    
    def size(self) -> int:
        """
        Get the total number of items in the tree (including bulk buffer).
        Enhanced to include bulk buffer data.
        
        Returns:
            Total number of items
        """
        tree_size = self._size_helper(self.root)
        buffer_size = len(self._bulk_buffer)  # Dict length
        return tree_size + buffer_size
    
    def is_empty(self) -> bool:
        """
        Check if the tree is empty.
        Enhanced to consider bulk buffer.
        
        Returns:
            True if tree is empty, False otherwise
        """
        return self.size() == 0
    
    def height(self) -> int:
        """
        Calculate the height of the tree.
        
        Returns:
            Height of the tree (0 for empty tree)
        """
        return self._height_helper(self.root)
    
    def _height_helper(self, node: TreeNode) -> int:
        """
        Helper method to calculate tree height.
        
        Args:
            node: Current node
            
        Returns:
            Height from this node
        """
        if node.is_leaf:
            return 1
        
        if not node.children:
            return 1
        
        max_child_height = 0
        for child in node.children.values():
            child_height = self._height_helper(child)
            max_child_height = max(max_child_height, child_height)
        
        return max_child_height + 1
    
    def print_tree(self, indent: int = 0) -> None:
        """
        Print a visual representation of the tree.
        
        Args:
            indent: Initial indentation level
        """
        self._print_helper(self.root, indent)
    
    def _print_helper(self, node: TreeNode, indent: int) -> None:
        """
        Helper method for printing tree structure.
        
        Args:
            node: Current node
            indent: Current indentation level
        """
        prefix = "  " * indent
        
        if node.is_leaf:
            print(f"{prefix}Leaf({node.key}): {dict(sorted(node.data.items()))}")
        else:
            print(f"{prefix}Internal({node.key}): {len(node.children)} children")
            for child_key in sorted(node.children.keys()):
                self._print_helper(node.children[child_key], indent + 1)
    
    def validate(self) -> bool:
        """
        Validate the tree structure and properties.
        
        Returns:
            True if tree is valid, False otherwise
        """
        try:
            return self._validate_helper(self.root)
        except Exception:
            return False
    
    def _validate_helper(self, node: TreeNode) -> bool:
        """
        Helper method for tree validation.
        
        Args:
            node: Current node to validate
            
        Returns:
            True if subtree is valid, False otherwise
        """
        # Check node capacity constraints (allow some flexibility for internal nodes during splits)
        if node.is_leaf and len(node.coverage) > self.config.max_capacity:
            return False
        
        # Internal nodes can temporarily exceed capacity during splitting
        if not node.is_leaf and len(node.coverage) > self.config.max_capacity * 2:
            return False
        
        # Check parent-child relationships
        if node.parent is not None:
            if node not in node.parent.children.values():
                return False
        
        # Recursively validate children
        if not node.is_leaf:
            for child in node.children.values():
                if child.parent != node:
                    return False
                if not self._validate_helper(child):
                    return False
        
        return True
    
    def flush_bulk_buffer(self) -> None:
        """Flush the bulk buffer to the tree."""
        if self._bulk_buffer:
            self._bulk_insert()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive tree statistics.
        
        Returns:
            Dictionary with performance and usage statistics
        """
        return {
            **self._stats,
            'config': self.config,
            'cache_size': len(self._node_cache),
            'buffer_size': len(self._bulk_buffer),
            'cache_hit_rate': (self._stats['cache_hits'] / 
                             max(1, self._stats['cache_hits'] + self._stats['cache_misses'])),
            'tree_size': self.size(),
            'tree_height': self.height(),
        }
    
    def optimize(self) -> None:
        """
        Perform tree optimization operations.
        """
        # Clear old cache entries
        if len(self._node_cache) > self.config.cache_size:
            # Remove least recently accessed entries
            sorted_cache = sorted(
                self._node_cache.items(),
                key=lambda x: x[1].stats.last_access_time
            )
            
            for key, _ in sorted_cache[:len(self._node_cache) - self.config.cache_size]:
                del self._node_cache[key]
        
        # Flush bulk buffer
        self.flush_bulk_buffer()
        
        # Perform compression if enabled
        if self.config.enable_compression:
            self._compress_nodes()
    
    def _compress_nodes(self) -> None:
        """Compress nodes that should be compressed."""
        def _compress_helper(node: TreeNode) -> None:
            if node.should_compress():
                # Simple compression: remove least accessed keys for leaf nodes
                if node.is_leaf and len(node.data) > self.config.max_capacity * 0.8:
                    # Keep only hot keys
                    hot_data = {k: v for k, v in node.data.items() 
                              if k in node.stats.hot_keys}
                    if len(hot_data) < len(node.data) * 0.5:
                        node.data = hot_data
            
            if not node.is_leaf:
                for child in node.children.values():
                    _compress_helper(child)
        
        _compress_helper(self.root)
    
    def _single_insert(self, key: int, value: Any) -> None:
        """Insert a single key-value pair."""
        # Find the appropriate leaf node
        current = self.root
        
        while not current.is_leaf:
            current.access()  # 记录访问统计
            
            if not current.children:
                # Create a new leaf child if no children exist
                new_leaf = TreeNode(self, key, is_leaf=True, parent=current)
                current.children[key] = new_leaf
                current = new_leaf
                break
            
            # Find the appropriate child based on key
            child_keys = sorted(current.children.keys())
            target_child = None
            
            for i, child_key in enumerate(child_keys):
                if key <= child_key:
                    target_child = current.children[child_key]
                    break
            
            # If key is larger than all child keys, use the last child
            if target_child is None:
                target_child = current.children[child_keys[-1]]
            
            current = target_child
        
        # Insert the data into the leaf node
        current.access(key)  # 记录访问统计
        current.data[key] = value
        
        # Check if we need to split the node
        self._check_and_split(current)
    
    def _bulk_insert(self) -> None:
        """
        Perform bulk insertion of buffered items.
        """
        if not self._bulk_buffer:
            return
        
        # Sort items by key for optimal insertion order
        sorted_items = sorted(self._bulk_buffer.items())
        
        # Insert all items
        for key, value in sorted_items:
            self._single_insert(key, value)
        
        # Clear buffer
        self._bulk_buffer.clear()
        self._stats['bulk_operations'] += 1
    
    def _split_node(self, node: TreeNode) -> None:
        """
        Split a node when it exceeds maximum capacity using configured strategy.
        
        Args:
            node: The node to split
        """
        if node.parent is None:
            # Cannot split root node in this implementation
            return
        
        node.stats.record_split()
        self._stats['total_splits'] += 1
        
        if node.is_leaf:
            # Split leaf node
            items = list(node.data.items())
            items.sort()  # 确保有序
            
            split_point = node.get_split_point(items)
            
            # Create new leaf for second half
            new_key = items[split_point][0]  # Use first key of second half as new node key
            new_leaf = TreeNode(self, new_key, is_leaf=True, parent=node.parent)
            
            # Distribute data based on split strategy
            new_leaf.data = dict(items[split_point:])
            node.data = dict(items[:split_point])
            
            # Add new leaf to parent
            node.parent.children[new_key] = new_leaf
            
        else:
            # Split internal node
            items = list(node.children.items())
            items.sort()  # 确保有序
            
            split_point = node.get_split_point(items)
            
            # Create new internal node for second half
            new_key = items[split_point][0]
            new_node = TreeNode(self, new_key, is_leaf=False, parent=node.parent)
            
            # Distribute children
            for key, child in items[split_point:]:
                new_node.children[key] = child
                child.parent = new_node
            
            # Keep first half in original node
            node.children = dict(items[:split_point])
            
            # Add new node to parent
            node.parent.children[new_key] = new_node
    
    def _check_and_split(self, node: TreeNode) -> None:
        """
        Check if a node needs splitting and handle it recursively.
        
        Args:
            node: The node to check
        """
        if len(node.coverage) < self.config.max_capacity:
            return
        
        # Try to redistribute to sibling nodes first
        if node.parent and self._try_redistribute(node):
            return
        
        # If redistribution fails, split the node
        self._split_node(node)
        
        # Recursively check parent nodes
        if node.parent:
            self._check_and_split(node.parent)
    
    def _try_redistribute(self, node: TreeNode) -> bool:
        """
        Try to redistribute items to sibling nodes to avoid splitting.
        
        Args:
            node: The node to redistribute from
            
        Returns:
            True if redistribution was successful, False otherwise
        """
        if not node.parent:
            return False
        
        # Find siblings with space
        for sibling in node.parent.children.values():
            if sibling != node and len(sibling.coverage) < self.config.max_capacity:
                # Calculate how many items to move
                total_items = len(node.coverage) + len(sibling.coverage)
                target_per_node = total_items // 2
                items_to_move = len(node.coverage) - target_per_node
                
                if items_to_move <= 0:
                    continue
                
                # Try to move some items to this sibling
                if node.is_leaf and sibling.is_leaf:
                    # Move multiple data items for better balance
                    items = list(node.data.items())
                    items.sort()
                    
                    for _ in range(min(items_to_move, len(items))):
                        key, value = items.pop()
                        del node.data[key]
                        sibling.data[key] = value
                    
                    return True
                    
                elif not node.is_leaf and not sibling.is_leaf:
                    # Move multiple children
                    items = list(node.children.items())
                    items.sort()
                    
                    for _ in range(min(items_to_move, len(items))):
                        key, child = items.pop()
                        del node.children[key]
                        sibling.children[key] = child
                        child.parent = sibling
                    
                    return True
        
        return False
    
    def search(self, key: Any) -> Optional[Any]:
        """
        Search for a key in the tree (including bulk buffer).
        Enhanced to search both tree and bulk buffer.
        
        Args:
            key: The key to search for
            
        Returns:
            The value if found, None otherwise
        """
        # First check bulk buffer
        if key in self._bulk_buffer:
            self._stats['cache_hits'] += 1
            return self._bulk_buffer[key]
        
        # Then search in tree
        result = self._search_helper(self.root, key)
        if result is not None:
            self._stats['cache_hits'] += 1
        else:
            self._stats['cache_misses'] += 1
        return result
    
    def _search_helper(self, node: TreeNode, key: Any) -> Optional[Any]:
        """
        Helper method to search for a key in the tree.
        
        Args:
            node: Current node to search
            key: The key to search for
            
        Returns:
            The value if found, None otherwise
        """
        if node.is_leaf:
            return node.data.get(key)
        
        # For internal nodes, find the appropriate child
        if not node.children:
            return None
        
        # Find the appropriate child
        child_keys = sorted(node.children.keys())
        target_child = None
        
        for child_key in child_keys:
            if key <= child_key:
                target_child = node.children[child_key]
                break
        
        if target_child is None:
            target_child = node.children[child_keys[-1]]
        
        return self._search_helper(target_child, key)
    
    def batch_search(self, keys: List[Any]) -> Dict[Any, Optional[Any]]:
        """
        Batch search for multiple keys.
        
        Args:
            keys: List of keys to search for
            
        Returns:
            Dictionary mapping keys to their values (or None if not found)
        """
        results = {}
        for key in keys:
            results[key] = self.search(key)
        return results
    
    def batch_delete(self, keys: List[Any]) -> Dict[Any, bool]:
        """
        Batch delete multiple keys.
        
        Args:
            keys: List of keys to delete
            
        Returns:
            Dictionary mapping keys to deletion success status
        """
        results = {}
        for key in keys:
            results[key] = self.delete(key)
        return results
    
    def batch_query(self, expressions: List[Union[QueryExpression, str, QueryBuilder]]) -> List[List[Tuple[Any, Any]]]:
        """
        Batch execute multiple queries.
        
        Args:
            expressions: List of query expressions
            
        Returns:
            List of query results for each expression
        """
        results = []
        for expression in expressions:
            results.append(self.query(expression))
        return results
    
    def batch_update(self, updates: List[Tuple[Any, Any]]) -> Dict[Any, bool]:
        """
        Batch update multiple key-value pairs.
        Updates existing keys or inserts new ones.
        
        Args:
            updates: List of (key, new_value) tuples
            
        Returns:
            Dictionary mapping keys to update success status
        """
        results = {}
        for key, value in updates:
            try:
                # Check if key exists
                existing = self.search(key)
                if existing is not None:
                    # Delete old value and insert new one
                    self.delete(key)
                self.insert(key, value)
                results[key] = True
            except Exception:
                results[key] = False
        return results
    
    def batch_exists(self, keys: List[Any]) -> Dict[Any, bool]:
        """
        Batch check existence of multiple keys.
        
        Args:
            keys: List of keys to check
            
        Returns:
            Dictionary mapping keys to existence status
        """
        results = {}
        for key in keys:
            results[key] = self.search(key) is not None
        return results
    
    def batch_get_size_by_keys(self, keys: List[Any]) -> Dict[Any, int]:
        """
        Batch get the size (length) of values for multiple keys.
        Only works for values that have a length (str, list, dict, etc.).
        
        Args:
            keys: List of keys to get value sizes for
            
        Returns:
            Dictionary mapping keys to their value sizes (or -1 if not found/no length)
        """
        results = {}
        for key in keys:
            value = self.search(key)
            if value is not None:
                try:
                    results[key] = len(value)
                except TypeError:
                    results[key] = -1  # Value doesn't have length
            else:
                results[key] = -1  # Key not found
        return results
    
    def batch_filter_by_type(self, keys: List[Any], target_type: type) -> Dict[Any, bool]:
        """
        Batch filter keys by value type.
        
        Args:
            keys: List of keys to check
            target_type: The type to filter by
            
        Returns:
            Dictionary mapping keys to whether their values match the target type
        """
        results = {}
        for key in keys:
            value = self.search(key)
            if value is not None:
                results[key] = isinstance(value, target_type)
            else:
                results[key] = False
        return results
    
    def _size_helper(self, node: TreeNode) -> int:
        """
        Helper method to calculate the size of the tree.
        
        Args:
            node: Current node
            
        Returns:
            Number of items in the subtree
        """
        if node.is_leaf:
            return len(node.data)
        
        total_size = 0
        for child in node.children.values():
            total_size += self._size_helper(child)
        
        return total_size
    
    def _delete_helper(self, node: TreeNode, key: Any) -> bool:
        """
        Helper method to delete a key from the tree.
        
        Args:
            node: Current node
            key: The key to delete
            
        Returns:
            True if deleted, False if not found
        """
        if node.is_leaf:
            if key in node.data:
                del node.data[key]
                return True
            return False
        
        # For internal nodes, find the appropriate child
        if not node.children:
            return False
        
        # Find the appropriate child
        child_keys = sorted(node.children.keys())
        target_child = None
        
        for child_key in child_keys:
            if key <= child_key:
                target_child = node.children[child_key]
                break
        
        if target_child is None:
            target_child = node.children[child_keys[-1]]
        
        return self._delete_helper(target_child, key)
    
    def query_limit(self, expression: Union[QueryExpression, str, QueryBuilder], 
                   limit: int = 10, offset: int = 0) -> List[Tuple[Any, Any]]:
        """
        Query the tree with limit and offset (pagination support).
        
        Args:
            expression: Query expression
            limit: Maximum number of results to return
            offset: Number of results to skip
            
        Returns:
            Limited list of (key, value) tuples matching the query
        """
        all_results = self.query(expression)
        return all_results[offset:offset + limit]
    
    def query_sorted(self, expression: Union[QueryExpression, str, QueryBuilder], 
                    sort_by: str = 'key', reverse: bool = False) -> List[Tuple[Any, Any]]:
        """
        Query the tree and return sorted results.
        
        Args:
            expression: Query expression
            sort_by: Sort by 'key' or 'value'
            reverse: Sort in descending order if True
            
        Returns:
            Sorted list of (key, value) tuples matching the query
        """
        results = self.query(expression)
        
        if sort_by == 'key':
            try:
                return sorted(results, key=lambda x: x[0], reverse=reverse)
            except TypeError:
                # Handle mixed types by converting to string
                return sorted(results, key=lambda x: str(x[0]), reverse=reverse)
        elif sort_by == 'value':
            try:
                return sorted(results, key=lambda x: x[1], reverse=reverse)
            except TypeError:
                # Handle mixed types by converting to string
                return sorted(results, key=lambda x: str(x[1]), reverse=reverse)
        else:
            raise ValueError("sort_by must be 'key' or 'value'")
    
    def query_paginated(self, expression: Union[QueryExpression, str, QueryBuilder], 
                       page: int = 1, page_size: int = 10, 
                       sort_by: str = 'key', reverse: bool = False) -> Dict[str, Any]:
        """
        Query the tree with pagination support.
        
        Args:
            expression: Query expression
            page: Page number (1-based)
            page_size: Number of items per page
            sort_by: Sort by 'key' or 'value'
            reverse: Sort in descending order if True
            
        Returns:
            Dictionary containing results, pagination info, and metadata
        """
        if page < 1:
            raise ValueError("Page number must be >= 1")
        if page_size < 1:
            raise ValueError("Page size must be >= 1")
        
        # Get all results and sort them
        all_results = self.query_sorted(expression, sort_by, reverse)
        total_count = len(all_results)
        total_pages = (total_count + page_size - 1) // page_size  # Ceiling division
        
        # Calculate offset
        offset = (page - 1) * page_size
        
        # Get page results
        page_results = all_results[offset:offset + page_size]
        
        return {
            'results': page_results,
            'pagination': {
                'current_page': page,
                'page_size': page_size,
                'total_pages': total_pages,
                'total_count': total_count,
                'has_next': page < total_pages,
                'has_prev': page > 1,
                'next_page': page + 1 if page < total_pages else None,
                'prev_page': page - 1 if page > 1 else None
            },
            'sort_info': {
                'sort_by': sort_by,
                'reverse': reverse
            }
        }
    
    def query_distinct(self, expression: Union[QueryExpression, str, QueryBuilder], 
                      field: str = 'value') -> List[Any]:
        """
        Query the tree and return distinct values.
        
        Args:
            expression: Query expression
            field: Field to get distinct values from ('key' or 'value')
            
        Returns:
            List of distinct values
        """
        results = self.query(expression)
        
        if field == 'key':
            values = [key for key, _ in results]
        elif field == 'value':
            values = [value for _, value in results]
        else:
            raise ValueError("field must be 'key' or 'value'")
        
        # Use dict.fromkeys to preserve order while removing duplicates
        # This works for hashable types, for non-hashable types we fall back to slower method
        try:
            return list(dict.fromkeys(values))
        except TypeError:
            # Handle non-hashable types
            distinct_values = []
            for value in values:
                if value not in distinct_values:
                    distinct_values.append(value)
            return distinct_values
    
    def query_top(self, expression: Union[QueryExpression, str, QueryBuilder], 
                 n: int = 10, sort_by: str = 'key', reverse: bool = True) -> List[Tuple[Any, Any]]:
        """
        Query the tree and return top N results.
        
        Args:
            expression: Query expression
            n: Number of top results to return
            sort_by: Sort by 'key' or 'value'
            reverse: Sort in descending order if True (default for "top")
            
        Returns:
            List of top N (key, value) tuples matching the query
        """
        return self.query_sorted(expression, sort_by, reverse)[:n]
    
    def query_sample(self, expression: Union[QueryExpression, str, QueryBuilder], 
                    n: int = 10, seed: Optional[int] = None) -> List[Tuple[Any, Any]]:
        """
        Query the tree and return a random sample of results.
        
        Args:
            expression: Query expression
            n: Number of samples to return
            seed: Random seed for reproducible results
            
        Returns:
            List of randomly sampled (key, value) tuples matching the query
        """
        import random
        
        results = self.query(expression)
        
        if seed is not None:
            random.seed(seed)
        
        # Return sample without replacement
        sample_size = min(n, len(results))
        return random.sample(results, sample_size)
    
    def query_with_conditions(self, 
                             expression: Union[QueryExpression, str, QueryBuilder],
                             limit: Optional[int] = None,
                             offset: int = 0,
                             sort_by: Optional[str] = None,
                             reverse: bool = False,
                             distinct: bool = False,
                             distinct_field: str = 'value') -> List[Tuple[Any, Any]]:
        """
        Advanced query with multiple conditions and options.
        
        Args:
            expression: Query expression
            limit: Maximum number of results to return
            offset: Number of results to skip
            sort_by: Sort by 'key' or 'value' (None for no sorting)
            reverse: Sort in descending order if True
            distinct: Return only distinct results
            distinct_field: Field to check for distinctness ('key' or 'value')
            
        Returns:
            List of (key, value) tuples matching the query with applied conditions
        """
        # Start with basic query
        results = self.query(expression)
        
        # Apply sorting if requested
        if sort_by:
            try:
                if sort_by == 'key':
                    results = sorted(results, key=lambda x: x[0], reverse=reverse)
                elif sort_by == 'value':
                    results = sorted(results, key=lambda x: x[1], reverse=reverse)
                else:
                    raise ValueError("sort_by must be 'key' or 'value'")
            except TypeError:
                # Handle mixed types by converting to string
                if sort_by == 'key':
                    results = sorted(results, key=lambda x: str(x[0]), reverse=reverse)
                else:
                    results = sorted(results, key=lambda x: str(x[1]), reverse=reverse)
        
        # Apply distinct filter if requested
        if distinct:
            seen = set()
            distinct_results = []
            
            for key, value in results:
                check_value = key if distinct_field == 'key' else value
                
                # Handle non-hashable types
                try:
                    if check_value not in seen:
                        seen.add(check_value)
                        distinct_results.append((key, value))
                except TypeError:
                    # For non-hashable types, use slower comparison
                    is_duplicate = False
                    for existing_key, existing_value in distinct_results:
                        existing_check = existing_key if distinct_field == 'key' else existing_value
                        if check_value == existing_check:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        distinct_results.append((key, value))
            
            results = distinct_results
        
        # Apply offset and limit
        if offset > 0:
            results = results[offset:]
        
        if limit is not None:
            results = results[:limit]
        
        return results
    
    def advanced_query(self, 
                      expression: Optional[Union[QueryExpression, str, QueryBuilder]] = None,
                      # 基础查询参数
                      min_key: Optional[Any] = None,
                      max_key: Optional[Any] = None,
                      include_min: bool = True,
                      include_max: bool = True,
                      prefix: Optional[str] = None,
                      regex_pattern: Optional[str] = None,
                      value_type: Optional[str] = None,
                      field: str = 'value',
                      # 结果处理参数
                      limit: Optional[int] = None,
                      offset: int = 0,
                      sort_by: Optional[str] = None,
                      reverse: bool = False,
                      distinct: bool = False,
                      distinct_field: str = 'value',
                      # 聚合参数
                      aggregation: Optional[str] = None,
                      group_by: Optional[str] = None,
                      # 采样参数
                      sample_size: Optional[int] = None,
                      sample_seed: Optional[int] = None,
                      # 返回格式参数
                      return_format: str = 'tuples',  # 'tuples', 'keys', 'values', 'dict', 'count', 'exists', 'first'
                      # 分页参数
                      page: Optional[int] = None,
                      page_size: int = 10) -> Union[List[Tuple[Any, Any]], List[Any], Dict[str, Any], int, bool, Any]:
        """
        高级查询函数，集成所有查询方式和功能。
        
        Args:
            expression: 查询表达式（字符串、QueryExpression或QueryBuilder）
            min_key: 最小键值（范围查询）
            max_key: 最大键值（范围查询）
            include_min: 是否包含最小值
            include_max: 是否包含最大值
            prefix: 前缀匹配字符串
            regex_pattern: 正则表达式模式
            value_type: 值类型过滤
            field: 操作字段（'key'或'value'）
            limit: 结果数量限制
            offset: 结果偏移量
            sort_by: 排序字段（'key'或'value'）
            reverse: 是否逆序排序
            distinct: 是否去重
            distinct_field: 去重字段
            aggregation: 聚合操作（'count', 'sum', 'avg', 'max', 'min'）
            group_by: 分组字段
            sample_size: 随机采样大小
            sample_seed: 随机种子
            return_format: 返回格式
            page: 页码（分页查询）
            page_size: 每页大小
            
        Returns:
            根据return_format返回不同格式的结果
        """
        # 收集所有查询条件
        query_conditions = []
        
        # 1. 处理主要表达式
        if expression is not None:
            if isinstance(expression, str):
                parser = QueryParser()
                main_expr = parser.parse(expression)
            elif isinstance(expression, QueryBuilder):
                main_expr = expression.build()
            else:
                main_expr = expression
            
            if main_expr:
                query_conditions.append(main_expr)
        
        # 2. 处理范围查询
        if min_key is not None or max_key is not None:
            if min_key is not None and max_key is not None:
                range_expr = RangeExpression('key', QueryOperator.BETWEEN, (min_key, max_key))
            elif min_key is not None:
                range_expr = ComparisonExpression('key', QueryOperator.GE if include_min else QueryOperator.GT, min_key)
            else:  # max_key is not None
                range_expr = ComparisonExpression('key', QueryOperator.LE if include_max else QueryOperator.LT, max_key)
            query_conditions.append(range_expr)
        
        # 3. 处理前缀查询
        if prefix is not None:
            prefix_expr = StringExpression(field, QueryOperator.STARTSWITH, prefix)
            query_conditions.append(prefix_expr)
        
        # 4. 处理正则表达式查询
        if regex_pattern is not None:
            regex_expr = StringExpression(field, QueryOperator.MATCHES, regex_pattern)
            query_conditions.append(regex_expr)
        
        # 5. 处理类型查询
        if value_type is not None:
            type_expr = SpecialExpression(QueryOperator.TYPE_IS, field, value_type)
            query_conditions.append(type_expr)
        
        # 6. 组合所有查询条件
        if not query_conditions:
            # 如果没有任何查询条件，返回所有数据
            final_expr = None
        elif len(query_conditions) == 1:
            final_expr = query_conditions[0]
        else:
            # 使用AND组合所有条件
            final_expr = LogicalExpression(QueryOperator.AND, *query_conditions)
        
        # 7. 执行基础查询
        if final_expr is None:
            # 获取所有数据
            base_results = self.inorder_traversal()
        else:
            base_results = self.query(final_expr)
        
        # 8. 处理聚合查询
        if aggregation is not None:
            if final_expr is None:
                # 创建一个匹配所有的表达式
                all_expr = SpecialExpression(QueryOperator.EXISTS, 'key')
                return self.aggregate_query(all_expr, aggregation)
            else:
                return self.aggregate_query(final_expr, aggregation)
        
        # 9. 处理分组查询
        if group_by is not None:
            if final_expr is None:
                all_expr = SpecialExpression(QueryOperator.EXISTS, 'key')
                return self.group_by_query(all_expr, group_by)
            else:
                return self.group_by_query(final_expr, group_by)
        
        # 10. 处理随机采样
        if sample_size is not None:
            import random
            if sample_seed is not None:
                random.seed(sample_seed)
            if len(base_results) <= sample_size:
                sampled_results = base_results
            else:
                sampled_results = random.sample(base_results, sample_size)
            base_results = sampled_results
        
        # 11. 处理排序
        if sort_by is not None:
            try:
                if sort_by == 'key':
                    base_results.sort(key=lambda x: x[0], reverse=reverse)
                elif sort_by == 'value':
                    # 处理不同类型值的排序，使用类型名称和字符串表示进行排序
                    def sort_key(item):
                        value = item[1]
                        # 返回 (类型优先级, 类型名, 字符串表示) 作为排序键
                        type_priority = {
                            'NoneType': 0,
                            'bool': 1,
                            'int': 2,
                            'float': 3,
                            'str': 4,
                            'list': 5,
                            'tuple': 6,
                            'dict': 7,
                            'set': 8
                        }
                        type_name = type(value).__name__
                        priority = type_priority.get(type_name, 9)
                        return (priority, type_name, str(value))
                    
                    base_results.sort(key=sort_key, reverse=reverse)
                else:
                    raise ValueError(f"Invalid sort_by field: {sort_by}")
            except Exception as e:
                # 如果排序失败，记录错误但继续执行
                print(f"Warning: Sorting failed: {e}")
                # 尝试简单的字符串排序作为备选方案
                try:
                    if sort_by == 'key':
                        base_results.sort(key=lambda x: str(x[0]), reverse=reverse)
                    elif sort_by == 'value':
                        base_results.sort(key=lambda x: str(x[1]), reverse=reverse)
                except:
                    pass  # 如果还是失败，就不排序了
        
        # 12. 处理去重
        if distinct:
            seen = set()
            unique_results = []
            for key, value in base_results:
                if distinct_field == 'key':
                    item = key
                elif distinct_field == 'value':
                    item = value
                else:
                    raise ValueError(f"Invalid distinct_field: {distinct_field}")
                
                # 处理不可哈希类型
                try:
                    if item not in seen:
                        seen.add(item)
                        unique_results.append((key, value))
                except TypeError:
                    # 对于不可哈希类型，使用线性搜索
                    if item not in [getattr(r, distinct_field == 'value' and 1 or 0) for r in unique_results]:
                        unique_results.append((key, value))
            base_results = unique_results
        
        # 13. 处理分页
        if page is not None:
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paginated_results = base_results[start_idx:end_idx]
            
            return {
                'data': paginated_results,
                'page': page,
                'page_size': page_size,
                'total_results': len(base_results),
                'total_pages': (len(base_results) + page_size - 1) // page_size,
                'has_next': end_idx < len(base_results),
                'has_prev': page > 1
            }
        
        # 14. 处理限制和偏移
        if limit is not None or offset > 0:
            end_idx = len(base_results) if limit is None else offset + limit
            base_results = base_results[offset:end_idx]
        
        # 15. 根据返回格式处理结果
        if return_format == 'tuples':
            return base_results
        elif return_format == 'keys':
            return [key for key, _ in base_results]
        elif return_format == 'values':
            return [value for _, value in base_results]
        elif return_format == 'dict':
            return dict(base_results)
        elif return_format == 'count':
            return len(base_results)
        elif return_format == 'exists':
            return len(base_results) > 0
        elif return_format == 'first':
            return base_results[0] if base_results else None
        else:
            raise ValueError(f"Invalid return_format: {return_format}")
    
    def query_builder(self) -> QueryBuilder:
        """
        创建一个新的QueryBuilder实例，用于构建复杂查询。
        
        Returns:
            新的QueryBuilder实例
        """
        return QueryBuilder()
    
    def search_advanced(self, **kwargs) -> Union[List[Tuple[Any, Any]], List[Any], Dict[str, Any], int, bool, Any]:
        """
        advanced_query的别名，提供更简洁的调用方式。
        
        Args:
            **kwargs: 传递给advanced_query的所有参数
            
        Returns:
            查询结果
        """
        return self.advanced_query(**kwargs)


# Utility functions and examples

def create_sample_tree(config: Optional[TreeConfig] = None) -> Tree:
    """
    Create a sample tree with some test data.
    
    Args:
        config: Tree configuration (uses default if None)
        
    Returns:
        A tree with sample data
    """
    tree = Tree(config or TreeConfig(max_capacity=4))
    
    # Insert some sample data
    sample_data = [
        (10, "ten"), (5, "five"), (15, "fifteen"), (3, "three"),
        (7, "seven"), (12, "twelve"), (18, "eighteen"), (1, "one"),
        (4, "four"), (6, "six"), (8, "eight"), (11, "eleven"),
        (13, "thirteen"), (16, "sixteen"), (20, "twenty")
    ]
    
    for key, value in sample_data:
        tree.insert(key, value)
    
    # Flush any remaining bulk buffer
    tree.flush_bulk_buffer()
    
    return tree


def create_large_scale_tree(
    size: int = 10000,
    config: Optional[TreeConfig] = None
) -> Tree:
    """
    Create a large-scale tree for performance testing.
    
    Args:
        size: Number of items to insert
        config: Tree configuration
        
    Returns:
        A tree with large-scale test data
    """
    import random
    
    if config is None:
        config = TreeConfig(
            max_capacity=128,
            min_capacity=64,
            split_strategy=SplitStrategy.ADAPTIVE,
            merge_strategy=MergeStrategy.THRESHOLD,
            cache_size=1000,
            bulk_load_threshold=100
        )
    
    tree = Tree(config)
    
    # Generate random data
    keys = list(range(size))
    random.shuffle(keys)
    
    for key in keys:
        tree.insert(key, f"value_{key}")
    
    tree.flush_bulk_buffer()
    return tree


def benchmark_tree_operations(
    num_operations: int = 1000,
    config: Optional[TreeConfig] = None
) -> Dict[str, Any]:
    """
    Benchmark tree operations for performance testing with enhanced metrics.
    
    Args:
        num_operations: Number of operations to perform
        config: Tree configuration
        
    Returns:
        Dictionary with comprehensive timing and performance results
    """
    import time
    import random
    
    if config is None:
        config = TreeConfig(max_capacity=64, cache_size=500)
    
    tree = Tree(config)
    results = {}
    
    # Benchmark insertions
    keys = list(range(num_operations))
    random.shuffle(keys)
    
    start_time = time.time()
    for key in keys:
        tree.insert(key, f"value_{key}")
    tree.flush_bulk_buffer()
    insert_time = time.time() - start_time
    results['insert_time'] = insert_time
    results['insert_ops_per_sec'] = num_operations / insert_time
    
    # Benchmark searches (with cache warming)
    search_keys = random.sample(keys, min(1000, len(keys)))
    
    # Cold cache searches
    tree._node_cache.clear()
    start_time = time.time()
    for key in search_keys:
        tree.search(key)
    cold_search_time = time.time() - start_time
    results['cold_search_time'] = cold_search_time
    results['cold_search_ops_per_sec'] = len(search_keys) / cold_search_time
    
    # Warm cache searches
    start_time = time.time()
    for key in search_keys:
        tree.search(key)
    warm_search_time = time.time() - start_time
    results['warm_search_time'] = warm_search_time
    results['warm_search_ops_per_sec'] = len(search_keys) / warm_search_time
    
    # Benchmark deletions
    delete_keys = random.sample(keys, min(1000, len(keys)))
    start_time = time.time()
    for key in delete_keys:
        tree.delete(key)
    delete_time = time.time() - start_time
    results['delete_time'] = delete_time
    results['delete_ops_per_sec'] = len(delete_keys) / delete_time
    
    # Benchmark traversal
    start_time = time.time()
    tree.inorder_traversal()
    traversal_time = time.time() - start_time
    results['traversal_time'] = traversal_time
    
    # Get tree statistics
    tree_stats = tree.get_stats()
    results.update(tree_stats)
    
    # Calculate efficiency metrics
    results['cache_efficiency'] = tree_stats['cache_hit_rate']
    results['split_ratio'] = tree_stats['total_splits'] / max(1, num_operations)
    results['merge_ratio'] = tree_stats['total_merges'] / max(1, len(delete_keys))
    
    return results


def compare_strategies(num_operations: int = 5000) -> Dict[str, Dict[str, Any]]:
    """
    Compare performance of different split and merge strategies.
    
    Args:
        num_operations: Number of operations for each test
        
    Returns:
        Dictionary with results for each strategy combination
    """
    strategies = [
        (SplitStrategy.EVEN, MergeStrategy.IMMEDIATE),
        (SplitStrategy.LEFT_HEAVY, MergeStrategy.THRESHOLD),
        (SplitStrategy.RIGHT_HEAVY, MergeStrategy.LAZY),
        (SplitStrategy.ADAPTIVE, MergeStrategy.THRESHOLD),
    ]
    
    results = {}
    
    for split_strategy, merge_strategy in strategies:
        config = TreeConfig(
            max_capacity=64,
            split_strategy=split_strategy,
            merge_strategy=merge_strategy,
            cache_size=1000
        )
        
        strategy_name = f"{split_strategy.value}_{merge_strategy.value}"
        print(f"Testing strategy: {strategy_name}")
        
        results[strategy_name] = benchmark_tree_operations(num_operations, config)
    
    return results


def analyze_memory_usage(tree: Tree) -> Dict[str, Any]:
    """
    Analyze memory usage of the tree structure.
    
    Args:
        tree: Tree to analyze
        
    Returns:
        Dictionary with memory usage statistics
    """
    import sys
    
    def get_size(obj, seen=None):
        """Calculate deep size of an object."""
        size = sys.getsizeof(obj)
        if seen is None:
            seen = set()
        
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        
        seen.add(obj_id)
        
        if isinstance(obj, dict):
            size += sum([get_size(v, seen) for v in obj.values()])
            size += sum([get_size(k, seen) for k in obj.keys()])
        elif hasattr(obj, '__dict__'):
            size += get_size(obj.__dict__, seen)
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
            size += sum([get_size(i, seen) for i in obj])
        
        return size
    
    def count_nodes(node: TreeNode) -> tuple:
        """Count leaf and internal nodes."""
        if node.is_leaf:
            return 1, 0
        
        leaf_count = 0
        internal_count = 1
        
        for child in node.children.values():
            l, i = count_nodes(child)
            leaf_count += l
            internal_count += i
        
        return leaf_count, internal_count
    
    leaf_count, internal_count = count_nodes(tree.root)
    
    return {
        'total_size_bytes': get_size(tree),
        'tree_size_bytes': get_size(tree.root),
        'cache_size_bytes': get_size(tree._node_cache),
        'buffer_size_bytes': get_size(tree._bulk_buffer),
        'leaf_nodes': leaf_count,
        'internal_nodes': internal_count,
        'average_node_size': get_size(tree.root) / max(1, leaf_count + internal_count),
        'memory_per_item': get_size(tree) / max(1, tree.size()),
    }


class TreeStats:
    """
    Enhanced class to collect and display statistics about a tree.
    """
    
    def __init__(self, tree: Tree):
        """
        Initialize with a tree to analyze.
        
        Args:
            tree: The tree to analyze
        """
        self.tree = tree
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the tree.
        
        Returns:
            Dictionary with various statistics
        """
        stats = {
            'size': self.tree.size(),
            'height': self.tree.height(),
            'is_empty': self.tree.is_empty(),
            'config': self.tree.config,
            'is_valid': self.tree.validate(),
            'leaf_count': self._count_leaves(),
            'internal_count': self._count_internal_nodes(),
            'average_leaf_utilization': self._average_leaf_utilization(),
            'average_internal_utilization': self._average_internal_utilization(),
            'hot_nodes': self._count_hot_nodes(),
            'tree_stats': self.tree.get_stats(),
        }
        return stats
    
    def _count_leaves(self) -> int:
        """Count the number of leaf nodes."""
        return self._count_leaves_helper(self.tree.root)
    
    def _count_leaves_helper(self, node: TreeNode) -> int:
        """Helper method to count leaves."""
        if node.is_leaf:
            return 1
        
        count = 0
        for child in node.children.values():
            count += self._count_leaves_helper(child)
        return count
    
    def _count_internal_nodes(self) -> int:
        """Count the number of internal nodes."""
        return self._count_internal_helper(self.tree.root)
    
    def _count_internal_helper(self, node: TreeNode) -> int:
        """Helper method to count internal nodes."""
        if node.is_leaf:
            return 0
        
        count = 1  # Count this internal node
        for child in node.children.values():
            count += self._count_internal_helper(child)
        return count
    
    def _average_leaf_utilization(self) -> float:
        """Calculate average utilization of leaf nodes."""
        leaf_sizes = []
        self._collect_leaf_sizes(self.tree.root, leaf_sizes)
        
        if not leaf_sizes:
            return 0.0
        
        avg_size = sum(leaf_sizes) / len(leaf_sizes)
        return avg_size / self.tree.config.max_capacity
    
    def _collect_leaf_sizes(self, node: TreeNode, sizes: List[int]) -> None:
        """Collect sizes of all leaf nodes."""
        if node.is_leaf:
            sizes.append(len(node.data))
        else:
            for child in node.children.values():
                self._collect_leaf_sizes(child, sizes)
    
    def _average_internal_utilization(self) -> float:
        """Calculate average utilization of internal nodes."""
        internal_sizes = []
        self._collect_internal_sizes(self.tree.root, internal_sizes)
        
        if not internal_sizes:
            return 0.0
        
        avg_size = sum(internal_sizes) / len(internal_sizes)
        return avg_size / self.tree.config.max_capacity
    
    def _collect_internal_sizes(self, node: TreeNode, sizes: List[int]) -> None:
        """Collect sizes of all internal nodes."""
        if not node.is_leaf:
            sizes.append(len(node.children))
            for child in node.children.values():
                self._collect_internal_sizes(child, sizes)
    
    def _count_hot_nodes(self) -> int:
        """Count nodes that are frequently accessed."""
        return self._count_hot_helper(self.tree.root)
    
    def _count_hot_helper(self, node: TreeNode) -> int:
        """Helper to count hot nodes."""
        count = 1 if node.stats.is_hot else 0
        
        if not node.is_leaf:
            for child in node.children.values():
                count += self._count_hot_helper(child)
        
        return count
    
    def print_stats(self) -> None:
        """Print formatted statistics."""
        stats = self.get_stats()
        tree_stats = stats['tree_stats']
        
        print("Enhanced Tree Statistics:")
        print(f"  Size: {stats['size']}")
        print(f"  Height: {stats['height']}")
        print(f"  Configuration: {stats['config']}")
        print(f"  Is Empty: {stats['is_empty']}")
        print(f"  Is Valid: {stats['is_valid']}")
        print(f"  Leaf Nodes: {stats['leaf_count']}")
        print(f"  Internal Nodes: {stats['internal_count']}")
        print(f"  Hot Nodes: {stats['hot_nodes']}")
        print(f"  Avg Leaf Utilization: {stats['average_leaf_utilization']:.2%}")
        print(f"  Avg Internal Utilization: {stats['average_internal_utilization']:.2%}")
        print(f"  Cache Hit Rate: {tree_stats['cache_hit_rate']:.2%}")
        print(f"  Total Splits: {tree_stats['total_splits']}")
        print(f"  Total Merges: {tree_stats['total_merges']}")
        print(f"  Cache Size: {tree_stats['cache_size']}")
        print(f"  Buffer Size: {tree_stats['buffer_size']}")


class AsyncTreeWrapper:
    """
    Async wrapper for Tree class using __getattribute__ to automatically
    wrap methods for async execution with batch operation support.
    """
    
    def __init__(self, tree: Tree):
        """Initialize async wrapper."""
        # Use object.__setattr__ to avoid triggering our custom __setattr__
        object.__setattr__(self, '_tree', tree)
        object.__setattr__(self, '_executor', None)
        object.__setattr__(self, '_semaphore', None)
        object.__setattr__(self, '_batch_threshold', 10)  # Threshold for using batch operations
        
        # Initialize executor and semaphore based on config
        if hasattr(tree, 'config') and tree.config and tree.config.async_io_config:
            max_workers = tree.config.async_io_config.max_workers
            max_concurrent = tree.config.async_io_config.max_concurrent_operations
        else:
            max_workers = 4
            max_concurrent = 10
            
        object.__setattr__(self, '_executor', ThreadPoolExecutor(max_workers=max_workers))
        object.__setattr__(self, '_semaphore', asyncio.Semaphore(max_concurrent))
        
        # Define batch-capable methods
        object.__setattr__(self, '_batch_methods', {
            'search': 'batch_search',
            'delete': 'batch_delete', 
            'query': 'batch_query',
            'exists': 'batch_exists'
        })
        
        # Define async-native methods that should not be wrapped
        object.__setattr__(self, '_async_methods', {
            'async_insert', 'async_insert_batch', 'async_search', 'async_query',
            'async_delete', 'async_save_tree', 'async_load_tree', 'async_backup',
            'start_async_operations', 'stop_async_operations'
        })
    
    def __getattribute__(self, name: str):
        """
        Intercept attribute access to provide async versions of methods.
        Supports batch operations for better performance.
        """
        # First, get private attributes directly
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        
        # Special methods that should be handled directly
        if name in ('cleanup', '__aenter__', '__aexit__'):
            return object.__getattribute__(self, name)
        
        # Special handling for certain attributes
        if name in ('config', 'root', 'storage_manager'):
            tree = object.__getattribute__(self, '_tree')
            return getattr(tree, name)
        
        tree = object.__getattribute__(self, '_tree')
        async_methods = object.__getattribute__(self, '_async_methods')
        batch_methods = object.__getattribute__(self, '_batch_methods')
        
        # If it's an async-native method, return it directly
        if name in async_methods:
            return getattr(tree, name)
        
        # Get the original attribute
        attr = getattr(tree, name)
        
        # If it's not callable, return as-is
        if not callable(attr):
            return attr
        
        # If it's a batch-capable method, create enhanced async wrapper
        if name in batch_methods:
            return self._make_batch_async_wrapper(attr, batch_methods[name])
        
        # For other callable methods, create regular async wrapper
        return self._make_async_wrapper(attr)
    
    def _make_batch_async_wrapper(self, method, batch_method_name):
        """
        Create an async wrapper that can use batch operations for better performance.
        """
        async def batch_async_wrapper(*args, **kwargs):
            semaphore = object.__getattribute__(self, '_semaphore')
            executor = object.__getattribute__(self, '_executor')
            tree = object.__getattribute__(self, '_tree')
            
            async with semaphore:
                # For single operations, use regular method
                if len(args) <= 1:
                    return await asyncio.get_event_loop().run_in_executor(
                        executor, method, *args, **kwargs
                    )
                
                # For multiple operations, try to use batch method if available
                try:
                    batch_method = getattr(tree, batch_method_name)
                    # Assume first argument is a list for batch operations
                    if isinstance(args[0], (list, tuple)) and len(args[0]) > 1:
                        return await asyncio.get_event_loop().run_in_executor(
                            executor, batch_method, *args, **kwargs
                        )
                except AttributeError:
                    pass
                
                # Fallback to regular method
                return await asyncio.get_event_loop().run_in_executor(
                    executor, method, *args, **kwargs
                )
        
        return batch_async_wrapper
    
    def _make_async_wrapper(self, method):
        """
        Create an async wrapper for a synchronous method with concurrency control.
        """
        async def async_wrapper(*args, **kwargs):
            semaphore = object.__getattribute__(self, '_semaphore')
            executor = object.__getattribute__(self, '_executor')
            
            async with semaphore:
                # Run the synchronous method in a thread pool to avoid blocking
                return await asyncio.get_event_loop().run_in_executor(
                    executor, method, *args, **kwargs
                )
        
        return async_wrapper
    
    def __setattr__(self, name: str, value):
        """
        Set attributes on the wrapped tree.
        
        Args:
            name: Attribute name
            value: Attribute value
        """
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, '_tree'), name, value)
    
    def __repr__(self) -> str:
        """String representation of the async wrapper."""
        tree = object.__getattribute__(self, '_tree')
        return f"AsyncTreeWrapper({repr(tree)})"
    
    async def __aenter__(self):
        """Async context manager entry."""
        tree = object.__getattribute__(self, '_tree')
        if hasattr(tree, 'start_async_operations'):
            tree.start_async_operations()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
        tree = object.__getattribute__(self, '_tree')
        if hasattr(tree, 'stop_async_operations'):
            tree.stop_async_operations()
    
    async def cleanup(self):
        """Clean up resources."""
        executor = object.__getattribute__(self, '_executor')
        if executor:
            executor.shutdown(wait=True)


def create_async_tree(config: Optional[TreeConfig] = None) -> AsyncTreeWrapper:
    """
    Factory function to create an async-enabled tree.
    
    Args:
        config: Tree configuration. If None, creates default config with async enabled.
        
    Returns:
        AsyncTreeWrapper instance
        
    Example:
        # Create with default async config
        async_tree = create_async_tree()
        
        # Create with custom config
        config = TreeConfig(
            max_capacity=64,
            enable_async_mode=True,
            async_io_config=AsyncIOConfig(max_workers=8)
        )
        async_tree = create_async_tree(config)
    """
    if config is None:
        # Create default config with async enabled
        config = TreeConfig(
            enable_async_mode=True,
            async_io_config=AsyncIOConfig()
        )
    elif not config.enable_async_mode:
        # Force enable async mode
        config.enable_async_mode = True
        if config.async_io_config is None:
            config.async_io_config = AsyncIOConfig()
    
    # Create tree - will automatically return AsyncTreeWrapper due to enable_async_mode=True
    tree = Tree(config)
    
    # Ensure we return AsyncTreeWrapper
    if isinstance(tree, AsyncTreeWrapper):
        return tree
    else:
        return AsyncTreeWrapper(tree)


# Example usage function
async def demo_async_tree():
    """Demonstrate async tree functionality."""
    print("=== Async Tree Demo ===")
    
    # Create async tree
    async_tree = create_async_tree()
    
    async with async_tree:
        # Insert data asynchronously
        await async_tree.insert(1, "first")
        await async_tree.insert(2, "second") 
        await async_tree.insert(3, "third")
        
        # Search asynchronously
        result = await async_tree.search(2)
        print(f"Async search result: {result}")
        
        # Query asynchronously
        query = QueryBuilder().value_contains("i").build()
        results = await async_tree.query(query)
        print(f"Async query results: {results}")
        
        # Get stats synchronously (no async needed)
        stats = async_tree.get_stats()
        print(f"Tree stats: {stats}")
        
        # Backup asynchronously
        backup_id = await async_tree.backup()
        print(f"Backup created: {backup_id}")


if __name__ == "__main__":
    # Run async demo
    asyncio.run(demo_async_tree())
        
        
        