#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ION Helper Library

提供ION系列文件的辅助功能，包括：
1. 异步任务管理系统（AsyncTaskPoolMixin、AsyncTaskTeam、AsyncTask）
2. 索引处理函数生成器
3. BaseMixin抽象基类

后续导入ion.py
如果你需要创建一个用于导入到ion.py的文件，请使用这个文件，可以有Async功能和教程
"""

import nest_asyncio
nest_asyncio.apply() # spyder 需要

import asyncio
import enum
import uuid
import weakref
import logging
import time
import inspect
from abc import ABC, abstractmethod
from typing import Optional, Dict, Set, Any, Callable, Union, List, Tuple
from functools import wraps
from collections import defaultdict

# ============= 异步任务系统 =============

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


# ============= 索引处理函数生成器 =============

class IndexingHandlerGenerator:
    """索引处理函数生成器"""
    
    @staticmethod
    def generate_btree_handler(index_methods: Dict[str, List[str]]) -> Callable:
        """
        生成B-tree索引处理函数
        
        Parameters
        ----------
        index_methods : Dict[str, List[str]]
            索引方法映射，格式：{
                'create': ['create_node', 'batch_create_nodes'],
                'update': ['update_node_value', 'update_node_metadata'],
                'delete': ['remove_node_by_key', 'remove_node_by_key_base']
            }
        
        Returns
        -------
        Callable
            生成的处理函数
        """
        def handle_btree_indexing(self, method_name: str, args: tuple, kwargs: dict, result: Any):
            """处理B-tree索引的统一方法"""
            if not hasattr(self, 'btree_manager') or not self.btree_manager:
                return
            
            try:
                # 节点创建
                if method_name in index_methods.get('create', []):
                    if result and hasattr(result, 'key'):
                        if hasattr(self, 'smart_btree_insert_node'):
                            self.smart_btree_insert_node(result, method_context='create')
                            
                # 节点删除
                elif method_name in index_methods.get('delete', []):
                    if args:
                        node = self.get_node_by_key(args[0]) if hasattr(self, 'get_node_by_key') else None
                        if node and hasattr(self, 'smart_btree_remove_node'):
                            self.smart_btree_remove_node(node, method_context='delete')
                            
                # 节点更新
                elif method_name in index_methods.get('update', []):
                    if args and hasattr(args[0], 'key'):
                        node = args[0]
                        if hasattr(self, 'smart_btree_remove_node') and hasattr(self, 'smart_btree_insert_node'):
                            self.smart_btree_remove_node(node, method_context='update_pre')
                            self.smart_btree_insert_node(node, method_context='update_post')
                            
                # 批量操作
                elif method_name in index_methods.get('batch', []):
                    if isinstance(result, list):
                        for item in result:
                            if hasattr(item, 'key') and hasattr(self, 'smart_btree_insert_node'):
                                self.smart_btree_insert_node(item, method_context='batch')
                                
            except Exception as e:
                logging.warning(f"B-tree自动索引处理失败: {e}")
        
        return handle_btree_indexing
    
    @staticmethod
    def generate_mbtree_handler(index_methods: Dict[str, List[str]]) -> Callable:
        """
        生成MBTree索引处理函数
        
        Parameters
        ----------
        index_methods : Dict[str, List[str]]
            索引方法映射
        
        Returns
        -------
        Callable
            生成的处理函数
        """
        def handle_mbtree_indexing(self, method_name: str, args: tuple, kwargs: dict, result: Any):
            """处理多路树索引的自动更新"""
            if not hasattr(self, 'mbtree_manager') or not self.mbtree_manager:
                return
            
            try:
                # 节点创建方法
                if method_name in index_methods.get('create', []):
                    if result and hasattr(result, 'key'):
                        if hasattr(self, 'smart_mbtree_insert_node'):
                            self.smart_mbtree_insert_node(result, method_context='create')
                
                # 节点删除方法  
                elif method_name in index_methods.get('delete', []):
                    if args and hasattr(args[0], 'key'):
                        node = args[0]
                        if hasattr(self, 'smart_mbtree_remove_node'):
                            self.smart_mbtree_remove_node(node, method_context='delete')
                
                # 节点修改方法
                elif method_name in index_methods.get('update', []):
                    if args and hasattr(args[0], 'key'):
                        node = args[0]
                        if hasattr(self, 'smart_mbtree_remove_node') and hasattr(self, 'smart_mbtree_insert_node'):
                            self.smart_mbtree_remove_node(node, method_context='update_pre')
                            self.smart_mbtree_insert_node(node, method_context='update_post')
                
                # 批量操作
                elif method_name in index_methods.get('batch', []):
                    if isinstance(result, list):
                        for item in result:
                            if hasattr(item, 'key') and hasattr(self, 'smart_mbtree_insert_node'):
                                self.smart_mbtree_insert_node(item, method_context='batch')
            except Exception as e:
                logging.warning(f"多路树自动索引处理失败: {e}")

        return handle_mbtree_indexing
    
    @staticmethod
    def generate_custom_handler(index_type: str, 
                              handler_method_name: str,
                              index_methods: Dict[str, List[str]]) -> Callable:
        """
        生成自定义索引处理函数
        
        Parameters
        ----------
        index_type : str
            索引类型名称
        handler_method_name : str
            处理方法名称模式，如'smart_{}_insert_node'
        index_methods : Dict[str, List[str]]
            索引方法映射
        
        Returns
        -------
        Callable
            生成的处理函数
        """
        def handle_custom_indexing(self, method_name: str, args: tuple, kwargs: dict, result: Any):
            f"""处理{index_type}索引的自动更新"""
            manager_attr = f'{index_type.lower()}_manager'
            if not hasattr(self, manager_attr) or not getattr(self, manager_attr):
                return
            
            try:
                # 动态生成方法名
                insert_method = handler_method_name.format(index_type.lower(), 'insert')
                remove_method = handler_method_name.format(index_type.lower(), 'remove')
                
                # 节点创建方法
                if method_name in index_methods.get('create', []):
                    if result and hasattr(result, 'key'):
                        if hasattr(self, insert_method):
                            getattr(self, insert_method)(result, method_context='create')
                
                # 节点删除方法  
                elif method_name in index_methods.get('delete', []):
                    if args and hasattr(args[0], 'key'):
                        node = args[0]
                        if hasattr(self, remove_method):
                            getattr(self, remove_method)(node, method_context='delete')
                
                # 节点修改方法
                elif method_name in index_methods.get('update', []):
                    if args and hasattr(args[0], 'key'):
                        node = args[0]
                        if hasattr(self, remove_method) and hasattr(self, insert_method):
                            getattr(self, remove_method)(node, method_context='update_pre')
                            getattr(self, insert_method)(node, method_context='update_post')
                
                # 批量操作
                elif method_name in index_methods.get('batch', []):
                    if isinstance(result, list):
                        for item in result:
                            if hasattr(item, 'key') and hasattr(self, insert_method):
                                getattr(self, insert_method)(item, method_context='batch')
            except Exception as e:
                logging.warning(f"{index_type}自动索引处理失败: {e}")

        return handle_custom_indexing
    
    @classmethod
    def create_default_handlers(cls) -> Dict[str, Callable]:
        """创建默认的索引处理函数集合"""
        # 默认方法映射
        default_btree_methods = {
            'create': ['create_node', 'batch_create_nodes', 'batch_create_table_nodes'],
            'delete': ['remove_node_by_key', 'remove_node_by_key_base'],
            'update': ['update_node_value', 'update_node_metadata', 'update_node_tag', 
                      'update_node_weight', 'update_node_relations', 'update_node_row',
                      'update_node_val_locker'],
            'batch': ['batch_create_nodes', 'batch_insert']
        }
        
        default_mbtree_methods = {
            'create': ['create_node', 'batch_create_nodes', 'batch_create_table_nodes'],
            'delete': ['remove_node_by_key', 'remove_node_by_key_base'],
            'update': ['update_node_value', 'update_node_metadata', 'update_node_tag', 
                      'update_node_weight', 'update_node_relations'],
            'batch': ['batch_create_nodes', 'batch_insert']
        }
        
        return {
            'btree': cls.generate_btree_handler(default_btree_methods),
            'mbtree': cls.generate_mbtree_handler(default_mbtree_methods)
        }


# ============= BaseMixin抽象基类 =============

class BaseMixin(ABC):
    """ION系列的基础混入抽象类"""
    
    @abstractmethod
    def init_storage(self, **kwargs):
        """初始化存储系统"""
        pass
    
    @abstractmethod
    def create_node(self, key: Any, val: Any, **kwargs) -> Any:
        """创建节点"""
        pass
    
    @abstractmethod
    def get_node_by_key(self, key: Any) -> Optional[Any]:
        """通过键获取节点"""
        pass
    
    @abstractmethod
    def remove_node_by_key(self, key: Any) -> bool:
        """通过键删除节点"""
        pass
    
    @abstractmethod
    def update_node_value(self, node: Any, new_value: Any) -> bool:
        """更新节点值"""
        pass
    
    def get_all_keys(self) -> List[Any]:
        """获取所有键（可选实现）"""
        raise NotImplementedError("Subclass should implement get_all_keys if needed")
    
    def size(self) -> int:
        """获取存储大小（可选实现）"""
        raise NotImplementedError("Subclass should implement size if needed")
    
    def is_empty(self) -> bool:
        """检查是否为空（可选实现）"""
        return self.size() == 0


class IndexableMixin(BaseMixin):
    """可索引的混入类"""
    
    @abstractmethod
    def init_indexing(self, **kwargs):
        """初始化索引系统"""
        pass
    
    @abstractmethod
    def create_index(self, index_name: str, field_name: str, **kwargs) -> bool:
        """创建索引"""
        pass
    
    @abstractmethod
    def drop_index(self, index_name: str) -> bool:
        """删除索引"""
        pass
    
    def rebuild_index(self, index_name: str) -> bool:
        """重建索引（可选实现）"""
        if self.drop_index(index_name):
            # 这里需要子类提供重建逻辑
            return True
        return False


class SearchableMixin(BaseMixin):
    """可搜索的混入类"""
    
    @abstractmethod
    def search(self, query: Any, **kwargs) -> List[Any]:
        """搜索"""
        pass
    
    def find_by_value(self, value: Any) -> List[Any]:
        """通过值查找（可选实现）"""
        raise NotImplementedError("Subclass should implement find_by_value if needed")
    
    def find_by_pattern(self, pattern: str) -> List[Any]:
        """通过模式查找（可选实现）"""
        raise NotImplementedError("Subclass should implement find_by_pattern if needed")


class TransactionalMixin(BaseMixin):
    """事务性混入类"""
    
    @abstractmethod
    def begin_transaction(self) -> Any:
        """开始事务"""
        pass
    
    @abstractmethod
    def commit_transaction(self, transaction_id: Any) -> bool:
        """提交事务"""
        pass
    
    @abstractmethod
    def rollback_transaction(self, transaction_id: Any) -> bool:
        """回滚事务"""
        pass


class ComprehensiveMixin(IndexableMixin, SearchableMixin, TransactionalMixin, AsyncTaskPoolMixin):
    """综合混入类，包含所有功能"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 初始化异步任务系统
        if hasattr(self, 'init_async_task'):
            self.init_async_task()
    
    def init_all_systems(self, **kwargs):
        """初始化所有系统"""
        self.init_storage(**kwargs)
        self.init_indexing(**kwargs)
        if hasattr(self, 'init_async_task'):
            self.init_async_task()


# ============= 工具函数 =============

def create_handler_factory(handler_generators: Dict[str, Callable] = None) -> Callable:
    """
    创建处理函数工厂
    
    Parameters
    ----------
    handler_generators : Dict[str, Callable], optional
        处理函数生成器映射
    
    Returns
    -------
    Callable
        处理函数工厂
    """
    if handler_generators is None:
        handler_generators = IndexingHandlerGenerator.create_default_handlers()
    
    def factory(handler_type: str, **kwargs) -> Optional[Callable]:
        """
        创建指定类型的处理函数
        
        Parameters
        ----------
        handler_type : str
            处理函数类型
        **kwargs : dict
            传递给生成器的参数
        
        Returns
        -------
        Optional[Callable]
            生成的处理函数
        """
        if handler_type in handler_generators:
            return handler_generators[handler_type]
        return None
    
    return factory


def async_method_decorator(func: Callable) -> Callable:
    """
    异步方法装饰器，自动处理同步/异步执行
    
    Parameters
    ----------
    func : Callable
        要装饰的函数
    
    Returns
    -------
    Callable
        装饰后的函数
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # 检查是否启用异步模式
        if hasattr(self, 'enable_all_async') and self.enable_all_async:
            # 创建异步任务
            if asyncio.iscoroutinefunction(func):
                return self.create_task(func(self, *args, **kwargs))
            else:
                async def async_wrapper():
                    return func(self, *args, **kwargs)
                return self.create_task(async_wrapper())
        else:
            # 同步执行
            return func(self, *args, **kwargs)
    
    return wrapper


def batch_operation_decorator(batch_size: int = 100):
    """
    批量操作装饰器
    
    Parameters
    ----------
    batch_size : int
        批量大小
    
    Returns
    -------
    Callable
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, items: List[Any], *args, **kwargs):
            if len(items) <= batch_size:
                return func(self, items, *args, **kwargs)
            
            # 分批处理
            results = []
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                batch_result = func(self, batch, *args, **kwargs)
                if isinstance(batch_result, list):
                    results.extend(batch_result)
                else:
                    results.append(batch_result)
            
            return results
        
        return wrapper
    return decorator


# ============= 导出 =============

__all__ = [
    # 异步任务系统
    'AsyncTask',
    'AsyncTaskTeam', 
    'AsyncTaskPoolMixin',
    'TaskStatus',
    
    # 索引处理
    'IndexingHandlerGenerator',
    
    # 基础混入类
    'BaseMixin',
    'IndexableMixin',
    'SearchableMixin',
    'TransactionalMixin',
    'ComprehensiveMixin',
    
    # 工具函数
    'create_handler_factory',
    'async_method_decorator',
    'batch_operation_decorator',
]


# ============= 使用示例 =============

if __name__ == "__main__":
    print("ION Helper Library - 使用示例")
    
    # 1. 索引处理函数生成器示例
    print("\n1. 索引处理函数生成器示例:")
    
    custom_methods = {
        'create': ['my_create_method'],
        'update': ['my_update_method'],
        'delete': ['my_delete_method']
    }
    
    btree_handler = IndexingHandlerGenerator.generate_btree_handler(custom_methods)
    print(f"生成的B-tree处理函数: {btree_handler}")
    
    # 2. 异步任务系统示例
    print("\n2. 异步任务系统示例:")
    
    class TestClass(AsyncTaskPoolMixin):
        def __init__(self):
            self.init_async_task()
        
        @async_method_decorator
        def test_method(self, x: int, y: int) -> int:
            return x + y
    
    # 创建测试实例
    test_obj = TestClass()
    
    # 使用装饰器创建任务
    @test_obj.task(name="AddTask")
    async def add_numbers(a: int, b: int) -> int:
        await asyncio.sleep(0.1)  # 模拟异步操作
        return a + b
    
    print("异步任务系统初始化完成")
    print(f"默认团队: {test_obj.default_team}")
    
    print("\nION Helper Library 示例完成!") 
