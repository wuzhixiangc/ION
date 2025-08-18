# distutils: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

"""
ION Expression - 完整NBT系统实现
支持所有NBT标签类型和ION函数
"""

import uuid
import json
import re
import struct
import sys
sys.path.append("/")
from typing import Any, Dict, List, Optional, Union
from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF, Py_DECREF

# C库导入
from libc.stdint cimport uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t
from libc.stdlib cimport malloc, free, realloc
from libc.string cimport memcpy, strlen, memset

HAS_BPTREE = True
# distutils: language = c
# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False

from cpython.ref cimport Py_XINCREF, Py_XDECREF
from cpython.object cimport PyObject, Py_LT, Py_EQ, PyObject_RichCompareBool
from libc.stddef cimport size_t
from libc.stdlib cimport malloc, free, realloc
from libc.string cimport memset, memmove
from cython cimport NULL

ctypedef struct BPTNode:
    bint leaf
    int nkeys
    int order
    PyObject **keys          # 引用持有
    PyObject **values        # 仅叶子有效，引用持有
    BPTNode **children       # 仅内部有效
    BPTNode *next_leaf       # 叶子链

cdef inline void* xmalloc(size_t n):
    cdef void* p
    p = malloc(n)
    if p == NULL:
        raise MemoryError()
    return p

cdef inline void* xrealloc(void* p, size_t n):
    cdef void* q
    q = realloc(p, n)
    if q == NULL:
        raise MemoryError()
    return q

cdef inline int pykey_lt(PyObject* a, PyObject* b):
    # 必须把 PyObject* 显式转为 <object> 再比较
    return PyObject_RichCompareBool(<object>a, <object>b, Py_LT)

cdef inline int pykey_eq(PyObject* a, PyObject* b):
    return PyObject_RichCompareBool(<object>a, <object>b, Py_EQ)

cdef inline void shift_right_ptrs(void** base, size_t stride, int pos, int n):
    cdef size_t count
    if n - pos > 0:
        count = (n - pos) * stride
        memmove(<char*>base + (pos + 1) * stride, <char*>base + pos * stride, count)

cdef inline void shift_left_ptrs(void** base, size_t stride, int pos, int n):
    cdef size_t count
    if n - pos - 1 > 0:
        count = (n - pos - 1) * stride
        memmove(<char*>base + pos * stride, <char*>base + (pos + 1) * stride, count)

cdef BPTNode* node_new(bint leaf, int order):
    cdef BPTNode* n
    cdef size_t szk
    cdef size_t szv
    cdef size_t szc
    n = <BPTNode*> xmalloc(sizeof(BPTNode))
    n.leaf = leaf
    n.nkeys = 0
    n.order = order
    n.next_leaf = NULL
    szk = <size_t> order * sizeof(PyObject*)
    n.keys = <PyObject**> xmalloc(szk)
    memset(n.keys, 0, szk)
    if leaf:
        szv = <size_t> order * sizeof(PyObject*)
        n.values = <PyObject**> xmalloc(szv)
        memset(n.values, 0, szv)
        n.children = NULL
    else:
        n.values = NULL
        szc = <size_t> (order + 1) * sizeof(BPTNode*)
        n.children = <BPTNode**> xmalloc(szc)
        memset(n.children, 0, szc)
    return n

cdef void node_free(BPTNode* n):
    cdef int i
    if n == NULL:
        return
    if n.leaf:
        for i in range(n.nkeys):
            if n.keys[i] != NULL:
                Py_XDECREF(n.keys[i])
            if n.values[i] != NULL:
                Py_XDECREF(n.values[i])
        if n.values != NULL:
            free(n.values)
    else:
        for i in range(n.nkeys + 1):
            if n.children[i] != NULL:
                node_free(n.children[i])
        if n.children != NULL:
            free(n.children)
        for i in range(n.nkeys):
            if n.keys[i] != NULL:
                Py_XDECREF(n.keys[i])
    if n.keys != NULL:
        free(n.keys)
    free(n)

cdef int key_lower_bound(BPTNode* n, PyObject* key):
    cdef int lo
    cdef int hi
    cdef int mid
    lo = 0
    hi = n.nkeys
    while lo < hi:
        mid = (lo + hi) >> 1
        if pykey_lt(n.keys[mid], key) == 1:
            lo = mid + 1
        else:
            hi = mid
    return lo

cdef BPTNode* find_leaf(BPTNode* root, PyObject* key):
    cdef BPTNode* n
    cdef int pos
    n = root
    while n != NULL and not n.leaf:
        pos = key_lower_bound(n, key)
        n = n.children[pos]
    return n

cdef int leaf_find_value(BPTNode* leaf, PyObject* key, PyObject** out):
    cdef int pos
    cdef int eq
    pos = key_lower_bound(leaf, key)
    if pos < leaf.nkeys:
        eq = pykey_eq(leaf.keys[pos], key)
        if eq == 1:
            out[0] = leaf.values[pos]
            return 1
    return 0

cdef void leaf_insert_at(BPTNode* leaf, int pos, PyObject* key, PyObject* val):
    cdef int n
    n = leaf.nkeys
    shift_right_ptrs(<void**> leaf.keys, sizeof(PyObject*), pos, n)
    shift_right_ptrs(<void**> leaf.values, sizeof(PyObject*), pos, n)
    Py_XINCREF(key)
    Py_XINCREF(val)
    leaf.keys[pos] = key
    leaf.values[pos] = val
    leaf.nkeys = n + 1

# incref_key=True: 父节点需要对 key 拥有独立引用（叶子分裂）
# incref_key=False: 提升键从子内部结点“移交”，不增引用（内部结点分裂）
cdef void internal_insert_at(BPTNode* node, int pos, PyObject* key, BPTNode* right, bint incref_key):
    cdef int n
    n = node.nkeys
    shift_right_ptrs(<void**> node.keys, sizeof(PyObject*), pos, n)
    shift_right_ptrs(<void**> node.children, sizeof(BPTNode*), pos + 1, n + 1)
    if incref_key:
        Py_XINCREF(key)
    node.keys[pos] = key
    node.children[pos + 1] = right
    node.nkeys = n + 1

cdef void leaf_replace_value(BPTNode* leaf, int pos, PyObject* val_new):
    cdef PyObject* vold
    vold = leaf.values[pos]
    Py_XINCREF(val_new)
    leaf.values[pos] = val_new
    if vold != NULL:
        Py_XDECREF(vold)

cdef void leaf_split(BPTNode* leaf, PyObject** up_key, BPTNode** out_right):
    cdef int order
    cdef int mid
    cdef int i
    cdef BPTNode* right
    order = leaf.order
    mid = leaf.nkeys // 2
    right = node_new(1, order)
    for i in range(mid, leaf.nkeys):
        right.keys[i - mid] = leaf.keys[i]
        right.values[i - mid] = leaf.values[i]
        leaf.keys[i] = NULL
        leaf.values[i] = NULL
    right.nkeys = leaf.nkeys - mid
    leaf.nkeys = mid
    right.next_leaf = leaf.next_leaf
    leaf.next_leaf = right
    # 父与右叶子共享该 key -> 父需 INCREF（由上层控制 incref）
    up_key[0] = right.keys[0]
    out_right[0] = right

cdef void internal_split(BPTNode* node, PyObject** up_key, BPTNode** out_right):
    cdef int mid
    cdef int i
    cdef BPTNode* right
    mid = node.nkeys // 2
    right = node_new(0, node.order)
    for i in range(mid + 1, node.nkeys + 1):
        right.children[i - (mid + 1)] = node.children[i]
        node.children[i] = NULL
    for i in range(mid + 1, node.nkeys):
        right.keys[i - (mid + 1)] = node.keys[i]
        node.keys[i] = NULL
    right.nkeys = node.nkeys - (mid + 1)
    # 提升的中位键从当前结点“移交”到父节点 -> 不 INCREF
    up_key[0] = node.keys[mid]
    node.keys[mid] = NULL
    node.nkeys = mid
    out_right[0] = right

cdef int insert_recursive(BPTNode* n, PyObject* key, PyObject* val,
                          int order, PyObject** up_key, BPTNode** out_right, bint* up_from_leaf):
    cdef int pos
    cdef int eq
    cdef PyObject* upk_child
    cdef BPTNode* right_child
    cdef BPTNode* child
    cdef int rc

    if n.leaf:
        pos = key_lower_bound(n, key)
        if pos < n.nkeys:
            eq = pykey_eq(n.keys[pos], key)
            if eq == 1:
                leaf_replace_value(n, pos, val)
                return 0
        leaf_insert_at(n, pos, key, val)
        if n.nkeys >= order:
            leaf_split(n, up_key, out_right)
            up_from_leaf[0] = 1
            return 1
        return 0
    else:
        pos = key_lower_bound(n, key)
        child = n.children[pos]
        rc = insert_recursive(child, key, val, order, &upk_child, &right_child, up_from_leaf)
        if rc == 1:
            internal_insert_at(n, pos, upk_child, right_child, up_from_leaf[0])
            if n.nkeys >= order:
                internal_split(n, up_key, out_right)
                up_from_leaf[0] = 0  # 这一层分裂是内部结点
                return 1
            return 0
        elif rc == -1:
            return -1
        else:
            return 0

cdef int leaf_delete(BPTNode* leaf, PyObject* key):
    cdef int pos
    cdef int eq
    pos = key_lower_bound(leaf, key)
    if pos < leaf.nkeys:
        eq = pykey_eq(leaf.keys[pos], key)
        if eq == 1:
            Py_XDECREF(leaf.keys[pos])
            Py_XDECREF(leaf.values[pos])
            shift_left_ptrs(<void**> leaf.keys, sizeof(PyObject*), pos, leaf.nkeys)
            shift_left_ptrs(<void**> leaf.values, sizeof(PyObject*), pos, leaf.nkeys)
            leaf.nkeys -= 1
            return 1
    return 0

cdef class BPTree:
    cdef BPTNode* root
    cdef int order

    def __cinit__(self, int order=32):
        cdef int o
        if order < 3:
            o = 3
        else:
            o = order
        self.order = o
        self.root = node_new(1, o)

    def __dealloc__(self):
        node_free(self.root)
        self.root = NULL

    cpdef insert(self, object key, object value):
        cdef PyObject* pkey
        cdef PyObject* pvalue
        cdef PyObject* upk
        cdef BPTNode* right
        cdef int split
        cdef BPTNode* newroot
        cdef bint up_from_leaf
        # 移除可调用对象的限制，允许存储任何对象
        pkey = <PyObject*> key
        pvalue = <PyObject*> value
        split = insert_recursive(self.root, pkey, pvalue, self.order, &upk, &right, &up_from_leaf)
        if split == 1:
            newroot = node_new(0, self.order)
            if up_from_leaf:
                Py_XINCREF(upk)   # 与右叶共享 -> 父节点需要独立引用
            newroot.keys[0] = upk
            newroot.children[0] = self.root
            newroot.children[1] = right
            newroot.nkeys = 1
            self.root = newroot
        elif split == -1:
            raise MemoryError("insert failed")

    cpdef object get(self, object key, default=None):
        cdef BPTNode* leaf
        cdef PyObject* out
        cdef object obj
        leaf = find_leaf(self.root, <PyObject*> key)
        if leaf == NULL:
            return default
        if leaf_find_value(leaf, <PyObject*> key, &out) == 1:
            Py_XINCREF(out)
            obj = <object> out
            return obj
        return default

    # 特殊方法要用 def
    def __contains__(self, key):
        return self.get(key, None) is not None

    # 为兼容性改为 def（支持 *args/**kwargs）
    # 注意：此方法仅适用于可调用的值
    def call(self, object key, args, kwargs):
        cdef object f
        f = self.get(key, None)
        if f is None:
            raise KeyError(key)
        if not callable(f):
            raise TypeError(f"Value for key '{key}' is not callable")
        return f(*args, **kwargs)

    cpdef remove(self, object key):
        cdef BPTNode* leaf
        cdef int ok
        leaf = find_leaf(self.root, <PyObject*> key)
        if leaf == NULL:
            raise KeyError(key)
        ok = leaf_delete(leaf, <PyObject*> key)
        if ok == 0:
            raise KeyError(key)
        # 简化删除：不做借位/合并

    cpdef list items(self):
        cdef BPTNode* n
        cdef int i
        cdef list out
        cdef object key_obj
        cdef object value_obj
        n = self.root
        while n != NULL and not n.leaf:
            n = n.children[0]
        out = []
        while n != NULL:
            for i in range(n.nkeys):
                Py_XINCREF(n.keys[i])
                Py_XINCREF(n.values[i])
                key_obj = <object> n.keys[i]
                value_obj = <object> n.values[i]
                out.append((key_obj, value_obj))
            n = n.next_leaf
        return out

HAS_BPTREE = True

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# -*- coding: utf-8 -*-

# =============================================================================
# 低层依赖（手动内存管理 / 编解码）
# =============================================================================
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t
from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsStringAndSize
from cpython.unicode cimport PyUnicode_DecodeUTF8
cimport cython
import uuid  # 仅用于UUID对象的Python层封装/解析

# =============================================================================
# NBT Type 枚举
# =============================================================================
cdef enum NBTType:
    TAG_END = 0
    TAG_BYTE = 1
    TAG_SHORT = 2
    TAG_INT = 3
    TAG_LONG = 4
    TAG_FLOAT = 5
    TAG_DOUBLE = 6
    TAG_STRING = 7
    TAG_COMPOUND = 8
    TAG_ARRAY = 9
    TAG_FARRAY = 10
    TAG_FCOMPOUND = 11
    TAG_VAR = 12
    TAG_POINT = 13

# =============================================================================
# 可增长写缓冲 & 顺序读取器（malloc/realloc/free）
# =============================================================================
cdef struct Buf:
    unsigned char* data
    Py_ssize_t cap
    Py_ssize_t len

cdef inline void buf_init(Buf* b, Py_ssize_t initial):
    if initial < 64:
        initial = 64
    b.data = <unsigned char*>malloc(initial)
    if b.data == NULL:
        raise MemoryError("malloc failed")
    b.cap = initial
    b.len = 0

cdef inline void buf_free(Buf* b):
    if b.data != NULL:
        free(b.data)
        b.data = NULL
    b.cap = 0
    b.len = 0

cdef inline void buf_grow(Buf* b, Py_ssize_t need):
    cdef Py_ssize_t required = b.len + need
    if required <= b.cap:
        return
    cdef Py_ssize_t newcap = b.cap
    while newcap < required:
        if newcap > (1 << 60):
            raise MemoryError("buffer too large")
        newcap <<= 1
    cdef unsigned char* p = <unsigned char*>realloc(b.data, newcap)
    if p == NULL:
        raise MemoryError("realloc failed")
    b.data = p
    b.cap = newcap

cdef inline void buf_write_u8(Buf* b, uint8_t v):
    buf_grow(b, 1)
    b.data[b.len] = v
    b.len += 1

cdef inline void buf_write_u16be(Buf* b, uint16_t v):
    buf_grow(b, 2)
    b.data[b.len]   = <uint8_t>((v >> 8) & 0xFF)
    b.data[b.len+1] = <uint8_t>(v & 0xFF)
    b.len += 2

cdef inline void buf_write_u32be(Buf* b, uint32_t v):
    buf_grow(b, 4)
    b.data[b.len]   = <uint8_t>((v >> 24) & 0xFF)
    b.data[b.len+1] = <uint8_t>((v >> 16) & 0xFF)
    b.data[b.len+2] = <uint8_t>((v >> 8)  & 0xFF)
    b.data[b.len+3] = <uint8_t>(v & 0xFF)
    b.len += 4

cdef inline void buf_write_u64be(Buf* b, uint64_t v):
    buf_grow(b, 8)
    b.data[b.len]   = <uint8_t>((v >> 56) & 0xFF)
    b.data[b.len+1] = <uint8_t>((v >> 48) & 0xFF)
    b.data[b.len+2] = <uint8_t>((v >> 40) & 0xFF)
    b.data[b.len+3] = <uint8_t>((v >> 32) & 0xFF)
    b.data[b.len+4] = <uint8_t>((v >> 24) & 0xFF)
    b.data[b.len+5] = <uint8_t>((v >> 16) & 0xFF)
    b.data[b.len+6] = <uint8_t>((v >> 8)  & 0xFF)
    b.data[b.len+7] = <uint8_t>(v & 0xFF)
    b.len += 8

cdef inline void buf_write_bytes(Buf* b, const unsigned char* src, Py_ssize_t n):
    if n <= 0:
        return
    buf_grow(b, n)
    memcpy(b.data + b.len, src, n)
    b.len += n

# float/double 的位视图
ctypedef union U32F:
    uint32_t u
    float f

ctypedef union U64D:
    uint64_t u
    double d

cdef inline void buf_write_f32be(Buf* b, float f):
    cdef U32F u
    u.f = f
    buf_write_u32be(b, u.u)

cdef inline void buf_write_f64be(Buf* b, double d):
    cdef U64D u
    u.d = d
    buf_write_u64be(b, u.u)

# 读取器
cdef struct Reader:
    const unsigned char* data
    Py_ssize_t size
    Py_ssize_t pos

cdef inline void reader_from_bytes(Reader* r, bytes data):
    cdef char* p
    cdef Py_ssize_t n
    if PyBytes_AsStringAndSize(data, &p, &n) < 0:
        raise TypeError("data must be bytes")
    r.data = <const unsigned char*>p
    r.size = n
    r.pos = 0

cdef inline void need(Reader* r, Py_ssize_t n):
    if r.pos + n > r.size:
        raise ValueError("数据不完整或已越界")

cdef inline uint8_t rd_u8(Reader* r):
    need(r, 1)
    cdef uint8_t v = r.data[r.pos]
    r.pos += 1
    return v

cdef inline uint16_t rd_u16be(Reader* r):
    need(r, 2)
    cdef uint16_t v = (r.data[r.pos] << 8) | r.data[r.pos+1]
    r.pos += 2
    return v

cdef inline uint32_t rd_u32be(Reader* r):
    need(r, 4)
    cdef uint32_t v = (r.data[r.pos]   << 24) | (r.data[r.pos+1] << 16) | \
                      (r.data[r.pos+2] << 8)  |  r.data[r.pos+3]
    r.pos += 4
    return v

cdef inline uint64_t rd_u64be(Reader* r):
    need(r, 8)
    cdef uint64_t v = 0
    for i in range(8):
        v = (v << 8) | r.data[r.pos + i]
    r.pos += 8
    return v

cdef inline float rd_f32be(Reader* r):
    cdef U32F u
    u.u = rd_u32be(r)
    return u.f

cdef inline double rd_f64be(Reader* r):
    cdef U64D u
    u.u = rd_u64be(r)
    return u.d

cdef inline const unsigned char* rd_ptr(Reader* r, Py_ssize_t n):
    need(r, n)
    cdef const unsigned char* p = r.data + r.pos
    r.pos += n
    return p

cdef inline object rd_utf8_str(Reader* r):
    cdef uint16_t L = rd_u16be(r)
    cdef const unsigned char* p = rd_ptr(r, L)
    # 直接UTF-8解码为PyUnicode，避免中间bytes复制
    return <object>PyUnicode_DecodeUTF8(<const char*>p, L, "strict")

# =============================================================================
# NBT 基类 & 工具
# =============================================================================
cdef class NBTBase:
    """NBT标签基础类（接口保留；实现改为使用手动内存缓冲）"""
    
    cpdef int get_type(self):
        return TAG_END
    
    def to_dict(self):
        return {"type": self.get_type()}
    
    def to_bytes(self):
        raise NotImplementedError("子类必须实现to_bytes")
    
    @classmethod
    def from_bytes(cls, bytes data):
        raise NotImplementedError("子类必须实现from_bytes")

    cdef void _write_payload(self, Buf* b):  # 子类实现：仅写有效载荷
        raise NotImplementedError()
    
    def __str__(self):
        return f"NBTBase(type={self.get_type()})"

# 前置声明
cdef class NBTEnd
cdef class NBTByte
cdef class NBTShort
cdef class NBTInt
cdef class NBTLong
cdef class NBTFloat
cdef class NBTDouble
cdef class NBTString
cdef class NBTCompound
cdef class NBTArray
cdef class NBTFArray
cdef class NBTFCompound
cdef class NBTVar
cdef class NBTPoint

cdef object _get_nbt_class_by_type(int tag_type):
    if tag_type == TAG_END: return NBTEnd
    elif tag_type == TAG_BYTE: return NBTByte
    elif tag_type == TAG_SHORT: return NBTShort
    elif tag_type == TAG_INT: return NBTInt
    elif tag_type == TAG_LONG: return NBTLong
    elif tag_type == TAG_FLOAT: return NBTFloat
    elif tag_type == TAG_DOUBLE: return NBTDouble
    elif tag_type == TAG_STRING: return NBTString
    elif tag_type == TAG_COMPOUND: return NBTCompound
    elif tag_type == TAG_ARRAY: return NBTArray
    elif tag_type == TAG_FARRAY: return NBTFArray
    elif tag_type == TAG_FCOMPOUND: return NBTFCompound
    elif tag_type == TAG_VAR: return NBTVar
    elif tag_type == TAG_POINT: return NBTPoint
    else: return None

cdef object read_value_by_tag(int tag_type, Reader* r):
    # ---- 把需要的 C 局部变量统一提前声明 ----
    cdef const unsigned char* up = NULL
    cdef const unsigned char* up2 = NULL
    cdef bytes ubytes, ubytes2
    cdef object uobj = None
    cdef object uobj2 = None
    cdef object v = None
    cdef int vtype = 0

    if tag_type == TAG_END:
        return NBTEnd()
    elif tag_type == TAG_BYTE:
        return NBTByte(<int8_t>rd_u8(r))
    elif tag_type == TAG_SHORT:
        return NBTShort(<int16_t>rd_u16be(r))
    elif tag_type == TAG_INT:
        return NBTInt(<int32_t>rd_u32be(r))
    elif tag_type == TAG_LONG:
        return NBTLong(<int64_t>rd_u64be(r))
    elif tag_type == TAG_FLOAT:
        return NBTFloat(rd_f32be(r))
    elif tag_type == TAG_DOUBLE:
        return NBTDouble(rd_f64be(r))
    elif tag_type == TAG_STRING:
        return NBTString(<str>rd_utf8_str(r))
    elif tag_type == TAG_ARRAY:
        return NBTArray.from_reader(r, False)
    elif tag_type == TAG_FARRAY:
        return NBTArray.from_reader(r, True)
    elif tag_type == TAG_COMPOUND:
        return NBTCompound.from_reader(r, False)
    elif tag_type == TAG_FCOMPOUND:
        return NBTCompound.from_reader(r, True)
    elif tag_type == TAG_VAR:
        # UUID(16) + type(1) + payload
        up = rd_ptr(r, 16)
        ubytes = PyBytes_FromStringAndSize(<char*>up, 16)
        uobj = uuid.UUID(bytes=ubytes)
        vtype = rd_u8(r)
        v = read_value_by_tag(vtype, r)
        return NBTVar(<NBTBase>v, str(uobj))
    elif tag_type == TAG_POINT:
        up2 = rd_ptr(r, 16)
        ubytes2 = PyBytes_FromStringAndSize(<char*>up2, 16)
        uobj2 = uuid.UUID(bytes=ubytes2)
        return NBTPoint(str(uobj2))
    else:
        raise ValueError(f"未知的标签类型: {tag_type}")


# =============================================================================
# 基础数据类型
# =============================================================================
cdef class NBTEnd(NBTBase):
    cpdef int get_type(self): return TAG_END

    def to_dict(self): return {"type": TAG_END, "value": None}
    
    def to_bytes(self):
        # 无载荷
        return b""
    
    @classmethod
    def from_bytes(cls, bytes data):
        return cls()
    
    cdef void _write_payload(self, Buf* b):
        # 无内容
        return
    
    def __str__(self):
        return "NBTEnd()"


cdef class NBTByte(NBTBase):
    cdef public int8_t value
    def __init__(self, int8_t value=0): self.value = value
    cpdef int get_type(self): return TAG_BYTE
    def to_dict(self): return {"type": TAG_BYTE, "value": self.value}
    def to_bytes(self):
        cdef Buf b
        buf_init(&b, 1)
        try:
            self._write_payload(&b)
            return PyBytes_FromStringAndSize(<char*>b.data, b.len)
        finally:
            buf_free(&b)
    cdef void _write_payload(self, Buf* b):
        buf_write_u8(b, <uint8_t><int8_t>self.value)
    @classmethod
    def from_bytes(cls, bytes data):
        cdef Reader r
        reader_from_bytes(&r, data)
        return cls(<int8_t>rd_u8(&r))
    def __str__(self): return f"NBTByte({self.value})"


cdef class NBTShort(NBTBase):
    cdef public int16_t value
    def __init__(self, int16_t value=0): self.value = value
    cpdef int get_type(self): return TAG_SHORT
    def to_dict(self): return {"type": TAG_SHORT, "value": self.value}
    def to_bytes(self):
        cdef Buf b
        buf_init(&b, 2)
        try:
            self._write_payload(&b)
            return PyBytes_FromStringAndSize(<char*>b.data, b.len)
        finally:
            buf_free(&b)
    cdef void _write_payload(self, Buf* b):
        buf_write_u16be(b, <uint16_t><int16_t>self.value)
    @classmethod
    def from_bytes(cls, bytes data):
        cdef Reader r
        reader_from_bytes(&r, data)
        return cls(<int16_t>rd_u16be(&r))
    def __str__(self): return f"NBTShort({self.value})"


cdef class NBTInt(NBTBase):
    cdef public int32_t value
    def __init__(self, int32_t value=0): self.value = value
    cpdef int get_type(self): return TAG_INT
    def to_dict(self): return {"type": TAG_INT, "value": self.value}
    def to_bytes(self):
        cdef Buf b
        buf_init(&b, 4)
        try:
            self._write_payload(&b)
            return PyBytes_FromStringAndSize(<char*>b.data, b.len)
        finally:
            buf_free(&b)
    cdef void _write_payload(self, Buf* b):
        buf_write_u32be(b, <uint32_t><int32_t>self.value)
    @classmethod
    def from_bytes(cls, bytes data):
        cdef Reader r
        reader_from_bytes(&r, data)
        return cls(<int32_t>rd_u32be(&r))
    def __str__(self): return f"NBTInt({self.value})"


cdef class NBTLong(NBTBase):
    cdef public int64_t value
    def __init__(self, int64_t value=0): self.value = value
    cpdef int get_type(self): return TAG_LONG
    def to_dict(self): return {"type": TAG_LONG, "value": self.value}
    def to_bytes(self):
        cdef Buf b
        buf_init(&b, 8)
        try:
            self._write_payload(&b)
            return PyBytes_FromStringAndSize(<char*>b.data, b.len)
        finally:
            buf_free(&b)
    cdef void _write_payload(self, Buf* b):
        buf_write_u64be(b, <uint64_t><int64_t>self.value)
    @classmethod
    def from_bytes(cls, bytes data):
        cdef Reader r
        reader_from_bytes(&r, data)
        return cls(<int64_t>rd_u64be(&r))
    def __str__(self): return f"NBTLong({self.value})"


cdef class NBTFloat(NBTBase):
    cdef public float value
    def __init__(self, float value=0.0): self.value = value
    cpdef int get_type(self): return TAG_FLOAT
    def to_dict(self): return {"type": TAG_FLOAT, "value": self.value}
    def to_bytes(self):
        cdef Buf b
        buf_init(&b, 4)
        try:
            self._write_payload(&b)
            return PyBytes_FromStringAndSize(<char*>b.data, b.len)
        finally:
            buf_free(&b)
    cdef void _write_payload(self, Buf* b):
        buf_write_f32be(b, self.value)
    @classmethod
    def from_bytes(cls, bytes data):
        cdef Reader r
        reader_from_bytes(&r, data)
        return cls(rd_f32be(&r))
    def __str__(self): return f"NBTFloat({self.value})"


cdef class NBTDouble(NBTBase):
    cdef public double value
    def __init__(self, double value=0.0): self.value = value
    cpdef int get_type(self): return TAG_DOUBLE
    def to_dict(self): return {"type": TAG_DOUBLE, "value": self.value}
    def to_bytes(self):
        cdef Buf b
        buf_init(&b, 8)
        try:
            self._write_payload(&b)
            return PyBytes_FromStringAndSize(<char*>b.data, b.len)
        finally:
            buf_free(&b)
    cdef void _write_payload(self, Buf* b):
        buf_write_f64be(b, self.value)
    @classmethod
    def from_bytes(cls, bytes data):
        cdef Reader r
        reader_from_bytes(&r, data)
        return cls(rd_f64be(&r))
    def __str__(self): return f"NBTDouble({self.value})"


cdef class NBTString(NBTBase):
    cdef public str value
    def __init__(self, str value=""): self.value = value
    cpdef int get_type(self): return TAG_STRING
    def to_dict(self): return {"type": TAG_STRING, "value": self.value}
    def to_bytes(self):
        cdef Buf b
        buf_init(&b, 16 + (len(self.value) << 1))
        try:
            self._write_payload(&b)
            return PyBytes_FromStringAndSize(<char*>b.data, b.len)
        finally:
            buf_free(&b)
    cdef void _write_payload(self, Buf* b):
        cdef bytes utf8 = self.value.encode("utf-8")
        cdef char* p
        cdef Py_ssize_t n
        if PyBytes_AsStringAndSize(utf8, &p, &n) < 0:
            raise ValueError("字符串编码失败")
        if n > 65535:
            raise ValueError("字符串太长，超过65535字节")
        buf_write_u16be(b, <uint16_t>n)
        buf_write_bytes(b, <const unsigned char*>p, n)
    @classmethod
    def from_bytes(cls, bytes data):
        cdef Reader r
        reader_from_bytes(&r, data)
        return cls(<str>rd_utf8_str(&r))
    def __str__(self): return f'NBTString("{self.value}")'

# =============================================================================
# 复合与数组（直接写入/读取到同一缓冲/游标）
# =============================================================================
cdef class NBTCompound(NBTBase):
    cdef public dict data
    cdef public bint locked
    def __init__(self, dict data=None, bint locked=False):
        self.data = data or {}
        self.locked = locked
    cpdef int get_type(self): return TAG_COMPOUND
    def put(self, str key, NBTBase value):
        if self.locked: raise RuntimeError("复合标签已锁定，无法修改")
        self.data[key] = value
    def get(self, str key): return self.data.get(key)
    def remove(self, str key):
        if self.locked: raise RuntimeError("复合标签已锁定，无法修改")
        return self.data.pop(key, None)
    def keys(self): return list(self.data.keys())
    def values(self): return list(self.data.values())
    def to_dict(self):
        result = {"type": TAG_COMPOUND, "locked": self.locked, "data": {}}
        for k, v in self.data.items():
            if hasattr(v, "to_dict"):
                result["data"][k] = v.to_dict()
            else:
                result["data"][k] = v
        return result
    def to_bytes(self):
        cdef Buf b
        buf_init(&b, 64)
        try:
            self._write_payload(&b)
            return PyBytes_FromStringAndSize(<char*>b.data, b.len)
        finally:
            buf_free(&b)

    cdef void _write_payload(self, Buf* b):
        cdef object key
        cdef object val
        cdef bytes kbytes
        cdef char* kp
        cdef Py_ssize_t kn
        for key, val in self.data.items():
            # 尝试把普通类型转为NBT
            if not isinstance(val, NBTBase):
                if isinstance(val, str):
                    val = NBTString(val)
                elif isinstance(val, int):
                    val = NBTInt(<int>val)
                elif isinstance(val, float):
                    val = NBTFloat(<float>val)
                else:
                    continue
            buf_write_u8(b, <uint8_t>(<NBTBase>val).get_type())
            kbytes = (<str>key).encode("utf-8")
            if PyBytes_AsStringAndSize(kbytes, &kp, &kn) < 0:
                raise ValueError("键名编码失败")
            if kn > 65535:
                raise ValueError("键名过长")
            buf_write_u16be(b, <uint16_t>kn)
            buf_write_bytes(b, <const unsigned char*>kp, kn)
            (<NBTBase>val)._write_payload(b)
        # 结束标记
        buf_write_u8(b, <uint8_t>TAG_END)

    @staticmethod
    cdef NBTCompound from_reader(Reader* r, bint force_locked):
        cdef NBTCompound comp = NBTCompound()
        cdef int t
        cdef str name
        cdef object v
        comp.locked = force_locked
        while True:
            t = rd_u8(r)
            if t == TAG_END:
                break
            name = <str>rd_utf8_str(r)
            v = read_value_by_tag(t, r)
            comp.data[name] = v
        return comp
    
    @classmethod
    def from_bytes(cls, bytes data):
        cdef Reader r
        reader_from_bytes(&r, data)
        return NBTCompound.from_reader(&r, False)
    
    def __str__(self):
        return f"NBTCompound(data={self.data}, locked={self.locked})"


cdef class NBTArray(NBTBase):
    # 通用数据存储（用于混合类型或未指定类型）
    cdef public list data
    cdef public object element_type  # 元素类型类
    cdef public bint locked
    
    # 类型化内存缓冲区（用于指定类型的高效存储）
    cdef public bint use_typed_buffer
    cdef public int typed_element_tag  # 元素的 NBT 标签类型
    cdef public int capacity  # 缓冲区容量
    cdef public int length    # 实际元素数量
    
    # 不同类型的专用缓冲区（C 指针不能是 public）
    cdef int8_t* byte_buffer
    cdef int16_t* short_buffer  
    cdef int32_t* int_buffer
    cdef int64_t* long_buffer
    cdef float* float_buffer
    cdef double* double_buffer
    
    def __init__(self, list data=None, element_type=None, bint locked=False, int initial_capacity=16):
        self.data = []
        self.element_type = element_type
        self.locked = locked
        self.use_typed_buffer = False
        self.capacity = 0
        self.length = 0
        self.typed_element_tag = TAG_END
        
        # 初始化所有缓冲区指针为 NULL
        self.byte_buffer = NULL
        self.short_buffer = NULL
        self.int_buffer = NULL
        self.long_buffer = NULL
        self.float_buffer = NULL
        self.double_buffer = NULL
        
        # 如果指定了元素类型，尝试使用类型化缓冲区
        if element_type is not None:
            self._try_enable_typed_buffer(element_type, initial_capacity)
        
        # 添加初始数据
        if data:
            for item in data:
                self.append(item)
    
    def __dealloc__(self):
        """释放分配的内存缓冲区"""
        self._free_typed_buffers()
    
    cdef void _free_typed_buffers(self):
        """释放所有类型化缓冲区"""
        if self.byte_buffer != NULL:
            free(self.byte_buffer)
            self.byte_buffer = NULL
        if self.short_buffer != NULL:
            free(self.short_buffer)
            self.short_buffer = NULL
        if self.int_buffer != NULL:
            free(self.int_buffer)
            self.int_buffer = NULL
        if self.long_buffer != NULL:
            free(self.long_buffer)
            self.long_buffer = NULL
        if self.float_buffer != NULL:
            free(self.float_buffer)
            self.float_buffer = NULL
        if self.double_buffer != NULL:
            free(self.double_buffer)
            self.double_buffer = NULL
        self.capacity = 0
    
    cdef bint _try_enable_typed_buffer(self, element_type, int initial_capacity):
        """尝试为指定类型启用类型化缓冲区"""
        cdef int element_tag = TAG_END
        
        # 确定元素的 NBT 标签类型
        if element_type == NBTByte:
            element_tag = TAG_BYTE
        elif element_type == NBTShort:
            element_tag = TAG_SHORT
        elif element_type == NBTInt:
            element_tag = TAG_INT
        elif element_type == NBTLong:
            element_tag = TAG_LONG
        elif element_type == NBTFloat:
            element_tag = TAG_FLOAT
        elif element_type == NBTDouble:
            element_tag = TAG_DOUBLE
        else:
            # 不支持的类型，使用通用存储
            return False
        
        # 分配对应类型的缓冲区
        if element_tag == TAG_BYTE:
            self.byte_buffer = <int8_t*>malloc(initial_capacity * sizeof(int8_t))
            if self.byte_buffer == NULL:
                return False
        elif element_tag == TAG_SHORT:
            self.short_buffer = <int16_t*>malloc(initial_capacity * sizeof(int16_t))
            if self.short_buffer == NULL:
                return False
        elif element_tag == TAG_INT:
            self.int_buffer = <int32_t*>malloc(initial_capacity * sizeof(int32_t))
            if self.int_buffer == NULL:
                return False
        elif element_tag == TAG_LONG:
            self.long_buffer = <int64_t*>malloc(initial_capacity * sizeof(int64_t))
            if self.long_buffer == NULL:
                return False
        elif element_tag == TAG_FLOAT:
            self.float_buffer = <float*>malloc(initial_capacity * sizeof(float))
            if self.float_buffer == NULL:
                return False
        elif element_tag == TAG_DOUBLE:
            self.double_buffer = <double*>malloc(initial_capacity * sizeof(double))
            if self.double_buffer == NULL:
                return False
        
        self.use_typed_buffer = True
        self.typed_element_tag = element_tag
        self.capacity = initial_capacity
        self.length = 0
        return True
    
    cdef bint _ensure_capacity(self, int min_capacity):
        """确保缓冲区有足够的容量"""
        if not self.use_typed_buffer:
            return True
        
        if min_capacity <= self.capacity:
            return True
        
        # 扩容策略：至少双倍增长
        cdef int new_capacity = max(min_capacity, self.capacity * 2)
        
        if self.typed_element_tag == TAG_BYTE:
            self.byte_buffer = <int8_t*>realloc(self.byte_buffer, new_capacity * sizeof(int8_t))
            if self.byte_buffer == NULL:
                return False
        elif self.typed_element_tag == TAG_SHORT:
            self.short_buffer = <int16_t*>realloc(self.short_buffer, new_capacity * sizeof(int16_t))
            if self.short_buffer == NULL:
                return False
        elif self.typed_element_tag == TAG_INT:
            self.int_buffer = <int32_t*>realloc(self.int_buffer, new_capacity * sizeof(int32_t))
            if self.int_buffer == NULL:
                return False
        elif self.typed_element_tag == TAG_LONG:
            self.long_buffer = <int64_t*>realloc(self.long_buffer, new_capacity * sizeof(int64_t))
            if self.long_buffer == NULL:
                return False
        elif self.typed_element_tag == TAG_FLOAT:
            self.float_buffer = <float*>realloc(self.float_buffer, new_capacity * sizeof(float))
            if self.float_buffer == NULL:
                return False
        elif self.typed_element_tag == TAG_DOUBLE:
            self.double_buffer = <double*>realloc(self.double_buffer, new_capacity * sizeof(double))
            if self.double_buffer == NULL:
                return False
        
        self.capacity = new_capacity
        return True
    
    cpdef int get_type(self): return TAG_ARRAY
    
    def append(self, object item):
        """添加元素到数组"""
        if self.locked: 
            raise RuntimeError("数组已锁定，无法修改")
        
        # 移除严格的类型检查，让上层的 ion_run 处理类型转换
        # 这样避免重复转换导致的错误
        
        # 如果使用类型化缓冲区
        if self.use_typed_buffer:
            self._append_to_typed_buffer(item)
        else:
            # 使用通用列表存储
            if not isinstance(item, NBTBase):
                item = _to_nbt(item)
            self.data.append(item)
    
    cdef void _append_to_typed_buffer(self, object item):
        """添加元素到类型化缓冲区"""
        if not self._ensure_capacity(self.length + 1):
            raise MemoryError("无法分配内存")
        
        if self.typed_element_tag == TAG_BYTE:
            if isinstance(item, NBTByte):
                self.byte_buffer[self.length] = item.value
            else:
                self.byte_buffer[self.length] = <int8_t>int(item)
        elif self.typed_element_tag == TAG_SHORT:
            if isinstance(item, NBTShort):
                self.short_buffer[self.length] = item.value
            else:
                self.short_buffer[self.length] = <int16_t>int(item)
        elif self.typed_element_tag == TAG_INT:
            if isinstance(item, NBTInt):
                self.int_buffer[self.length] = item.value
            else:
                self.int_buffer[self.length] = <int32_t>int(item)
        elif self.typed_element_tag == TAG_LONG:
            if isinstance(item, NBTLong):
                self.long_buffer[self.length] = item.value
            else:
                self.long_buffer[self.length] = <int64_t>int(item)
        elif self.typed_element_tag == TAG_FLOAT:
            if isinstance(item, NBTFloat):
                self.float_buffer[self.length] = item.value
            else:
                self.float_buffer[self.length] = <float>float(item)
        elif self.typed_element_tag == TAG_DOUBLE:
            if isinstance(item, NBTDouble):
                self.double_buffer[self.length] = item.value
            else:
                self.double_buffer[self.length] = <double>float(item)
        
        self.length += 1
    
    def get(self, int index):
        """获取指定索引的元素"""
        if self.use_typed_buffer:
            return self._get_from_typed_buffer(index)
        else:
            if 0 <= index < len(self.data): 
                return self.data[index]
            raise IndexError("索引超出范围")
    
    cdef object _get_from_typed_buffer(self, int index):
        """从类型化缓冲区获取元素"""
        if index < 0 or index >= self.length:
            raise IndexError("索引超出范围")
        
        if self.typed_element_tag == TAG_BYTE:
            return NBTByte(self.byte_buffer[index])
        elif self.typed_element_tag == TAG_SHORT:
            return NBTShort(self.short_buffer[index])
        elif self.typed_element_tag == TAG_INT:
            return NBTInt(self.int_buffer[index])
        elif self.typed_element_tag == TAG_LONG:
            return NBTLong(self.long_buffer[index])
        elif self.typed_element_tag == TAG_FLOAT:
            return NBTFloat(self.float_buffer[index])
        elif self.typed_element_tag == TAG_DOUBLE:
            return NBTDouble(self.double_buffer[index])
        
        return NBTEnd()  # 不应该到达这里
    
    def set(self, int index, object value):
        """设置指定索引的元素值"""
        if self.locked:
            raise RuntimeError("数组已锁定，无法修改")
        
        if self.use_typed_buffer:
            self._set_in_typed_buffer(index, value)
        else:
            if 0 <= index < len(self.data):
                if not isinstance(value, NBTBase):
                    value = _to_nbt(value)
                self.data[index] = value
            else:
                raise IndexError("索引超出范围")
    
    cdef void _set_in_typed_buffer(self, int index, object value):
        """在类型化缓冲区中设置元素值"""
        if index < 0 or index >= self.length:
            raise IndexError("索引超出范围")
        
        if self.typed_element_tag == TAG_BYTE:
            if isinstance(value, NBTByte):
                self.byte_buffer[index] = value.value
            else:
                self.byte_buffer[index] = <int8_t>int(value)
        elif self.typed_element_tag == TAG_SHORT:
            if isinstance(value, NBTShort):
                self.short_buffer[index] = value.value
            else:
                self.short_buffer[index] = <int16_t>int(value)
        elif self.typed_element_tag == TAG_INT:
            if isinstance(value, NBTInt):
                self.int_buffer[index] = value.value
            else:
                self.int_buffer[index] = <int32_t>int(value)
        elif self.typed_element_tag == TAG_LONG:
            if isinstance(value, NBTLong):
                self.long_buffer[index] = value.value
            else:
                self.long_buffer[index] = <int64_t>int(value)
        elif self.typed_element_tag == TAG_FLOAT:
            if isinstance(value, NBTFloat):
                self.float_buffer[index] = value.value
            else:
                self.float_buffer[index] = <float>float(value)
        elif self.typed_element_tag == TAG_DOUBLE:
            if isinstance(value, NBTDouble):
                self.double_buffer[index] = value.value
            else:
                self.double_buffer[index] = <double>float(value)
    
    def size(self): 
        """返回数组大小"""
        if self.use_typed_buffer:
            return self.length
        else:
            return len(self.data)
    
    def get_memory_usage(self):
        """获取内存使用情况（字节）"""
        if self.use_typed_buffer:
            if self.typed_element_tag == TAG_BYTE:
                return self.capacity * sizeof(int8_t)
            elif self.typed_element_tag == TAG_SHORT:
                return self.capacity * sizeof(int16_t)
            elif self.typed_element_tag == TAG_INT:
                return self.capacity * sizeof(int32_t)
            elif self.typed_element_tag == TAG_LONG:
                return self.capacity * sizeof(int64_t)
            elif self.typed_element_tag == TAG_FLOAT:
                return self.capacity * sizeof(float)
            elif self.typed_element_tag == TAG_DOUBLE:
                return self.capacity * sizeof(double)
        else:
            # 估算列表的内存使用（Python 对象引用）
            return len(self.data) * sizeof(void*)
        return 0
    def to_dict(self):
        res = {"type": self.get_type(), "locked": self.locked,
               "element_type": self.element_type.__name__ if self.element_type else None,
               "use_typed_buffer": self.use_typed_buffer,
               "data": []}
        
        if self.use_typed_buffer:
            # 从类型化缓冲区导出数据
            for i in range(self.length):
                item = self._get_from_typed_buffer(i)
                if hasattr(item, "to_dict"):
                    res["data"].append(item.to_dict())
                else:
                    res["data"].append(item)
        else:
            # 从通用列表导出数据
            for it in self.data:
                if hasattr(it, "to_dict"):
                    res["data"].append(it.to_dict())
                else:
                    res["data"].append(it)
        return res
    
    def to_bytes(self):
        cdef Buf b
        cdef int estimated_size
        if self.use_typed_buffer:
            estimated_size = 16 + (self.length * 8)
        else:
            estimated_size = 16 + (len(self.data) * 8)
        buf_init(&b, estimated_size)
        try:
            self._write_payload(&b)
            return PyBytes_FromStringAndSize(<char*>b.data, b.len)
        finally:
            buf_free(&b)

    cdef void _write_payload(self, Buf* b):
        cdef int data_size
        cdef int et
        cdef object first
        
        if self.use_typed_buffer:
            data_size = self.length
            et = self.typed_element_tag
        else:
            data_size = len(self.data)
            if data_size == 0:
                buf_write_u8(b, <uint8_t>TAG_END)
                buf_write_u32be(b, 0)
                return
            # 从第一个元素推断类型
            first = self.data[0]
            if isinstance(first, NBTBase):
                et = (<NBTBase>first).get_type()
            else:
                if isinstance(first, str): et = TAG_STRING
                elif isinstance(first, int): et = TAG_INT
                elif isinstance(first, float): et = TAG_FLOAT
                else: et = TAG_STRING
        
        buf_write_u8(b, <uint8_t>et)
        buf_write_u32be(b, <uint32_t>data_size)
        
        if self.use_typed_buffer:
            self._write_typed_buffer_payload(b)
        else:
            self._write_generic_payload(b)
    
    cdef void _write_typed_buffer_payload(self, Buf* b):
        """写入类型化缓冲区的数据"""
        cdef int i
        for i in range(self.length):
            if self.typed_element_tag == TAG_BYTE:
                buf_write_u8(b, <uint8_t><int8_t>self.byte_buffer[i])
            elif self.typed_element_tag == TAG_SHORT:
                buf_write_u16be(b, <uint16_t><int16_t>self.short_buffer[i])
            elif self.typed_element_tag == TAG_INT:
                buf_write_u32be(b, <uint32_t><int32_t>self.int_buffer[i])
            elif self.typed_element_tag == TAG_LONG:
                buf_write_u64be(b, <uint64_t><int64_t>self.long_buffer[i])
            elif self.typed_element_tag == TAG_FLOAT:
                buf_write_f32be(b, self.float_buffer[i])
            elif self.typed_element_tag == TAG_DOUBLE:
                buf_write_f64be(b, self.double_buffer[i])
    
    cdef void _write_generic_payload(self, Buf* b):
        """写入通用列表的数据"""
        cdef object it
        for it in self.data:
            if not isinstance(it, NBTBase):
                if isinstance(it, str): it = NBTString(it)
                elif isinstance(it, int): it = NBTInt(<int>it)
                elif isinstance(it, float): it = NBTFloat(<float>it)
                else: it = NBTString(str(it))
            (<NBTBase>it)._write_payload(b)

    @staticmethod
    cdef NBTArray from_reader(Reader* r, bint force_locked):
        cdef int et = rd_u8(r)
        cdef uint32_t n = rd_u32be(r)
        cdef NBTArray arr = NBTArray()
        arr.locked = force_locked
        if n == 0:
            return arr
        cdef object item
        for i in range(n):
            item = read_value_by_tag(et, r)
            arr.data.append(item)
        return arr
    
    @classmethod
    def from_bytes(cls, bytes data):
        cdef Reader r
        reader_from_bytes(&r, data)
        return NBTArray.from_reader(&r, False)
    
    def __str__(self):
        return f"NBTArray(size={len(self.data)}, type={self.element_type}, locked={self.locked})"


cdef class NBTFArray(NBTArray):
    def __init__(self, list data=None, element_type=None):
        super().__init__(data, element_type, True)
    cpdef int get_type(self): return TAG_FARRAY
    def __str__(self): return f"NBTFArray(size={len(self.data)}, type={self.element_type})"


cdef class NBTFCompound(NBTCompound):
    def __init__(self, dict data=None):
        super().__init__(data, True)
    cpdef int get_type(self): return TAG_FCOMPOUND
    def __str__(self): return f"NBTFCompound(data={self.data})"

# =============================================================================
# 特殊类型
# =============================================================================
cdef class NBTVar(NBTBase):
    cdef public str uuid_str
    cdef public object value  # 改为 object 类型以支持任意对象
    def __init__(self, object value=None, str uuid_str=None):
        self.value = value if value is not None else NBTString("")
        self.uuid_str = uuid_str or str(uuid.uuid4())
    cpdef int get_type(self): return TAG_VAR
    def set_value(self, object new_value): self.value = new_value
    def get_value(self): return self.value
    def to_dict(self):
        return {"type": TAG_VAR, "uuid": self.uuid_str,
                "value": self.value.to_dict() if hasattr(self.value, "to_dict") else self.value}
    def to_bytes(self):
        cdef Buf b
        buf_init(&b, 32)
        try:
            self._write_payload(&b)
            return PyBytes_FromStringAndSize(<char*>b.data, b.len)
        finally:
            buf_free(&b)
    cdef void _write_payload(self, Buf* b):
        cdef object uobj = uuid.UUID(self.uuid_str)
        cdef bytes ubytes = (<object>uobj).bytes
        cdef char* p
        cdef Py_ssize_t n
        cdef NBTBase nbt_value
        cdef NBTString nbt_str
        if PyBytes_AsStringAndSize(ubytes, &p, &n) < 0 or n != 16:
            raise ValueError("UUID编码失败")
        buf_write_bytes(b, <const unsigned char*>p, 16)
        # 对于任意对象，我们需要特殊处理序列化
        if isinstance(self.value, NBTBase):
            nbt_value = <NBTBase>self.value
            buf_write_u8(b, <uint8_t>nbt_value.get_type())
            nbt_value._write_payload(b)
        else:
            # 对于非 NBTBase 对象，序列化为字符串表示
            value_str = str(self.value)
            nbt_str = NBTString(value_str)
            buf_write_u8(b, <uint8_t>nbt_str.get_type())
            nbt_str._write_payload(b)
    @classmethod
    def from_bytes(cls, bytes data):
        cdef Reader r
        reader_from_bytes(&r, data)
        cdef const unsigned char* up = rd_ptr(&r, 16)
        cdef bytes ubytes = PyBytes_FromStringAndSize(<char*>up, 16)
        cdef object uo = uuid.UUID(bytes=ubytes)
        cdef int vt = rd_u8(&r)
        cdef object v = read_value_by_tag(vt, &r)
        return cls(<NBTBase>v, str(uo))
    def __str__(self): return f'NBTVar(uuid="{self.uuid_str}", value={self.value})'


cdef class NBTPoint(NBTBase):
    cdef public str uuid_str
    def __init__(self, str uuid_str=None):
        self.uuid_str = uuid_str or str(uuid.uuid4())
    cpdef int get_type(self): return TAG_POINT
    def to_dict(self): return {"type": TAG_POINT, "uuid": self.uuid_str}
    def to_bytes(self):
        cdef Buf b
        buf_init(&b, 16)
        try:
            self._write_payload(&b)
            return PyBytes_FromStringAndSize(<char*>b.data, b.len)
        finally:
            buf_free(&b)
    cdef void _write_payload(self, Buf* b):
        cdef object uobj = uuid.UUID(self.uuid_str)
        cdef bytes ubytes = (<object>uobj).bytes
        cdef char* p
        cdef Py_ssize_t n
        if PyBytes_AsStringAndSize(ubytes, &p, &n) < 0 or n != 16:
            raise ValueError("UUID编码失败")
        buf_write_bytes(b, <const unsigned char*>p, 16)
    @classmethod
    def from_bytes(cls, bytes data):
        cdef Reader r
        reader_from_bytes(&r, data)
        cdef const unsigned char* up = rd_ptr(&r, 16)
        cdef bytes ubytes = PyBytes_FromStringAndSize(<char*>up, 16)
        cdef object uo = uuid.UUID(bytes=ubytes)
        return cls(str(uo))
    def __str__(self): return f'NBTPoint(uuid="{self.uuid_str}")'


# ============================================================================
# ION Express 解释器（基于 NBT 的语言）
# 语法：使用 NBTCompound 作为指令节点，使用 key "$" 标识操作名（中文操作名），其余键为参数
# 示例：
#   序列： {"$": "序", "做": [ expr1, expr2, ... ]}
#   加法： {"$": "加", "与": [ 1, 2, 3 ]}
#   条件： {"$": "若", "判": cond, "是": then_expr_or_block, "非": else_expr_or_block}
#   造变： {"$": "造变", "值": init_expr } -> 返回 NBTPoint 指到一个可变变量
#   赋值： {"$": "记", "位": NBTPoint 或 名字(NBTString), "为": expr}
#   取值： {"$": "取", "位": NBTPoint} 或 {"$": "取", "名": NBTString}
#   函数： {"$": "函", "参": ["x", "y"], "体": block }
#   调用： {"$": "呼", "以": func_expr, "传": [ arg1, arg2 ]}
#   返回： {"$": "出", "值": expr}
# 说明：名字绑定与指针分离；名字绑定只是环境里的别名；指针(NBTPoint)用于原地修改 NBTVar 的值
# ============================================================================

cdef class _ReturnSignal(Exception):
    def __init__(self, value):
        super().__init__("return")
        self.value = value
    
cdef class _BreakSignal(Exception):
    def __init__(self):
        super().__init__("break")

cdef class _ContinueSignal(Exception):
    def __init__(self):
        super().__init__("continue")
    

cdef class IONFunction:
    """函数值：携带形参、体与闭包环境（cdef 优化）"""
    cdef public list param_names
    cdef public object body
    cdef public dict closure_env

    def __cinit__(self, param_names=None, body=None, closure_env=None):
        self.param_names = list(param_names) if param_names is not None else []
        self.body = body
        self.closure_env = dict(closure_env) if closure_env is not None else {}


cdef class _ScopeEnv:
    """作用域环境（cdef 优化，使用 malloc/realloc/free 管理 UUID->NBTVar 顺序存储）
    - name_to_value: 名字到 NBTBase 或 Python 对象（如 IONFunction）的绑定
    - parent: 上级作用域
    - UUID->Var 存储：两个 PyObject** 数组保存 uuid 与 var，线性查找
    """
    cdef public dict name_to_value
    cdef public object parent
    cdef PyObject **_uuid_arr
    cdef PyObject **_var_arr
    cdef Py_ssize_t _var_count
    cdef Py_ssize_t _var_cap

    def __cinit__(self, name_to_value=None, uuid_to_var=None, parent=None):
        self.name_to_value = {} if name_to_value is None else dict(name_to_value)
        self.parent = parent
        self._uuid_arr = <PyObject **>NULL
        self._var_arr = <PyObject **>NULL
        self._var_count = 0
        self._var_cap = 0
        if uuid_to_var is not None:
            for k, v in uuid_to_var.items():
                self.register_var(v)

    def __dealloc__(self):
        cdef Py_ssize_t i
        if self._uuid_arr != <PyObject **>NULL and self._var_arr != <PyObject **>NULL:
            for i in range(self._var_count):
                if self._uuid_arr[i] != <PyObject*>NULL:
                    Py_DECREF(<object> self._uuid_arr[i])
                if self._var_arr[i] != <PyObject*>NULL:
                    Py_DECREF(<object> self._var_arr[i])
            free(self._uuid_arr)
            free(self._var_arr)
        self._uuid_arr = <PyObject **>NULL
        self._var_arr = <PyObject **>NULL
        self._var_count = 0
        self._var_cap = 0

    cdef void _ensure_var_cap(self, Py_ssize_t need):
        cdef Py_ssize_t new_cap
        if self._var_cap >= need:
            return
        new_cap = 8 if self._var_cap == 0 else self._var_cap
        while new_cap < need:
            new_cap = new_cap * 2
        if self._uuid_arr == <PyObject **>NULL:
            self._uuid_arr = <PyObject **>malloc(new_cap * sizeof(PyObject *))
            self._var_arr = <PyObject **>malloc(new_cap * sizeof(PyObject *))
    else:
            self._uuid_arr = <PyObject **>realloc(self._uuid_arr, new_cap * sizeof(PyObject *))
            self._var_arr = <PyObject **>realloc(self._var_arr, new_cap * sizeof(PyObject *))
        if self._uuid_arr == <PyObject **>NULL or self._var_arr == <PyObject **>NULL:
            raise MemoryError("内存分配失败")
        self._var_cap = new_cap

    def child(self):
        return _ScopeEnv(name_to_value={}, uuid_to_var=self.export_uuid_map(), parent=self)

    def get_name(self, name: str):
        cdef _ScopeEnv cur = self
        while cur is not None:
            if name in cur.name_to_value:
                return cur.name_to_value[name]
            cur = <_ScopeEnv>cur.parent if cur.parent is not None else None
        return None

    def set_name(self, name: str, value):
        cdef _ScopeEnv cur = self
        while cur is not None:
            if name in cur.name_to_value:
                cur.name_to_value[name] = value
                return
            cur = <_ScopeEnv>cur.parent if cur.parent is not None else None
        self.name_to_value[name] = value

    def register_var(self, NBTVar var):
        cdef Py_ssize_t i
        cdef object uuid_str = var.uuid_str
        for i in range(self._var_count):
            if self._uuid_arr[i] is not <PyObject*>NULL and (<object>self._uuid_arr[i]) == uuid_str:
                Py_INCREF(<object> var)
                if self._var_arr[i] is not <PyObject*>NULL:
                    Py_DECREF(<object> self._var_arr[i])
                self._var_arr[i] = <PyObject*>var
                return
        self._ensure_var_cap(self._var_count + 1)
        Py_INCREF(<object> uuid_str)
        Py_INCREF(<object> var)
        self._uuid_arr[self._var_count] = <PyObject*>uuid_str
        self._var_arr[self._var_count] = <PyObject*>var
        self._var_count += 1

    def get_var_by_uuid(self, uuid_str: str):
        cdef Py_ssize_t i
        for i in range(self._var_count):
            if (<object>self._uuid_arr[i]) == uuid_str:
                return <object>self._var_arr[i]
        return None

    def export_uuid_map(self):
        cdef dict d = {}
        cdef Py_ssize_t i
        for i in range(self._var_count):
            d[<object>self._uuid_arr[i]] = <object>self._var_arr[i]
        return d


# ======== 类与接口运行时（cdef） ========

cdef class IONInterface:
    cdef public object name
    cdef public set required
    def __cinit__(self, required=None, name=None):
        self.required = set(required) if required is not None else set()
        self.name = name

cdef class IONClass:
    cdef public dict field_defaults
    cdef public dict methods
    cdef public list interfaces
    cdef public str init_method_name
    cdef public str getitem_method_name
    cdef public str setitem_method_name
    cdef public str convert_method_name
    
    def __cinit__(self, field_defaults=None, methods=None, interfaces=None, 
                  init_method=None, getitem_method=None, setitem_method=None, convert_method=None):
        self.field_defaults = dict(field_defaults) if field_defaults is not None else {}
        self.methods = dict(methods) if methods is not None else {}
        self.interfaces = list(interfaces) if interfaces is not None else []
        self.init_method_name = init_method
        self.getitem_method_name = getitem_method
        self.setitem_method_name = setitem_method
        self.convert_method_name = convert_method
    
    cpdef object get_method(self, str name):
        return self.methods.get(name)
    
    cpdef object get_init_method(self):
        if self.init_method_name and self.init_method_name in self.methods:
            return self.methods[self.init_method_name]
        return None
    
    cpdef object get_getitem_method(self):
        if self.getitem_method_name and self.getitem_method_name in self.methods:
            return self.methods[self.getitem_method_name]
        return None
    
    cpdef object get_setitem_method(self):
        if self.setitem_method_name and self.setitem_method_name in self.methods:
            return self.methods[self.setitem_method_name]
        return None
    
    cpdef bint implements(self, IONInterface itf):
        cdef object need
        for need in itf.required:
            if need not in self.methods:
                return False
        return True

cdef class IONInstance:
    cdef public IONClass cls
    cdef public dict fields
    def __cinit__(self, IONClass cls):
        self.cls = cls
        self.fields = dict(cls.field_defaults)
    cpdef object get_field(self, str name):
        return self.fields.get(name, NBTEnd())
    cpdef void set_field(self, str name, object value):
        self.fields[name] = value

cdef object _call_bound_method(IONInstance self_obj, IONFunction m, list args, _ScopeEnv caller_env):
    cdef _ScopeEnv call_env = _ScopeEnv(name_to_value=m.closure_env, uuid_to_var=caller_env.export_uuid_map(), parent=None).child()
    cdef list final_params = ["自"] + m.param_names
    cdef list final_args = [self_obj] + [ _to_nbt(a) for a in args ]
    cdef Py_ssize_t i
    for i in range(len(final_params)):
        call_env.set_name(final_params[i], final_args[i] if i < len(final_args) else NBTEnd())
    try:
        ret = _eval_block(m.body, call_env)
    except _ReturnSignal as rs:
        ret = rs.value
    return _to_nbt(_resolve_value(ret, call_env))

# =============== NBT <-> Python 转换与工具 ===================

def _to_nbt(obj):
    """将 Python 基础对象递归转换为 NBT 对象（宽松转换）"""
    if isinstance(obj, NBTBase):
        return obj
    if obj is None:
        return NBTEnd()
    if isinstance(obj, bool):
        return NBTByte(1 if obj else 0)
    if isinstance(obj, int):
        # 32位范围内用 NBTInt，否则用 NBTLong
        if obj < -2147483648 or obj > 2147483647:
            return NBTLong(obj)
        return NBTInt(obj)
    if isinstance(obj, float):
        return NBTDouble(obj)
    if isinstance(obj, str):
        return NBTString(obj)
    if isinstance(obj, list):
        items = [_to_nbt(it) for it in obj]
        return NBTArray(items)
    if isinstance(obj, dict):
        data = {}
        for k, v in obj.items():
            data[str(k)] = _to_nbt(v)
        return NBTCompound(data)
    # 特殊处理 IONInstance - 将其序列化为 NBTCompound
    if isinstance(obj, IONInstance):
        data = {}
        # 存储类信息和字段
        data["__ion_instance__"] = NBTString("true")
        data["__class_id__"] = NBTString(str(id(obj.cls)))
        for field_name, field_value in obj.fields.items():
            data[field_name] = _to_nbt(field_value)
        return NBTCompound(data)
    # 特殊处理 IONClass - 将其标记为类对象
    if isinstance(obj, IONClass):
        data = {
            "__ion_class__": NBTString("true"),
            "__class_id__": NBTString(str(id(obj))),
            "__field_count__": NBTInt(len(obj.field_defaults)),
            "__method_count__": NBTInt(len(obj.methods))
        }
        return NBTCompound(data)
    # 特殊处理 IONFunction - 将其标记为函数对象
    if isinstance(obj, IONFunction):
        data = {
            "__ion_function__": NBTString("true"),
            "__param_count__": NBTInt(len(obj.param_names))
        }
        return NBTCompound(data)
    # 其他类型保持原样，但包装为 NBTString 以确保类型兼容
    return NBTString(str(obj))


def _number_of(nbt: NBTBase):
    """取 NBT 数字的 Python 值；若非数字则抛错"""
    if isinstance(nbt, (NBTByte, NBTShort, NBTInt, NBTLong)):
        return int(nbt.value)
    if isinstance(nbt, (NBTFloat, NBTDouble)):
        return float(nbt.value)
    raise TypeError(f"需要数字类型，收到: {type(nbt).__name__}")


def _nbt_number(py_num):
    if isinstance(py_num, float):
        return NBTDouble(py_num)
    # 默认按 int 处理
    if py_num < -2147483648 or py_num > 2147483647:
        return NBTLong(int(py_num))
    return NBTInt(int(py_num))


def _is_truthy(val: object) -> bool:
    # 支持 NBT 类型与 Python 基础类型
    if isinstance(val, NBTVar):
        return _is_truthy(val.get_value())
    if isinstance(val, NBTPoint):
        # 指针本身视为 True；取值逻辑在调用点处理
        return True
    if isinstance(val, (NBTByte, NBTShort, NBTInt, NBTLong)):
        return int(val.value) != 0
    if isinstance(val, (NBTFloat, NBTDouble)):
        return float(val.value) != 0.0
    if isinstance(val, NBTString):
        return len(val.value) > 0
    if isinstance(val, NBTArray):
        return len(val.data) > 0
    if isinstance(val, NBTCompound):
        return len(val.data) > 0
    if isinstance(val, NBTEnd):
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val != 0
    if isinstance(val, str):
        return len(val) > 0
    return True


def _bool_to_nbt(flag: bool):
    return NBTByte(1 if flag else 0)


# =============== 快速操作分派器（基于操作码映射） ===================

# 操作码常量（用于快速分派）
cdef enum OpCode:
    OP_UNKNOWN = 0
    OP_SEQUENCE = 1      # 序
    OP_ADD = 2           # 加
    OP_SUB = 3           # 减
    OP_MUL = 4           # 乘
    OP_DIV = 5           # 除
    OP_MOD = 6           # 余
    OP_EQ = 7            # 等
    OP_GT = 8            # 大于
    OP_LT = 9            # 小于
    OP_AND = 10          # 且
    OP_OR = 11           # 或
    OP_NOT = 12          # 非
    OP_IF = 13           # 若
    OP_CREATE_VAR = 14   # 造变
    OP_ASSIGN = 15       # 记
    OP_GET = 16          # 取
    OP_DEBUG = 17        # 看
    OP_FUNCTION = 18     # 函
    OP_CALL = 19         # 呼
    OP_RETURN = 20       # 出
    OP_FOR_LOOP = 21     # 循环
    OP_WHILE_LOOP = 22   # 当
    OP_FOREACH = 23      # 遍历
    OP_BREAK = 24        # 跳出
    OP_CONTINUE = 25     # 继续
    OP_CONCAT = 26       # 接
    OP_INDEX_GET = 27    # 标
    OP_INDEX_SET = 28    # 记标
    OP_CREATE_ATTR = 29  # 创属
    OP_CONVERT = 30      # 转
    OP_FILE = 31         # 文件
    OP_SEARCH = 32       # 查
    OP_IN = 33           # 在
    OP_CLASS = 34        # 类
    OP_CREATE_INSTANCE = 35  # 造
    OP_GET_ATTR = 36     # 取属
    OP_SET_ATTR = 37     # 记属
    OP_CALL_ATTR = 38    # 呼属
    OP_IS_CLASS = 39     # 是类
    OP_VERIFY_INTERFACE = 40  # 验介
    OP_INTERFACE = 41    # 接口
    OP_SAVE = 42         # 存
    OP_LOAD = 43         # 载
    OP_EMPTY = 44        # 空
    OP_NOT_EMPTY = 45    # 非空
    OP_TYPE = 46         # 类型
    OP_SAFE_GET = 47     # 安全取
    OP_SAFE_SET = 48     # 安全记
    OP_GET_VAR = 49      # 取
    OP_GET_VAR_REF = 50  # 取位
    OP_ASSIGN_VAR = 51   # 记
    OP_DEBUG_PRINT = 52  # 看
    OP_FUNCTION_DEF = 53 # 函
    OP_FUNCTION_CALL = 54 # 呼
    OP_RETURN_VAL = 55   # 出
    OP_CONCAT_DATA = 56  # 接
    OP_CHANGE = 57       # 改

# 全局函数映射表（使用 B+树 进行快速查找）
cdef object _function_map = None

def _init_function_map():
    """初始化函数映射表"""
    global _function_map
    if _function_map is not None:
        return _function_map
    
    # 根据是否有 B+树选择不同的数据结构
    if HAS_BPTREE:
        _function_map = BPTree(32)
    else:
        _function_map = {}
    
    # 基础操作 - 根据数据结构类型选择插入方式
    if HAS_BPTREE:
        # B+树使用 insert 方法
        _function_map.insert("序", OP_SEQUENCE_func)
        _function_map.insert("加", OP_ADD_OP_SUB_OP_MUL_OP_DIV_OP_MOD_func)
        _function_map.insert("减", OP_ADD_OP_SUB_OP_MUL_OP_DIV_OP_MOD_func)
        _function_map.insert("乘", OP_ADD_OP_SUB_OP_MUL_OP_DIV_OP_MOD_func)
        _function_map.insert("除", OP_ADD_OP_SUB_OP_MUL_OP_DIV_OP_MOD_func)
        _function_map.insert("余", OP_ADD_OP_SUB_OP_MUL_OP_DIV_OP_MOD_func)
        _function_map.insert("等", OP_EQ_OP_GT_OP_LT_func)
        _function_map.insert("大于", OP_EQ_OP_GT_OP_LT_func)
        _function_map.insert("小于", OP_EQ_OP_GT_OP_LT_func)
        _function_map.insert("且", OP_AND_OP_OR_OP_NOT_func)
        _function_map.insert("或", OP_AND_OP_OR_OP_NOT_func)
        _function_map.insert("非", OP_AND_OP_OR_OP_NOT_func)
        _function_map.insert("若", OP_IF_func)
        _function_map.insert("造变", 造变_func)
        _function_map.insert("取", 取_func)
        _function_map.insert("看", 看_func)
        _function_map.insert("函", 函_func)
        _function_map.insert("呼", 呼_func)
        _function_map.insert("出", 出_func)
        _function_map.insert("循环", 循环_func)
        _function_map.insert("当", 当_func)
        _function_map.insert("遍历", 遍历_func)
        _function_map.insert("跳出", 跳出_func)
        _function_map.insert("继续", 继续_func)
        _function_map.insert("接", 接_func)
        _function_map.insert("标", OP_INDEX_GET_func)
        _function_map.insert("记标", OP_INDEX_SET_func)
        _function_map.insert("创属", OP_CREATE_ATTR_func)
        _function_map.insert("转", OP_CONVERT_func)
        _function_map.insert("文件", OP_FILE_func)
        _function_map.insert("查", OP_SEARCH_func)
        _function_map.insert("在", OP_IN_func)
        _function_map.insert("类", OP_CLASS_func)
        _function_map.insert("造", OP_CREATE_INSTANCE_func)
        _function_map.insert("取属", OP_GET_ATTR_func)
        _function_map.insert("记属", OP_SET_ATTR_func)
        _function_map.insert("呼属", OP_CALL_ATTR_func)
        _function_map.insert("是类", 是类_func)
        _function_map.insert("验介", 验介_func)
        _function_map.insert("接口", 接口_func)
        _function_map.insert("存", 存_func)
        _function_map.insert("载", 载_func)
        _function_map.insert("空", 空_func)
        _function_map.insert("非空", 非空_func)
        _function_map.insert("类型", 类型_func)
        _function_map.insert("安全取", 安全取_func)
        _function_map.insert("安全记", 安全记_func)
        _function_map.insert("取位", OP_GET_VAR_REF)
        _function_map.insert("记", 记_func)
        _function_map.insert("改", 改_func)
    else:
        # 字典使用标准赋值
        _function_map["序"] = OP_SEQUENCE_func
        _function_map["加"] = OP_ADD_OP_SUB_OP_MUL_OP_DIV_OP_MOD_func
        _function_map["减"] = OP_ADD_OP_SUB_OP_MUL_OP_DIV_OP_MOD_func
        _function_map["乘"] = OP_ADD_OP_SUB_OP_MUL_OP_DIV_OP_MOD_func
        _function_map["除"] = OP_ADD_OP_SUB_OP_MUL_OP_DIV_OP_MOD_func
        _function_map["余"] = OP_ADD_OP_SUB_OP_MUL_OP_DIV_OP_MOD_func
        _function_map["等"] = OP_EQ_OP_GT_OP_LT_func
        _function_map["大于"] = OP_EQ_OP_GT_OP_LT_func
        _function_map["小于"] = OP_EQ_OP_GT_OP_LT_func
        _function_map["且"] = OP_AND_OP_OR_OP_NOT_func
        _function_map["或"] = OP_AND_OP_OR_OP_NOT_func
        _function_map["非"] = OP_AND_OP_OR_OP_NOT_func
        _function_map["若"] = OP_IF_func
        _function_map["造变"] = 造变_func
        _function_map["取"] = 取_func
        _function_map["看"] = 看_func
        _function_map["函"] = 函_func
        _function_map["呼"] = 呼_func
        _function_map["出"] = 出_func
        _function_map["循环"] = 循环_func
        _function_map["当"] = 当_func
        _function_map["遍历"] = 遍历_func
        _function_map["跳出"] = 跳出_func
        _function_map["继续"] = 继续_func
        _function_map["接"] = 接_func
        _function_map["标"] = OP_INDEX_GET_func
        _function_map["记标"] = OP_INDEX_SET_func
        _function_map["创属"] = OP_CREATE_ATTR_func
        _function_map["转"] = OP_CONVERT_func
        _function_map["文件"] = OP_FILE_func
        _function_map["查"] = OP_SEARCH_func
        _function_map["在"] = OP_IN_func
        _function_map["类"] = OP_CLASS_func
        _function_map["造"] = OP_CREATE_INSTANCE_func
        _function_map["取属"] = OP_GET_ATTR_func
        _function_map["记属"] = OP_SET_ATTR_func
        _function_map["呼属"] = OP_CALL_ATTR_func
        _function_map["是类"] = 是类_func
        _function_map["验介"] = 验介_func
        _function_map["接口"] = 接口_func
        _function_map["存"] = 存_func
        _function_map["载"] = 载_func
        _function_map["空"] = 空_func
        _function_map["非空"] = 非空_func
        _function_map["类型"] = 类型_func
        _function_map["安全取"] = 安全取_func
        _function_map["安全记"] = 安全记_func
        _function_map["记"] = 记_func
        _function_map["改"] = 改_func
    
    return _function_map

cdef inline int _get_op_code(str op):
    """获取操作的操作码"""
    # 简单的操作码映射，用于需要操作码的函数
    if op == "加":
        return OP_ADD
    elif op == "减":
        return OP_SUB
    elif op == "乘":
        return OP_MUL
    elif op == "除":
        return OP_DIV
    elif op == "余":
        return OP_MOD
    elif op == "等":
        return OP_EQ
    elif op == "大于":
        return OP_GT
    elif op == "小于":
        return OP_LT
    elif op == "且":
        return OP_AND
    elif op == "或":
        return OP_OR
    elif op == "非":
        return OP_NOT
    else:
        return OP_UNKNOWN

cdef inline object _get_function(str op):
    """获取操作对应的函数"""
    _init_function_map()
    if HAS_BPTREE:
        return _function_map.get(op, None)
    else:
        return _function_map.get(op, None)


# =============== 求值器 ===================

def _eval_block(block, env: _ScopeEnv):
    # block 可是单一表达式，或 NBTArray 表达式序列
    if isinstance(block, NBTArray):
        last = NBTEnd()
        for item in block.data:
            last = _eval_expr(item, env)
        return last
    return _eval_expr(block, env)


def _deref_pointer(ptr: NBTPoint, env: _ScopeEnv):
    var = env.get_var_by_uuid(ptr.uuid_str)
    if var is None:
        raise RuntimeError(f"未找到指针所指变量: {ptr.uuid_str}")
    return var.get_value()


def _resolve_value(obj, env: _ScopeEnv):
    # 取值：对指针与变量做解引用，其余原样
    if isinstance(obj, NBTPoint):
        try:
            return _deref_pointer(obj, env)
        except RuntimeError:
            # 指针无效时返回 NBTEnd 而不是抛错
            return NBTEnd()
    if isinstance(obj, NBTVar):
        # 确保注册
        env.register_var(obj)
        return obj.get_value()
    return obj


def _single_index_access(array_val, index_val, env: _ScopeEnv):
    """单个索引访问的辅助函数，支持数字索引和字符串键"""
    
    # 数组访问
    if isinstance(array_val, NBTArray):
        # 检查索引类型
        if isinstance(index_val, (NBTByte, NBTShort, NBTInt, NBTLong)):
            access_index = int(index_val.value)
            arr_len = len(array_val.data)
            # 支持负数索引
            if access_index < 0:
                access_index += arr_len
            # 边界检查
            if access_index < 0 or access_index >= arr_len:
                raise IndexError(f"数组索引超出范围: {access_index}, 数组长度: {arr_len}")
            return array_val.data[access_index]
        else:
            raise TypeError("数组索引必须是整数")
    
    # 字符串访问（按字符索引）
    elif isinstance(array_val, NBTString):
        if isinstance(index_val, (NBTByte, NBTShort, NBTInt, NBTLong)):
            access_index = int(index_val.value)
            str_len = len(array_val.value)
            # 支持负数索引
            if access_index < 0:
                access_index += str_len
            # 边界检查
            if access_index < 0 or access_index >= str_len:
                raise IndexError(f"字符串索引超出范围: {access_index}, 字符串长度: {str_len}")
            return NBTString(array_val.value[access_index])
        else:
            raise TypeError("字符串索引必须是整数")
    
    # 复合标签访问（支持数字键和字符串键）
    elif isinstance(array_val, NBTCompound):
        if isinstance(index_val, (NBTByte, NBTShort, NBTInt, NBTLong)):
            # 数字索引转为字符串键
            key_str = str(int(index_val.value))
        elif isinstance(index_val, NBTString):
            # 字符串键直接使用
            key_str = index_val.value
        else:
            # 其他类型转为字符串
            key_str = str(index_val)
        
        if key_str in array_val.data:
            return array_val.data[key_str]
        else:
            raise KeyError(f"复合标签中不存在键: {key_str}")
    
    # 类实例访问（通过标函方法）
    elif isinstance(array_val, IONInstance):
        getitem_method = array_val.cls.get_getitem_method()
        if getitem_method is not None:
            # 调用标函方法
            args = [_to_nbt(index_val)]
            return _call_bound_method(array_val, getitem_method, args, env)
        else:
            raise TypeError(f"类实例没有定义标函方法")
    
    # 其他类型不支持索引访问
    else:
        raise TypeError(f"类型 {type(array_val).__name__} 不支持索引访问")


def _single_index_modify(array_val, index_val, new_val, env: _ScopeEnv):
    """单个索引修改的辅助函数，支持数字索引和字符串键"""
    
    # 数组修改
    if isinstance(array_val, NBTArray):
        if array_val.locked:
            raise RuntimeError("数组已锁定，无法修改")
        
        if isinstance(index_val, (NBTByte, NBTShort, NBTInt, NBTLong)):
            access_index = int(index_val.value)
            arr_len = len(array_val.data)
            # 支持负数索引
            if access_index < 0:
                access_index += arr_len
            # 边界检查
            if access_index < 0 or access_index >= arr_len:
                raise IndexError(f"数组索引超出范围: {access_index}, 数组长度: {arr_len}")
            
            # 修改数组元素
            array_val.data[access_index] = _to_nbt(new_val)
            return array_val.data[access_index]
        else:
            raise TypeError("数组索引必须是整数")
    
    # 复合标签修改（支持数字键和字符串键）
    elif isinstance(array_val, NBTCompound):
        if array_val.locked:
            raise RuntimeError("复合标签已锁定，无法修改")
        
        if isinstance(index_val, (NBTByte, NBTShort, NBTInt, NBTLong)):
            # 数字索引转为字符串键
            key_str = str(int(index_val.value))
        elif isinstance(index_val, NBTString):
            # 字符串键直接使用
            key_str = index_val.value
        else:
            # 其他类型转为字符串
            key_str = str(index_val)
        
        array_val.put(key_str, _to_nbt(new_val))
        return array_val.data[key_str]
    
    # 字符串不支持修改（字符串是不可变的）
    elif isinstance(array_val, NBTString):
        raise TypeError("字符串是不可变的，不支持修改")
    
    # 类实例修改（通过记函方法）
    elif isinstance(array_val, IONInstance):
        setitem_method = array_val.cls.get_setitem_method()
        if setitem_method is not None:
            # 调用记函方法
            args = [_to_nbt(index_val), _to_nbt(new_val)]
            return _call_bound_method(array_val, setitem_method, args, env)
            else:
            raise TypeError(f"类实例没有定义记函方法")
    
    # 其他类型不支持索引修改
    else:
        raise TypeError(f"类型 {type(array_val).__name__} 不支持索引修改")


def _nested_index_modify(array_val, index_list, new_val, env: _ScopeEnv, depth: int):
    """嵌套索引修改的递归函数"""
    if depth >= len(index_list):
        return array_val
    
    current_index = _resolve_value(index_list[depth], env)
    
    if depth == len(index_list) - 1:
        # 最后一层，直接修改
        return _single_index_modify(array_val, current_index, new_val, env)
                else:
        # 中间层，需要继续递归
        next_val = _single_index_access(array_val, current_index, env)
        return _nested_index_modify(next_val, index_list, new_val, env, depth + 1)

cdef OP_SEQUENCE_func(NBTCompound expr, _ScopeEnv env):
    cdef object raw, item, last
    
    raw = expr.data.get("做")
    raw = raw if isinstance(raw, NBTArray) else NBTArray([])
    last = NBTEnd()
    for item in raw.data:
        last = _eval_expr(item, env)
    return last

cdef OP_INDEX_GET_func(NBTCompound expr, _ScopeEnv env):
    # 数组索引访问: {"$": "标", "于": array_expr, "索": index_expr}
    cdef object array_expr, index_expr, array_val, index_val
    cdef object current_val, idx_item, idx_resolved
    
    array_expr = expr.data.get("于")
    index_expr = expr.data.get("索")
    
    if array_expr is None:
        raise TypeError("标 需要 '于' 参数指定数组")
    if index_expr is None:
        raise TypeError("标 需要 '索' 参数指定索引")
    
    # 求值数组和索引
    array_val = _resolve_value(_eval_expr(array_expr, env), env)
    index_val = _resolve_value(_eval_expr(index_expr, env), env)
    
    # 支持嵌套索引访问：如果索引是数组，则递归访问
    if isinstance(index_val, NBTArray):
        current_val = array_val
        for idx_item in index_val.data:
            idx_resolved = _resolve_value(idx_item, env)
            current_val = _single_index_access(current_val, idx_resolved, env)
        return current_val
    else:
        # 单个索引访问
        return _single_index_access(array_val, index_val, env)

cdef OP_INDEX_SET_func(NBTCompound expr, _ScopeEnv env):
    # 数组索引修改: {"$": "记标", "于": array_expr, "索": index_expr, "为": value_expr}
    cdef object array_expr, index_expr, value_expr
    cdef object array_val, index_val, new_val
    cdef object resolved_array, final_index
    
    array_expr = expr.data.get("于")
    index_expr = expr.data.get("索")
    value_expr = expr.data.get("为")
    
    if array_expr is None:
        raise TypeError("记标 需要 '于' 参数指定数组")
    if index_expr is None:
        raise TypeError("记标 需要 '索' 参数指定索引")
    if value_expr is None:
        raise TypeError("记标 需要 '为' 参数指定新值")
    
    # 求值数组、索引和新值
    array_val = _eval_expr(array_expr, env)  # 不解引用，保持原始引用
    index_val = _resolve_value(_eval_expr(index_expr, env), env)
    new_val = _resolve_value(_eval_expr(value_expr, env), env)
    
    # 支持嵌套索引修改：如果索引是数组，则递归访问到最后一层进行修改
    if isinstance(index_val, NBTArray):
        if len(index_val.data) == 0:
            raise TypeError("索引数组不能为空")
        elif len(index_val.data) == 1:
            # 单层索引，直接修改
            resolved_array = _resolve_value(array_val, env)
            final_index = _resolve_value(index_val.data[0], env)
            return _single_index_modify(resolved_array, final_index, new_val, env)
        else:
            # 多层索引，需要特殊处理以保持引用
            resolved_array = _resolve_value(array_val, env)
            
            # 使用递归的方式进行嵌套修改
            return _nested_index_modify(resolved_array, index_val.data, new_val, env, 0)
    else:
        # 单个索引修改
        resolved_array = _resolve_value(array_val, env)
        return _single_index_modify(resolved_array, index_val, new_val, env)

cdef OP_CREATE_ATTR_func(NBTCompound expr, _ScopeEnv env):
    # 动态创建属性: {"$": "创属", "于": instance_expr, "名": name_expr, "为": value_expr}
    instance_expr = expr.data.get("于")
    name_expr = expr.data.get("名")
    value_expr = expr.data.get("为")
    
    if instance_expr is None:
        raise TypeError("创属 需要 '于' 参数指定实例")
    if name_expr is None:
        raise TypeError("创属 需要 '名' 参数指定属性名")
    if value_expr is None:
        raise TypeError("创属 需要 '为' 参数指定属性值")
    
    # 求值实例、属性名和属性值
    instance_val = _eval_expr(instance_expr, env)
    name_val = _resolve_value(_eval_expr(name_expr, env), env)
    value_val = _resolve_value(_eval_expr(value_expr, env), env)
    
    # 确保实例是 IONInstance
    if not isinstance(instance_val, IONInstance):
        raise TypeError("创属 只能用于类实例")
    
    # 确保属性名是字符串
    if isinstance(name_val, NBTString):
        attr_name = name_val.value
    elif isinstance(name_val, str):
        attr_name = name_val
    else:
        raise TypeError("创属 的属性名必须是字符串")
    
    # 动态创建属性（即使在类的域中未声明）
    instance_val.fields[attr_name] = _to_nbt(value_val)
    
    return instance_val.fields[attr_name]

cdef OP_CONVERT_func(NBTCompound expr, _ScopeEnv env):
    # 类型转换: {"$": "转", "类": target_type, "值": value_expr}
    cdef object target_type_expr, value_expr, target_type, source_value
    cdef str type_name
    
    target_type_expr = expr.data.get("类")
    value_expr = expr.data.get("值")
    
    if target_type_expr is None:
        raise TypeError("转 需要 '类' 参数指定目标类型")
    if value_expr is None:
        raise TypeError("转 需要 '值' 参数指定要转换的值")
    
    # 求值目标类型和要转换的值
    target_type = _resolve_value(_eval_expr(target_type_expr, env), env)
    source_value = _resolve_value(_eval_expr(value_expr, env), env)
    
    # 处理内置类型转换
    if isinstance(target_type, NBTString):
        type_name = target_type.value
        if type_name == "Int":
            if isinstance(source_value, (NBTByte, NBTShort, NBTInt, NBTLong)):
                return NBTInt(int(source_value.value))
            elif isinstance(source_value, (NBTFloat, NBTDouble)):
                return NBTInt(int(source_value.value))
            elif isinstance(source_value, NBTString):
                try:
                    return NBTInt(int(source_value.value))
                except ValueError:
                    raise TypeError(f"无法将字符串 '{source_value.value}' 转换为 Int")
            else:
                raise TypeError(f"无法将 {type(source_value)} 转换为 Int")
        elif type_name == "Float":
            if isinstance(source_value, (NBTByte, NBTShort, NBTInt, NBTLong, NBTFloat, NBTDouble)):
                return NBTFloat(float(source_value.value))
            elif isinstance(source_value, NBTString):
                try:
                    return NBTFloat(float(source_value.value))
                except ValueError:
                    raise TypeError(f"无法将字符串 '{source_value.value}' 转换为 Float")
            else:
                raise TypeError(f"无法将 {type(source_value)} 转换为 Float")
        elif type_name == "String":
            # 任何类型都可以转换为字符串
            return NBTString(str(source_value))
        else:
            raise TypeError(f"未知的内置类型: {type_name}")
    
    # 处理自定义类的转函
    elif isinstance(target_type, IONClass):
        if target_type.convert_method_name is not None:
            # 获取转函方法
            convert_method = target_type.get_method(target_type.convert_method_name)
            if convert_method is None:
                raise TypeError(f"类没有定义转函方法: {target_type.convert_method_name}")
            
            # 调用转函方法 - 使用与"呼"相同的调用机制
            args = [source_value]
            # 建立调用作用域
            call_env = _ScopeEnv(name_to_value=convert_method.closure_env, uuid_to_var=env.export_uuid_map(), parent=None).child()
            # 形参与值绑定
            for i, name in enumerate(convert_method.param_names):
                bound = _to_nbt(args[i]) if i < len(args) else NBTEnd()
                call_env.set_name(name, bound)
            try:
                ret = _eval_block(convert_method.body, call_env)
            except _ReturnSignal as rs:
                ret = rs.value
            return ret
            else:
            raise TypeError("类没有定义转函方法")
    else:
        raise TypeError("转 的 '类' 参数必须是字符串（内置类型）或 IONClass（自定义类）")

cdef OP_FILE_func(NBTCompound expr, _ScopeEnv env):
    # 文件操作: {"$": "文件", "模": mode, "位": position, "文": content, "路": filepath}
    cdef object mode_expr, position_expr, content_expr, filepath_expr
    cdef object mode_val, filepath_val, position_val, content_val
    cdef str mode_str, filepath_str, content_str, file_content
    cdef list lines, position_list
    cdef int start_line, end_line, line_num
    
    mode_expr = expr.data.get("模")
    position_expr = expr.data.get("位")
    content_expr = expr.data.get("文")
    filepath_expr = expr.data.get("路")
    
    if mode_expr is None:
        raise TypeError("文件 需要 '模' 参数指定模式")
    if filepath_expr is None:
        raise TypeError("文件 需要 '路' 参数指定文件路径")
    
    # 求值模式和文件路径
    mode_val = _resolve_value(_eval_expr(mode_expr, env), env)
    filepath_val = _resolve_value(_eval_expr(filepath_expr, env), env)
    
    if not isinstance(mode_val, NBTString):
        raise TypeError("文件 的 '模' 参数必须是字符串")
    if not isinstance(filepath_val, NBTString):
        raise TypeError("文件 的 '路' 参数必须是字符串")
    
    mode = mode_val.value
    filepath = filepath_val.value
    
    if mode == "读":
        # 读取模式
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                if position_expr is not None:
                    position_val = _resolve_value(_eval_expr(position_expr, env), env)
                    if isinstance(position_val, NBTArray) and len(position_val.data) == 2:
                        # 指定行范围读取
                        start_line = int(_resolve_value(position_val.data[0], env).value) - 1  # 转为0索引
                        end_line = int(_resolve_value(position_val.data[1], env).value) - 1
                        lines = f.readlines()
                        if start_line < 0 or end_line >= len(lines) or start_line > end_line:
                            raise IndexError(f"行范围无效: {start_line+1}-{end_line+1}")
                        content = ''.join(lines[start_line:end_line+1])
                        return NBTString(content)
        else:
                        raise TypeError("读取模式的 '位' 参数必须是 [start_line, end_line] 格式")
            else:
                    # 读取整个文件
                    content = f.read()
                    return NBTString(content)
        except FileNotFoundError:
            raise FileNotFoundError(f"文件不存在: {filepath}")
        except Exception as e:
            raise RuntimeError(f"文件读取失败: {str(e)}")
    
    elif mode == "写":
        # 写入模式
        if content_expr is None:
            raise TypeError("写入模式需要 '文' 参数指定内容")
        
        content_val = _resolve_value(_eval_expr(content_expr, env), env)
        if not isinstance(content_val, NBTString):
            raise TypeError("文件 的 '文' 参数必须是字符串")
        
        content = content_val.value
        
        try:
            if position_expr is not None:
                position_val = _resolve_value(_eval_expr(position_expr, env), env)
                if isinstance(position_val, NBTString) and position_val.value == "全":
                    # 覆盖整个文件
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    return NBTString(f"已覆盖文件: {filepath}")
                elif isinstance(position_val, NBTArray) and len(position_val.data) == 2:
                    # 指定行范围覆盖
                    start_line = int(_resolve_value(position_val.data[0], env).value) - 1  # 转为0索引
                    end_line = int(_resolve_value(position_val.data[1], env).value) - 1
                    
                    # 读取现有文件
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                    except FileNotFoundError:
                        lines = []
                    
                    # 确保行数足够
                    while len(lines) <= end_line:
                        lines.append('\n')
                    
                    # 替换指定行范围
                    lines[start_line:end_line+1] = [content + '\n']
                    
                    # 写回文件
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                    
                    return NBTString(f"已修改文件 {filepath} 第{start_line+1}-{end_line+1}行")
            else:
                    raise TypeError("写入模式的 '位' 参数必须是 [start_line, end_line] 或 '全'")
                else:
                # 追加到文件末尾
                with open(filepath, 'a', encoding='utf-8') as f:
                    f.write(content)
                return NBTString(f"已追加到文件: {filepath}")
        except Exception as e:
            raise RuntimeError(f"文件写入失败: {str(e)}")
    
    else:
        raise TypeError(f"不支持的文件模式: {mode}")

cdef OP_IN_func(NBTCompound expr, _ScopeEnv env):
    # 包含检查: {"$": "在", "左": item, "右": container}
    item_expr = expr.data.get("左")
    container_expr = expr.data.get("右")
    
    if item_expr is None:
        raise TypeError("在 需要 '左' 参数指定要查找的项")
    if container_expr is None:
        raise TypeError("在 需要 '右' 参数指定容器")
    
    # 求值项和容器
    item_val = _resolve_value(_eval_expr(item_expr, env), env)
    container_val = _resolve_value(_eval_expr(container_expr, env), env)
    
    # 处理数组包含检查
    if isinstance(container_val, NBTArray):
        for element in container_val.data:
            resolved_element = _resolve_value(element, env)
            # 比较值
            if isinstance(item_val, NBTBase) and isinstance(resolved_element, NBTBase):
                if type(item_val) == type(resolved_element) and item_val.value == resolved_element.value:
                    return NBTByte(1)  # 找到了
            elif item_val == resolved_element:
                return NBTByte(1)  # 找到了
        return NBTByte(0)  # 没找到
    
    # 处理字符串包含检查
    elif isinstance(container_val, NBTString):
        if isinstance(item_val, NBTString):
            return NBTByte(1 if item_val.value in container_val.value else 0)
        else:
            return NBTByte(1 if str(item_val) in container_val.value else 0)
    
    # 处理字典包含检查
    elif isinstance(container_val, NBTCompound):
        # 检查是否为子字典
        if isinstance(item_val, NBTCompound):
            # 检查item_val的所有键值对是否都在container_val中
            for key, value in item_val.data.items():
                if key not in container_val.data:
                    return NBTByte(0)  # 键不存在
                container_value = _resolve_value(container_val.data[key], env)
                item_value = _resolve_value(value, env)
                # 比较值
                if isinstance(item_value, NBTBase) and isinstance(container_value, NBTBase):
                    if type(item_value) != type(container_value) or item_value.value != container_value.value:
                        return NBTByte(0)  # 值不匹配
                elif item_value != container_value:
                    return NBTByte(0)  # 值不匹配
            return NBTByte(1)  # 所有键值对都匹配
        else:
            # 检查键是否存在
            key_str = str(item_val)
            return NBTByte(1 if key_str in container_val.data else 0)
    
    else:
        raise TypeError("在 的 '右' 参数必须是数组、字符串或字典")

cdef OP_SEARCH_func(NBTCompound expr, _ScopeEnv env):
    # 查找过滤: {"$": "查", "匹": filter_function, "复合": data_structure}
    filter_expr = expr.data.get("匹")
    data_expr = expr.data.get("复合")
    
    if filter_expr is None:
        raise TypeError("查 需要 '匹' 参数指定过滤函数")
    if data_expr is None:
        raise TypeError("查 需要 '复合' 参数指定数据结构")
    
    # 求值过滤函数和数据结构
    filter_func = _eval_expr(filter_expr, env)
    data_val = _resolve_value(_eval_expr(data_expr, env), env)
    
    if not isinstance(filter_func, IONFunction):
        raise TypeError("查 的 '匹' 参数必须是函数")
    
    # 处理字典查找
    if isinstance(data_val, NBTCompound):
        result_data = {}
        for key, value in data_val.data.items():
            # 为每个键值对调用过滤函数
            resolved_value = _resolve_value(value, env)
            
            # 建立调用作用域
            call_env = _ScopeEnv(name_to_value=filter_func.closure_env, uuid_to_var=env.export_uuid_map(), parent=None).child()
            
            # 绑定参数：k=键, v=值
            if len(filter_func.param_names) >= 2:
                call_env.set_name(filter_func.param_names[0], _to_nbt(key))
                call_env.set_name(filter_func.param_names[1], _to_nbt(resolved_value))
            elif len(filter_func.param_names) == 1:
                # 只有一个参数，传递值
                call_env.set_name(filter_func.param_names[0], _to_nbt(resolved_value))
            
            # 调用过滤函数
            try:
                filter_result = _eval_block(filter_func.body, call_env)
            except _ReturnSignal as rs:
                filter_result = rs.value
            
            # 检查过滤结果
            filter_bool = _resolve_value(filter_result, env)
            is_match = False
            if isinstance(filter_bool, NBTByte):
                is_match = filter_bool.value != 0
            elif isinstance(filter_bool, (NBTShort, NBTInt, NBTLong)):
                is_match = filter_bool.value != 0
            elif isinstance(filter_bool, (NBTFloat, NBTDouble)):
                is_match = filter_bool.value != 0.0
            elif isinstance(filter_bool, NBTString):
                is_match = filter_bool.value != ""
            else:
                is_match = bool(filter_bool)
            
            if is_match:
                result_data[key] = value
        
        return NBTCompound(result_data)
    
    # 处理数组查找
    elif isinstance(data_val, NBTArray):
        result_items = []
        for i, element in enumerate(data_val.data):
            resolved_element = _resolve_value(element, env)
            
            # 建立调用作用域
            call_env = _ScopeEnv(name_to_value=filter_func.closure_env, uuid_to_var=env.export_uuid_map(), parent=None).child()
            
            # 绑定参数
            if len(filter_func.param_names) >= 2:
                call_env.set_name(filter_func.param_names[0], _to_nbt(i))  # 索引
                call_env.set_name(filter_func.param_names[1], _to_nbt(resolved_element))  # 值
            elif len(filter_func.param_names) == 1:
                call_env.set_name(filter_func.param_names[0], _to_nbt(resolved_element))  # 只传值
            
            # 调用过滤函数
            try:
                filter_result = _eval_block(filter_func.body, call_env)
            except _ReturnSignal as rs:
                filter_result = rs.value
            
            # 检查过滤结果
            filter_bool = _resolve_value(filter_result, env)
            is_match = False
            if isinstance(filter_bool, NBTByte):
                is_match = filter_bool.value != 0
            elif isinstance(filter_bool, (NBTShort, NBTInt, NBTLong)):
                is_match = filter_bool.value != 0
            elif isinstance(filter_bool, (NBTFloat, NBTDouble)):
                is_match = filter_bool.value != 0.0
            elif isinstance(filter_bool, NBTString):
                is_match = filter_bool.value != ""
            else:
                is_match = bool(filter_bool)
            
            if is_match:
                result_items.append(element)
        
        return NBTArray(result_items)
    
    else:
        raise TypeError("查 的 '复合' 参数必须是字典或数组")

cdef OP_CLASS_func(NBTCompound expr, _ScopeEnv env):
    # 类定义: {"$": "类", "域": fields, "法": methods, "介": interfaces, "初函": init_method, "标函": getitem_method, "记函": setitem_method, "转函": convert_method}
    fields = expr.data.get("域")
    methods = expr.data.get("法")
    impls = expr.data.get("介")
    init_method = expr.data.get("初函")
    getitem_method = expr.data.get("标函")
    setitem_method = expr.data.get("记函")
    convert_method = expr.data.get("转函")
    
    field_defaults = {}
    if isinstance(fields, NBTCompound):
        for k, v in fields.data.items():
            field_defaults[k] = _to_nbt(_resolve_value(_eval_expr(v, env), env))
    
    method_map = {}
    if isinstance(methods, NBTCompound):
        for mk, mv in methods.data.items():
            m = _eval_expr(mv, env)
            if not isinstance(m, IONFunction):
                raise TypeError("类 方法 需要 函 数值")
            method_map[mk] = m
    
    interfaces = []
    if isinstance(impls, NBTArray):
        for i in impls.data:
            iv = _eval_expr(i, env)
            if isinstance(iv, IONInterface):
                interfaces.append(iv)
    
    # 处理特殊方法名
    init_method_name = None
    getitem_method_name = None
    setitem_method_name = None
    convert_method_name = None
    
    if isinstance(init_method, NBTString):
        init_method_name = init_method.value
    elif init_method is not None:
        init_method_name = str(init_method)
        
    if isinstance(getitem_method, NBTString):
        getitem_method_name = getitem_method.value
    elif getitem_method is not None:
        getitem_method_name = str(getitem_method)
        
    if isinstance(setitem_method, NBTString):
        setitem_method_name = setitem_method.value
    elif setitem_method is not None:
        setitem_method_name = str(setitem_method)
        
    if isinstance(convert_method, NBTString):
        convert_method_name = convert_method.value
    elif convert_method is not None:
        convert_method_name = str(convert_method)
    
    return IONClass(field_defaults, method_map, interfaces, 
                   init_method_name, getitem_method_name, setitem_method_name, convert_method_name)

cdef OP_CREATE_INSTANCE_func(NBTCompound expr, _ScopeEnv env):
    # 实例创建: {"$": "造", "类": class_object, "传": args}
    cls_val = _eval_expr(expr.data.get("类"), env)
    if not isinstance(cls_val, IONClass):
        raise TypeError("造 的 '类' 需要 IONClass")
    
    # 创建实例（只使用类的默认域值）
    inst = IONInstance(cls_val)
    
    # 处理构造函数调用 - 所有参数都通过初函传递
    args_val = expr.data.get("传")
    if args_val is not None:
        # 获取初始化方法
        init_method = cls_val.get_init_method()
        if init_method is None:
            # 回退到传统的"初"方法
            init_method = cls_val.get_method("初")
        if init_method is None:
            raise RuntimeError("类无构造方法，但提供了参数")
        
        # 准备参数
        args = []
        if isinstance(args_val, NBTArray):
            args = [_resolve_value(_eval_expr(a, env), env) for a in args_val.data]
        else:
            # 单个参数也转换为数组
            args = [_resolve_value(_eval_expr(args_val, env), env)]
        
        # 调用初函
        _call_bound_method(inst, init_method, args, env)
    elif cls_val.get_init_method() is not None:
        # 如果有初始化方法但没有传参，调用无参初始化
        _call_bound_method(inst, cls_val.get_init_method(), [], env)
    
    return inst

cdef OP_GET_ATTR_func(NBTCompound expr, _ScopeEnv env):
    # 属性获取: {"$": "取属", "于": instance, "名": attr_name}
    objv = _eval_expr(expr.data.get("于"), env)
    namev = expr.data.get("名")
    if not isinstance(objv, IONInstance) or not isinstance(namev, NBTString):
        raise TypeError("取属 需要 IONInstance 和 NBTString")
    return objv.get_field(namev.value)

cdef OP_SET_ATTR_func(NBTCompound expr, _ScopeEnv env):
    # 属性设置: {"$": "记属", "于": instance, "名": attr_name, "为": value}
    objv = _eval_expr(expr.data.get("于"), env)
    namev = expr.data.get("名")
    valuev = _resolve_value(_eval_expr(expr.data.get("为"), env), env)
    if not isinstance(objv, IONInstance) or not isinstance(namev, NBTString):
        raise TypeError("记属 需要 IONInstance 和 NBTString")
    objv.set_field(namev.value, _to_nbt(valuev))
    return objv.get_field(namev.value)

cdef OP_CALL_ATTR_func(NBTCompound expr, _ScopeEnv env):
    # 方法调用: {"$": "呼属", "于": instance, "名": method_name, "传": args}
    objv = _eval_expr(expr.data.get("于"), env)
    namev = expr.data.get("名")
    argsv = expr.data.get("传")
    if not isinstance(objv, IONInstance) or not isinstance(namev, NBTString):
        raise TypeError("呼属 需要 IONInstance 和 NBTString")
    
    method = objv.cls.get_method(namev.value)
    if method is None:
        raise AttributeError(f"类没有方法: {namev.value}")
    
    args = []
    if isinstance(argsv, NBTArray):
        args = [_resolve_value(_eval_expr(a, env), env) for a in argsv.data]
    elif argsv is not None:
        args = [_resolve_value(_eval_expr(argsv, env), env)]
    
    return _call_bound_method(objv, method, args, env)

cdef OP_ADD_OP_SUB_OP_MUL_OP_DIV_OP_MOD_func(NBTCompound expr, int op_code, _ScopeEnv env):
    cdef object args, v, a
    cdef list nums
    cdef bint is_float = False
    cdef str op_name
    cdef double result_d, num_d
    cdef long result_i, num_i
    
    args = expr.data.get("与")
    if not isinstance(args, NBTArray):
        # 根据操作码确定操作名
        op_name = "未知操作"
        if op_code == OP_ADD:
            op_name = "加"
        elif op_code == OP_SUB:
            op_name = "减"
        elif op_code == OP_MUL:
            op_name = "乘"
        elif op_code == OP_DIV:
            op_name = "除"
        elif op_code == OP_MOD:
            op_name = "余"
        raise TypeError(f"操作{op_name} 需要数组参数 '与'")
    nums = []
    is_float = False
    for a in args.data:
        v = _resolve_value(_eval_expr(a, env), env)
        if isinstance(v, (NBTFloat, NBTDouble)):
            is_float = True
        nums.append(_number_of(v))
    if not nums:
        return NBTInt(0)
    if op_code == OP_ADD:
        acc = 0.0 if is_float else 0
        for n in nums:
            acc = acc + n
        return _nbt_number(float(acc) if is_float else int(acc))
    elif op_code == OP_SUB:
        acc = float(nums[0]) if (is_float or len(nums) > 1 and any(isinstance(_resolve_value(_eval_expr(a, env), env), (NBTFloat, NBTDouble)) for a in args.data)) else int(nums[0])
        for n in nums[1:]:
            acc = acc - n
        return _nbt_number(float(acc) if (isinstance(acc, float) or is_float) else int(acc))
    elif op_code == OP_MUL:
        acc = 1.0 if is_float else 1
        for n in nums:
            acc = acc * n
        return _nbt_number(float(acc) if is_float else int(acc))
    elif op_code == OP_DIV:
        acc = float(nums[0])
        for n in nums[1:]:
            acc = acc / n
        return NBTDouble(float(acc))
    elif op_code == OP_MOD:
        acc = int(nums[0])
        for n in nums[1:]:
            acc = acc % int(n)
        return _nbt_number(acc)

cdef OP_EQ_OP_GT_OP_LT_func(NBTCompound expr, int op_code, _ScopeEnv env):
    a = _resolve_value(_eval_expr(expr.data.get("左"), env), env)
    b = _resolve_value(_eval_expr(expr.data.get("右"), env), env)
    # 仅对数字与字符串提供定义良好的比较
    if isinstance(a, (NBTByte, NBTShort, NBTInt, NBTLong, NBTFloat, NBTDouble)) and \
       isinstance(b, (NBTByte, NBTShort, NBTInt, NBTLong, NBTFloat, NBTDouble)):
        av = _number_of(a)
        bv = _number_of(b)
    elif isinstance(a, NBTString) and isinstance(b, NBTString):
        av = a.value
        bv = b.value
    else:
        # 回退到字符串表示比较
        av = str(a)
        bv = str(b)
    if op_code == OP_EQ:
        return _bool_to_nbt(av == bv)
    elif op_code == OP_GT:
        return _bool_to_nbt(av > bv)
    elif op_code == OP_LT:
        return _bool_to_nbt(av < bv)

cdef OP_AND_OP_OR_OP_NOT_func(NBTCompound expr, int op_code, _ScopeEnv env):
    if op_code == OP_NOT:
        v = _resolve_value(_eval_expr(expr.data.get("值"), env), env)
        return _bool_to_nbt(not _is_truthy(v))
    a = _resolve_value(_eval_expr(expr.data.get("左"), env), env)
    if op_code == OP_AND:
        if not _is_truthy(a):
            return _bool_to_nbt(False)
        b = _resolve_value(_eval_expr(expr.data.get("右"), env), env)
        return _bool_to_nbt(_is_truthy(b))
    elif op_code == OP_OR:
        if _is_truthy(a):
            return _bool_to_nbt(True)
        b = _resolve_value(_eval_expr(expr.data.get("右"), env), env)
        return _bool_to_nbt(_is_truthy(b))

cdef OP_IF_func(NBTCompound expr, _ScopeEnv env):
    cond = _resolve_value(_eval_expr(expr.data.get("判"), env), env)
    branch = "是" if _is_truthy(cond) else "非"
    body = expr.data.get(branch)
    if body is None:
        return NBTEnd()
    return _eval_block(body, env)

cdef 造变_func(NBTCompound expr, _ScopeEnv env):
    """创建变量
    语法: 
    1. {"$": "造变", "值": init_value}  # 创建新变量
    2. {"$": "造变", "名": var_name}    # 指向已有变量（通过名字）
    3. {"$": "造变", "名": pointer}     # 指向已有变量（通过指针）
    """
    cdef object name_expr, value_expr, existing_value, name_val, init_val
    cdef NBTVar var
    
    name_expr = expr.data.get("名")
    value_expr = expr.data.get("值")
    
    if name_expr is not None:
        # 模式2和3：指向已有变量
        if isinstance(name_expr, NBTString):
            # 模式2：通过变量名指向已有变量
            existing_value = env.get_name(name_expr.value)
            if existing_value is None:
                raise RuntimeError(f"变量 '{name_expr.value}' 不存在，无法指向")
            
            # 如果现有值是指针，返回相同的指针（共享变量）
            if isinstance(existing_value, NBTPoint):
                return existing_value
            
            # 如果现有值不是指针，创建一个新变量来包装它
            var = NBTVar(_to_nbt(existing_value))
            env.register_var(var)
            # 同时更新原变量名指向这个新的变量指针
            env.set_name(name_expr.value, NBTPoint(var.uuid_str))
            return NBTPoint(var.uuid_str)
        else:
            # 模式3：通过表达式求值获得指针或变量
            name_val = _eval_expr(name_expr, env)
            if isinstance(name_val, NBTPoint):
                # 返回相同的指针（共享变量）
                return name_val
            elif isinstance(name_val, NBTVar):
                # 返回变量的指针
                return NBTPoint(name_val.uuid_str)
            else:
                raise TypeError(f"'名' 参数必须指向变量或指针，得到: {type(name_val)}")
    
    elif value_expr is not None:
        # 模式1：创建新变量
        init_val = _resolve_value(_eval_expr(value_expr, env), env)
        var = NBTVar(init_val)  # 直接存储原始对象，不进行 _to_nbt 转换
        env.register_var(var)
        return NBTPoint(var.uuid_str)
    
    else:
        raise TypeError("造变 需要 '值' 或 '名' 参数")

cdef 记_func(NBTCompound expr, _ScopeEnv env):
    cdef object target_raw, value, target
    
    target_raw = expr.data.get("位")
    value = _eval_expr(expr.data.get("为"), env)  # 不要自动解引用
    
    # 如果原始目标是字符串（NBTString），直接处理为变量名
    if isinstance(target_raw, NBTString):
        env.set_name(target_raw.value, value)  # 存储原始值（包括指针）
        return env.get_name(target_raw.value)
    
    # 否则求值目标表达式
    target = _eval_expr(target_raw, env)
    
    if isinstance(target, NBTPoint):
        var = env.get_var_by_uuid(target.uuid_str)
        if var is None:
            # 创建新变量而不是抛错
            var = NBTVar(_to_nbt(_resolve_value(value, env)), target.uuid_str)
            env.register_var(var)
            return var.get_value()
        # 对于指针赋值，需要解引用右值
        resolved_value = _resolve_value(value, env)
        var.set_value(_to_nbt(resolved_value))
        return var.get_value()
    if isinstance(target, NBTVar):
        env.register_var(target)
        resolved_value = _resolve_value(value, env)
        target.set_value(_to_nbt(resolved_value))
        return target.get_value()
    if isinstance(target, NBTString):
        resolved_value = _resolve_value(value, env)
        env.set_name(target.value, _to_nbt(resolved_value))
        return env.get_name(target.value)
    if isinstance(target, (NBTByte, NBTShort, NBTInt, NBTLong, NBTFloat, NBTDouble)):
        # 数字目标：可能是数组索引或其他用途，暂时忽略
        return _to_nbt(_resolve_value(value, env))
    if hasattr(target, '__class__'):
        # 其他对象类型：尝试设置属性（如果可能）
        return _to_nbt(_resolve_value(value, env))
    # 最宽松的处理：直接返回值
    return _to_nbt(_resolve_value(value, env))

cdef 改_func(NBTCompound expr, _ScopeEnv env):
    """改变变量的值，保持 UUID 不变
    语法: {"$": "改", "位": target, "为": new_value}
    - 如果 target 是变量名字符串，修改名字绑定的值
    - 如果 target 是指针 (NBTPoint)，修改指针指向的变量值
    - 如果 target 是变量 (NBTVar)，直接修改变量值
    """
    cdef object target_raw, new_value, existing_value, target, resolved_value
    cdef NBTVar var
    
    target_raw = expr.data.get("位")
    new_value = _eval_expr(expr.data.get("为"), env)
    
    if target_raw is None:
        raise TypeError("改 需要 '位' 参数指定目标")
    
    # 如果原始目标是字符串（NBTString），处理为变量名
    if isinstance(target_raw, NBTString):
        # 检查是否存在该名字的绑定
        existing_value = env.get_name(target_raw.value)
        if existing_value is None:
            raise RuntimeError(f"变量 '{target_raw.value}' 不存在，无法修改")
        
        # 如果绑定的是指针，修改指针指向的变量
        if isinstance(existing_value, NBTPoint):
            var = env.get_var_by_uuid(existing_value.uuid_str)
            if var is None:
                raise RuntimeError(f"指针指向的变量不存在: {existing_value.uuid_str}")
            resolved_value = _resolve_value(new_value, env)
            var.set_value(_to_nbt(resolved_value))
            return var.get_value()
        else:
            # 直接修改名字绑定
            resolved_value = _resolve_value(new_value, env)
            env.set_name(target_raw.value, _to_nbt(resolved_value))
            return env.get_name(target_raw.value)
    
    # 否则求值目标表达式
    target = _eval_expr(target_raw, env)
    
    if isinstance(target, NBTPoint):
        # 修改指针指向的变量值
        var = env.get_var_by_uuid(target.uuid_str)
        if var is None:
            raise RuntimeError(f"指针指向的变量不存在: {target.uuid_str}")
        resolved_value = _resolve_value(new_value, env)
        var.set_value(_to_nbt(resolved_value))
        return var.get_value()
    
    elif isinstance(target, NBTVar):
        # 直接修改变量值
        resolved_value = _resolve_value(new_value, env)
        target.set_value(_to_nbt(resolved_value))
        return target.get_value()
    
    elif isinstance(target, NBTString):
        # 字符串目标：修改对应的名字绑定
        existing_value = env.get_name(target.value)
        if existing_value is None:
            raise RuntimeError(f"变量 '{target.value}' 不存在，无法修改")
        resolved_value = _resolve_value(new_value, env)
        env.set_name(target.value, _to_nbt(resolved_value))
        return env.get_name(target.value)
    
    else:
        raise TypeError(f"无法修改类型 {type(target)} 的目标")

cdef 取_func(NBTCompound expr, _ScopeEnv env):
    cdef object ptr, name_val, got
    
    if "位" in expr.data:
        ptr = expr.data.get("位")
        # 位置也可能是表达式，先求值
        ptr = _eval_expr(ptr, env)
        if isinstance(ptr, NBTPoint):
            return _deref_pointer(ptr, env)
        elif isinstance(ptr, NBTVar):
            env.register_var(ptr)
            return ptr.get_value()
        elif isinstance(ptr, (NBTByte, NBTShort, NBTInt, NBTLong, NBTFloat, NBTDouble)):
            # 如果是数字，可能表示null指针或索引，返回0或NBTEnd
            if _number_of(ptr) == 0:
                return NBTEnd()
            else:
                return ptr  # 返回数字本身
        else:
            # 其他类型直接返回
            return ptr
    if "名" in expr.data:
        name_val = expr.data.get("名")
        if not isinstance(name_val, NBTString):
            raise TypeError("取 名 需要 NBTString")
        got = env.get_name(name_val.value)
        if got is None:
            return NBTEnd()
        return got
    raise TypeError("取 需要 '位' 或 '名'")

cdef 看_func(NBTCompound expr, _ScopeEnv env):  # 调试打印
    v = _resolve_value(_eval_expr(expr.data.get("值"), env), env)
    # 尽量打印 Python 原始值
    out = None
    if isinstance(v, (NBTByte, NBTShort, NBTInt, NBTLong)):
        out = int(v.value)
    elif isinstance(v, (NBTFloat, NBTDouble)):
        out = float(v.value)
    elif isinstance(v, NBTString):
        out = v.value
    else:
        out = v
    print(out)
    return _to_nbt(out)

cdef 函_func(NBTCompound expr, _ScopeEnv env):
    cdef object params_val, body, p
    cdef list param_names
    cdef IONFunction fn
    
    params_val = expr.data.get("参")
    body = expr.data.get("体")
    if not isinstance(params_val, NBTArray):
        raise TypeError("函 的 '参' 需要数组")
    param_names = []
    for p in params_val.data:
        p = _to_nbt(p)
        if not isinstance(p, NBTString):
            raise TypeError("参数名需要 NBTString")
        param_names.append(p.value)
    # 捕获闭包（名字绑定的浅拷贝）
    fn = IONFunction(param_names, body, env.name_to_value)
    return fn  # 作为 Python 对象存储

cdef 呼_func(NBTCompound expr, _ScopeEnv env):
    cdef object fn_val, args_val, bound, ret
    cdef list args
    cdef _ScopeEnv call_env
    cdef int i
    cdef str name
    
    fn_val = _eval_expr(expr.data.get("以"), env)
    if not isinstance(fn_val, IONFunction):
        raise TypeError("呼 的 '以' 需要 函 数值")
    args_val = expr.data.get("传")
    if args_val is None:
        args = []
    elif isinstance(args_val, NBTArray):
        args = [_eval_expr(a, env) for a in args_val.data]  # 不解引用，保持指针
    else:
        raise TypeError("呼 的 '传' 需要数组")
    # 建立调用作用域
    call_env = _ScopeEnv(name_to_value=fn_val.closure_env, uuid_to_var=env.export_uuid_map(), parent=None).child()
    # 形参与值绑定（保持原始类型，包括指针）
    for i, name in enumerate(fn_val.param_names):
        bound = args[i] if i < len(args) else NBTEnd()
        call_env.set_name(name, bound)
    try:
        ret = _eval_block(fn_val.body, call_env)
    except _ReturnSignal as rs:
        ret = rs.value
    return _to_nbt(_resolve_value(ret, call_env))

cdef 出_func(NBTCompound expr, _ScopeEnv env):
    val = _resolve_value(_eval_expr(expr.data.get("值"), env), env)
    raise _ReturnSignal(_to_nbt(val))

# ======== 循环指令 ========
cdef 循环_func(NBTCompound expr, _ScopeEnv env):
    # for循环: {"$": "循环", "从": start, "到": end, "步": step, "变": "i", "做": body}
    
    start_val = _resolve_value(_eval_expr(expr.data.get("从", 0), env), env)
    end_val = _resolve_value(_eval_expr(expr.data.get("到", 10), env), env)
    step_val = _resolve_value(_eval_expr(expr.data.get("步", 1), env), env)
    var_name = expr.data.get("变", "i")
    body = expr.data.get("做")
    
    if not isinstance(var_name, NBTString):
        var_name = NBTString(str(var_name))
    
    start_i = int(_number_of(start_val))
    end_i = int(_number_of(end_val))
    step_i = int(_number_of(step_val))
    
    if step_i == 0:
        step_i = 1  # 避免无限循环
    
    last_result = NBTEnd()
    
    if step_i > 0:
        i = start_i
        while i < end_i:
            env.set_name(var_name.value, NBTInt(i))
            try:
                last_result = _eval_block(body, env)
            except _ReturnSignal as rs:
                raise rs  # 传播返回信号
            except _BreakSignal:
                break  # 跳出循环
            except _ContinueSignal:
                pass  # 继续下次循环
            i += step_i
    else:
        i = start_i
        while i > end_i:
            env.set_name(var_name.value, NBTInt(i))
            try:
                last_result = _eval_block(body, env)
            except _ReturnSignal as rs:
                raise rs
            except _BreakSignal:
                break
            except _ContinueSignal:
                pass
            i += step_i
    
    return last_result

cdef 当_func(NBTCompound expr, _ScopeEnv env):
    # while循环: {"$": "当", "判": condition, "做": body}
    max_iterations = 10000  # 防止无限循环
    iteration_count = 0
    
    body = expr.data.get("做")
    last_result = NBTEnd()
    
    while iteration_count < max_iterations:
        condition = _resolve_value(_eval_expr(expr.data.get("判"), env), env)
        if not _is_truthy(condition):
            break
        
        try:
            last_result = _eval_block(body, env)
        except _ReturnSignal as rs:
            raise rs
        except _BreakSignal:
            break
        except _ContinueSignal:
            pass
        
        iteration_count += 1
    
    if iteration_count >= max_iterations:
        return NBTString("警告: 循环达到最大迭代次数限制")
    
    return last_result

cdef 遍历_func(NBTCompound expr, _ScopeEnv env):
    # 遍历数组: {"$": "遍历", "数组": array_expr, "变": "item", "做": body}
    
    array_val = _resolve_value(_eval_expr(expr.data.get("数组"), env), env)
    var_name = expr.data.get("变", "item")
    body = expr.data.get("做")
    
    if not isinstance(var_name, NBTString):
        var_name = NBTString(str(var_name))
    
    last_result = NBTEnd()
    
    if isinstance(array_val, NBTArray):
        arr_len = len(array_val.data)
        for i in range(arr_len):
            item = array_val.data[i]
            env.set_name(var_name.value, item)
            try:
                last_result = _eval_block(body, env)
            except _ReturnSignal as rs:
                raise rs
            except _BreakSignal:
                break
            except _ContinueSignal:
                continue
    elif isinstance(array_val, NBTCompound):
        # 遍历复合标签的值
        for key, value in array_val.data.items():
            env.set_name(var_name.value, value)
            try:
                last_result = _eval_block(body, env)
            except _ReturnSignal as rs:
                raise rs
            except _BreakSignal:
                break
            except _ContinueSignal:
                continue
    else:
        # 尝试转换为数组
        if hasattr(array_val, '__iter__'):
            for item in array_val:
                env.set_name(var_name.value, _to_nbt(item))
                try:
                    last_result = _eval_block(body, env)
                except _ReturnSignal as rs:
                    raise rs
                except _BreakSignal:
                    break
                except _ContinueSignal:
                    continue
    
    return last_result

cdef 跳出_func(NBTCompound expr, _ScopeEnv env):
    # 跳出循环 (break)
    raise _BreakSignal()

cdef 继续_func(NBTCompound expr, _ScopeEnv env):
    # 继续下次循环 (continue) 
    raise _ContinueSignal()

# ======== 拼接指令 ========
cdef 接_func(NBTCompound expr, _ScopeEnv env):
    # 拼接操作: {"$": "接", "与": [item1, item2, ...]}
    items_val = expr.data.get("与")
    if not isinstance(items_val, NBTArray):
        raise TypeError("接 的 '与' 需要数组参数")
    
    if len(items_val.data) == 0:
        return NBTEnd()
    
    # 求值所有参数
    resolved_items = []
    for item in items_val.data:
        resolved_items.append(_resolve_value(_eval_expr(item, env), env))
    
    if len(resolved_items) == 0:
        return NBTEnd()
    
    # 根据第一个元素的类型决定拼接方式
    first_item = resolved_items[0]
    
    # 字符串拼接
    if isinstance(first_item, NBTString):
        result_str = ""
        for item in resolved_items:
            if isinstance(item, NBTString):
                result_str += item.value
            elif isinstance(item, (NBTByte, NBTShort, NBTInt, NBTLong)):
                result_str += str(int(item.value))
            elif isinstance(item, (NBTFloat, NBTDouble)):
                result_str += str(float(item.value))
            else:
                # 对于其他NBT类型，尝试提取其值
                if isinstance(item, NBTString):
                    result_str += item.value
                elif hasattr(item, 'value'):
                    result_str += str(item.value)
                else:
                    result_str += str(item)
        return NBTString(result_str)
    
    # 数组拼接
    elif isinstance(first_item, NBTArray):
        result_array = NBTArray()
        for item in resolved_items:
            if isinstance(item, NBTArray):
                concat_len = len(item.data)
                for concat_idx in range(concat_len):
                    result_array.append(item.data[concat_idx])
            else:
                result_array.append(item)
        return result_array
    
    # 复合标签拼接（字典合并）
    elif isinstance(first_item, NBTCompound):
        result_compound = NBTCompound()
        for item in resolved_items:
            if isinstance(item, NBTCompound):
                for key, value in item.data.items():
                    result_compound.put(key, value)
            else:
                # 非复合标签作为键值对处理，使用索引作为键
                result_compound.put(str(len(result_compound.data)), item)
        return result_compound
    
    # 其他类型：转换为字符串拼接
    else:
        result_str = ""
        for item in resolved_items:
            if isinstance(item, NBTString):
                result_str += item.value
            elif isinstance(item, (NBTByte, NBTShort, NBTInt, NBTLong)):
                result_str += str(int(item.value))
            elif isinstance(item, (NBTFloat, NBTDouble)):
                result_str += str(float(item.value))
            else:
                # 对于其他NBT类型，尝试提取其值
                if isinstance(item, NBTString):
                    result_str += item.value
                elif hasattr(item, 'value'):
                    result_str += str(item.value)
                else:
                    result_str += str(item)
        return NBTString(result_str)







# ======== 接口 / 类 / 实例 / 成员 ========
cdef 接口_func(NBTCompound expr, _ScopeEnv env):
    req = expr.data.get("需")
    name_val = expr.data.get("名")
    if req is None or not isinstance(req, NBTArray):
        raise TypeError("接口 的 '需' 需要数组")
    needs = []
    for it in req.data:
        it = _to_nbt(it)
        if not isinstance(it, NBTString):
            raise TypeError("接口需求名需要 NBTString")
        needs.append(it.value)
    return IONInterface(set(needs), name_val.value if isinstance(name_val, NBTString) else None)

cdef 类_func(NBTCompound expr, _ScopeEnv env):
    fields = expr.data.get("域")
    methods = expr.data.get("法")
    impls = expr.data.get("介")
    init_method = expr.data.get("初函")
    getitem_method = expr.data.get("标函")
    setitem_method = expr.data.get("记函")
    convert_method = expr.data.get("转函")
    
    field_defaults = {}
    if isinstance(fields, NBTCompound):
        for k, v in fields.data.items():
            field_defaults[k] = _to_nbt(_resolve_value(_eval_expr(v, env), env))
    
    method_map = {}
    if isinstance(methods, NBTCompound):
        for mk, mv in methods.data.items():
            m = _eval_expr(mv, env)
            if not isinstance(m, IONFunction):
                raise TypeError("类 方法 需要 函 数值")
            method_map[mk] = m
    
    interfaces = []
    if isinstance(impls, NBTArray):
        for i in impls.data:
            iv = _eval_expr(i, env)
            if isinstance(iv, IONInterface):
                interfaces.append(iv)
    
    # 处理特殊方法名
    init_method_name = None
    getitem_method_name = None
    setitem_method_name = None
    convert_method_name = None
    
    if isinstance(init_method, NBTString):
        init_method_name = init_method.value
    elif init_method is not None:
        init_method_name = str(init_method)
        
    if isinstance(getitem_method, NBTString):
        getitem_method_name = getitem_method.value
    elif getitem_method is not None:
        getitem_method_name = str(getitem_method)
        
    if isinstance(setitem_method, NBTString):
        setitem_method_name = setitem_method.value
    elif setitem_method is not None:
        setitem_method_name = str(setitem_method)
        
    if isinstance(convert_method, NBTString):
        convert_method_name = convert_method.value
    elif convert_method is not None:
        convert_method_name = str(convert_method)
    
    return IONClass(field_defaults, method_map, interfaces, 
                   init_method_name, getitem_method_name, setitem_method_name, convert_method_name)

cdef 造_func(NBTCompound expr, _ScopeEnv env):
    cls_val = _eval_expr(expr.data.get("类"), env)
    if not isinstance(cls_val, IONClass):
        raise TypeError("造 的 '类' 需要 IONClass")
    
    # 创建实例（只使用类的默认域值）
    inst = IONInstance(cls_val)
    
    # 处理构造函数调用 - 所有参数都通过初函传递
    args_val = expr.data.get("传")
    if args_val is not None:
        # 获取初始化方法
        init_method = cls_val.get_init_method()
        if init_method is None:
            # 回退到传统的"初"方法
            init_method = cls_val.get_method("初")
        if init_method is None:
            raise RuntimeError("类无构造方法，但提供了参数")
        
        # 准备参数
        args = []
        if isinstance(args_val, NBTArray):
            args = [_resolve_value(_eval_expr(a, env), env) for a in args_val.data]
        else:
            # 单个参数也转换为数组
            args = [_resolve_value(_eval_expr(args_val, env), env)]
        
        # 调用初函
        _call_bound_method(inst, init_method, args, env)
    elif cls_val.get_init_method() is not None:
        # 如果有初始化方法但没有传参，调用无参初始化
        _call_bound_method(inst, cls_val.get_init_method(), [], env)
    
    return inst

cdef 取属_func(NBTCompound expr, _ScopeEnv env):
    objv = _eval_expr(expr.data.get("于"), env)
    namev = expr.data.get("名")
    if not isinstance(objv, IONInstance) or not isinstance(namev, NBTString):
        raise TypeError("取属 需要 实例 与 名字")
    return objv.get_field(namev.value)

cdef 记属_func(NBTCompound expr, _ScopeEnv env):
    objv = _eval_expr(expr.data.get("于"), env)
    namev = expr.data.get("名")
    valv = _resolve_value(_eval_expr(expr.data.get("为"), env), env)
    if not isinstance(objv, IONInstance) or not isinstance(namev, NBTString):
        raise TypeError("记属 需要 实例 与 名字")
    objv.set_field(namev.value, _to_nbt(valv))
    return objv.get_field(namev.value)

cdef 呼属_func(NBTCompound expr, _ScopeEnv env):
    objv = _eval_expr(expr.data.get("于"), env)
    namev = expr.data.get("名")
    args_val = expr.data.get("传")
    if not isinstance(objv, IONInstance) or not isinstance(namev, NBTString):
        raise TypeError("呼属 需要 实例 与 名字")
    m = objv.cls.get_method(namev.value)
    if m is None:
        raise AttributeError(f"实例无方法: {namev.value}")
    args = []
    if isinstance(args_val, NBTArray):
        args = [_resolve_value(_eval_expr(a, env), env) for a in args_val.data]
    return _call_bound_method(objv, m, args, env)

cdef 是类_func(NBTCompound expr, _ScopeEnv env):
    objv = _eval_expr(expr.data.get("值"), env)
    clsv = _eval_expr(expr.data.get("类"), env)
    return _bool_to_nbt(isinstance(objv, IONInstance) and isinstance(clsv, IONClass) and objv.cls is clsv)

cdef 验介_func(NBTCompound expr, _ScopeEnv env):
    objv = _eval_expr(expr.data.get("值"), env)
    iv = _eval_expr(expr.data.get("介"), env)
    if not isinstance(objv, IONInstance) or not isinstance(iv, IONInterface):
        return _bool_to_nbt(False)
    return _bool_to_nbt(objv.cls.implements(iv))

# ======== 文件操作指令 ========
cdef 存_func(NBTCompound expr, _ScopeEnv env):
    # 存储数据到文件: {"$": "存", "名": "filename", "值": data}
    name_val = expr.data.get("名")
    data_val = _resolve_value(_eval_expr(expr.data.get("值"), env), env)
    
    if not isinstance(name_val, NBTString):
        raise TypeError("存 的 '名' 需要 NBTString")
    
    filename = name_val.value + ".ION"
    
    try:
        import pickle
        import os
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
        
        # 序列化数据
        if isinstance(data_val, NBTBase):
            # NBT对象转换为可序列化的字典
            serialized_data = {
                "type": "nbt",
                "data": data_val.to_dict() if hasattr(data_val, 'to_dict') else str(data_val)
            }
        elif isinstance(data_val, IONFunction):
            # IONFunction 特殊处理
            serialized_data = {
                "type": "ion_function",
                "param_names": data_val.param_names,
                "body": data_val.body,
                "closure_env": data_val.closure_env
            }
        else:
            # 其他Python对象
            try:
                serialized_data = {
                    "type": "python",
                    "data": data_val
                }
            except:
                # 无法序列化的对象转为字符串
                serialized_data = {
                    "type": "string",
                    "data": str(data_val)
                }
        
        with open(filename, 'wb') as f:
            pickle.dump(serialized_data, f)
        
        return NBTString(f"已保存到 {filename}")
        
    except Exception as e:
        return NBTString(f"保存失败: {str(e)}")

cdef 载_func(NBTCompound expr, _ScopeEnv env):
    # 加载文件数据: {"$": "载", "路": "path"}
    path_val = expr.data.get("路")
    
    if not isinstance(path_val, NBTString):
        raise TypeError("载 的 '路' 需要 NBTString")
    
    filepath = path_val.value
    
    try:
        import pickle
        import os
        
        if not os.path.exists(filepath):
            return NBTString(f"文件不存在: {filepath}")
        
        with open(filepath, 'rb') as f:
            serialized_data = pickle.load(f)
        
        if isinstance(serialized_data, dict) and "type" in serialized_data:
            if serialized_data["type"] == "nbt":
                # 尝试重建NBT对象
                data = serialized_data["data"]
                if isinstance(data, dict) and "type" in data:
                    nbt_type = data["type"]
                    if nbt_type == 7:  # NBTString
                        return NBTString(data.get("value", ""))
                    elif nbt_type == 3:  # NBTInt
                        return NBTInt(data.get("value", 0))
                    elif nbt_type == 6:  # NBTDouble
                        return NBTDouble(data.get("value", 0.0))
                    elif nbt_type == 8:  # NBTCompound
                        compound = NBTCompound()
                        if "data" in data:
                            for k, v in data["data"].items():
                                compound.put(k, _to_nbt(v))
                        return compound
                    elif nbt_type == 9:  # NBTArray
                        array = NBTArray()
                        if "data" in data:
                            for item in data["data"]:
                                array.append(_to_nbt(item))
                        return array
                return NBTString(str(data))
            elif serialized_data["type"] == "ion_function":
                # 重建IONFunction
                return IONFunction(
                    serialized_data["param_names"],
                    serialized_data["body"],
                    serialized_data["closure_env"]
                )
            elif serialized_data["type"] == "string":
                # 字符串数据
                return NBTString(serialized_data["data"])
            else:
                # Python对象
                return _to_nbt(serialized_data["data"])
        else:
            # 直接返回数据
            return _to_nbt(serialized_data)
            
    except Exception as e:
        return NBTString(f"加载失败: {str(e)}")

# ======== 扩展指令：更包容的操作 ========
cdef 空_func(NBTCompound expr, _ScopeEnv env):
    # 检查是否为空/null
    val = _resolve_value(_eval_expr(expr.data.get("值"), env), env)
    if isinstance(val, NBTEnd):
        return _bool_to_nbt(True)
    if isinstance(val, (NBTByte, NBTShort, NBTInt, NBTLong)) and _number_of(val) == 0:
        return _bool_to_nbt(True)
    if isinstance(val, NBTString) and len(val.value) == 0:
        return _bool_to_nbt(True)
    return _bool_to_nbt(False)

cdef 非空_func(NBTCompound expr, _ScopeEnv env):
    # 检查是否非空
    val = _resolve_value(_eval_expr(expr.data.get("值"), env), env)
    return _bool_to_nbt(_is_truthy(val))

cdef 类型_func(NBTCompound expr, _ScopeEnv env):
    # 获取值的类型
    val = _resolve_value(_eval_expr(expr.data.get("值"), env), env)
    if isinstance(val, NBTBase):
        return NBTString(type(val).__name__)
    return NBTString(str(type(val).__name__))

cdef 安全取_func(NBTCompound expr, _ScopeEnv env):
    # 安全的取值操作，不会抛错
    if "位" in expr.data:
        ptr = expr.data.get("位")
        try:
            ptr = _eval_expr(ptr, env)
            if isinstance(ptr, NBTPoint):
                return _resolve_value(ptr, env)  # 使用改进的_resolve_value
            elif isinstance(ptr, NBTVar):
                env.register_var(ptr)
                return ptr.get_value()
            else:
                return ptr
        except:
            return NBTEnd()
    if "名" in expr.data:
        name_val = expr.data.get("名")
        try:
            if isinstance(name_val, NBTString):
                got = env.get_name(name_val.value)
                return got if got is not None else NBTEnd()
            return NBTEnd()
        except:
            return NBTEnd()
    return NBTEnd()

cdef 安全记_func(NBTCompound expr, _ScopeEnv env):
    # 安全的赋值操作，不会抛错
    target_raw = expr.data.get("位")
    value = _eval_expr(expr.data.get("为"), env)
    
    try:
        if isinstance(target_raw, NBTString):
            env.set_name(target_raw.value, value)
            return env.get_name(target_raw.value)
        
        target = _eval_expr(target_raw, env)
        
        if isinstance(target, NBTPoint):
            var = env.get_var_by_uuid(target.uuid_str)
            if var is None:
                var = NBTVar(_to_nbt(_resolve_value(value, env)), target.uuid_str)
                env.register_var(var)
            else:
                resolved_value = _resolve_value(value, env)
                var.set_value(_to_nbt(resolved_value))
            return var.get_value()
        elif isinstance(target, NBTVar):
            env.register_var(target)
            resolved_value = _resolve_value(value, env)
            target.set_value(_to_nbt(resolved_value))
            return target.get_value()
        elif isinstance(target, NBTString):
            resolved_value = _resolve_value(value, env)
            env.set_name(target.value, _to_nbt(resolved_value))
            return env.get_name(target.value)
        else:
            # 其他情况：直接返回值
            return _to_nbt(_resolve_value(value, env))
    except:
        # 任何错误都返回原值
        return _to_nbt(_resolve_value(value, env))

def _eval_expr(expr, env: _ScopeEnv):
    # 循环相关的 cdef 变量声明
    # 仅保留当前简化版本需要的变量
    cdef object op_name_val, op
    
    # 先做 Python->NBT 的宽松转换，便于直接传入 Python 基础类型或 dict/list
    expr = _to_nbt(expr)

    # 原子类型直接返回（但变量需注册）
    if isinstance(expr, NBTVar):
        env.register_var(expr)
        return expr
    if not isinstance(expr, NBTCompound):
        return expr

    # 指令节点：寻找 key "$"
    op_name_val = expr.data.get("$")
    if not isinstance(op_name_val, NBTString):
        # 普通数据聚合；不隐式递归求值
        return expr
    op = op_name_val.value

    # 使用 B+树 进行快速函数查找，然后进行分派
    cdef object handler_func = _get_function(op)
    
    # 基于函数的快速分派
    if handler_func is not None:
        # 根据操作类型调用不同的处理器
        if op in ("加", "减", "乘", "除", "余"):
            return handler_func(expr, _get_op_code(op), env)
        elif op in ("等", "大于", "小于"):
            return handler_func(expr, _get_op_code(op), env)
        elif op in ("且", "或", "非"):
            return handler_func(expr, _get_op_code(op), env)
        else:
            return handler_func(expr, env)
    
    # 未知指令：原样返回
    return expr


def ion_run(program, initial_names: dict = None, env: _ScopeEnv = None, return_env: bool = False):
    """执行一个 ION NBT 程序
    
    Args:
        program: ION 程序表达式
        initial_names: 初始变量名映射
        env: 可选的作用域环境，如果不提供则创建新环境
        return_env: 是否返回执行后的环境
    
    Returns:
        如果 return_env=False: 程序执行结果
        如果 return_env=True: (程序执行结果, 执行后的环境)
    """
    if env is None:
        env = _ScopeEnv()
    
    if initial_names:
        for k, v in initial_names.items():
            env.set_name(k, _to_nbt(v))
    
    result = _eval_expr(program, env)
    
    if return_env:
        return result, env
    else:
        return result


def ion_eval(expr, names: dict = None, uuids: dict = None, env: _ScopeEnv = None, return_env: bool = False):
    """对单一表达式求值（比 ion_run 更轻量，不包裹序列）
    
    Args:
        expr: 表达式
        names: 变量名映射
        uuids: UUID变量映射
        env: 可选的作用域环境，如果不提供则创建新环境
        return_env: 是否返回执行后的环境
    
    Returns:
        如果 return_env=False: 表达式执行结果
        如果 return_env=True: (表达式执行结果, 执行后的环境)
    """
    if env is None:
        env = _ScopeEnv()
    
    if names:
        for k, v in names.items():
            env.set_name(k, _to_nbt(v))
    if uuids:
        for uuid_str, v in uuids.items():
            # 为 UUID 变量创建 NBTVar 并注册
            var = NBTVar(_to_nbt(v), uuid_str)
            env.register_var(var)
    
    result = _eval_expr(expr, env)
    
    if return_env:
        return result, env
    else:
        return result
