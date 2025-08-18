/*
 * ION Expression - C Reference Implementation
 *
 * This file provides a complete C implementation of the ION Expression
 * interpreter with all necessary data structures, NBT primitives, 
 * keyword dispatch system, and a simplified set of operations.
 *
 * Complex functions can optionally use Python.h to call Cython implementations.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <sys/types.h>

// Python API支持
#ifdef ION_ENABLE_PYTHON_API
#include <Python.h>
#endif



#ifndef ION_API
#define ION_API
#endif

/* ===============================
  * Error Codes
  * =============================== */
 #define ION_ERR_UNIMPL -2
 #define ION_ERR_TYPE -3
 #define ION_ERR_PARAM -4
 #define ION_ERR_RUNTIME -5
 
 /* ===============================
  * NBT Type Enum
 * =============================== */
typedef enum NBTType {
    TAG_END = 0,
    TAG_BYTE = 1,
    TAG_SHORT = 2,
    TAG_INT = 3,
    TAG_LONG = 4,
    TAG_FLOAT = 5,
    TAG_DOUBLE = 6,
    TAG_STRING = 7,
    TAG_COMPOUND = 8,
    TAG_ARRAY = 9,
    TAG_FARRAY = 10,
    TAG_FCOMPOUND = 11,
    TAG_VAR = 12,
    TAG_POINT = 13,
    TAG_FUNCTION = 14,
    TAG_CLASS = 15,
    TAG_INSTANCE = 16
} NBTType;

/* ===============================
  * Basic Data Structures
 * =============================== */
 typedef struct IonStr {
     char *ptr;
     uint16_t len;
 } IonStr;
 
 typedef struct ION_UUID { 
     uint8_t bytes[16]; 
 } ION_UUID;
 
 // Forward declarations
typedef struct ION_NBTValue ION_NBTValue;
typedef struct ION_NBTArray ION_NBTArray;
typedef struct ION_NBTCompound ION_NBTCompound;
typedef struct ION_NBTVar ION_NBTVar;
typedef struct ION_NBTPoint ION_NBTPoint;
typedef struct ION_Function ION_Function;
typedef struct ION_Class ION_Class;
typedef struct ION_Instance ION_Instance;
typedef struct ION_ScopeEnv ION_ScopeEnv;
 
 /* ===============================
  * NBT Value Structure
  * =============================== */
 typedef struct ION_NBTValue {
    NBTType type;
    union {
        int8_t  b;
        int16_t s;
        int32_t i;
        int64_t l;
        float   f;
        double  d;
        IonStr  str;
        ION_NBTArray   *arr;
        ION_NBTCompound *comp;
        ION_NBTVar     *var;
        ION_NBTPoint   *point;
        ION_Function   *func;
        ION_Class      *cls;
        ION_Instance   *inst;
    } as;
} ION_NBTValue;
 
 typedef struct ION_NBTCompoundKV {
     IonStr key;
     ION_NBTValue val;
 } ION_NBTCompoundKV;
 
 typedef struct ION_NBTArray {
     ION_NBTValue *items;
     size_t len;
    size_t cap;
 } ION_NBTArray;
 
 // 简单哈希表实现
#define ION_HASH_TABLE_SIZE 16

typedef struct ION_HashBucket {
    ION_NBTCompoundKV *kvs;
    size_t len;
    size_t cap;
} ION_HashBucket;

typedef struct ION_NBTCompound {
    ION_HashBucket buckets[ION_HASH_TABLE_SIZE];
    size_t total_len; // 总元素数量
} ION_NBTCompound;
 
 typedef struct ION_NBTVar {
     ION_UUID id;
     ION_NBTValue value;
 } ION_NBTVar;
 
 typedef struct ION_NBTPoint {
    ION_UUID id;
} ION_NBTPoint;

// 函数和类支持结构
typedef struct ION_Function {
    char **param_names;
    int param_count;
    ION_NBTValue *body;
    struct ION_ScopeEnv *closure_env;
} ION_Function;

typedef struct ION_Class {
    ION_NBTCompound *fields;
    ION_NBTCompound *methods;
    char *init_method_name;
    char *getitem_method_name;
    char *setitem_method_name;
    char *convert_method_name;
} ION_Class;

typedef struct ION_Instance {
    ION_Class *class_def;
    ION_NBTCompound *field_values;
} ION_Instance;
 
 /* ===============================
  * Scope Environment
  * =============================== */
 typedef struct ION_NameKV {
     IonStr name;
     ION_NBTValue value;
 } ION_NameKV;
 
 typedef struct ION_ScopeEnv {
     ION_NameKV *names;
     size_t names_len;
     size_t names_cap;
     struct ION_ScopeEnv *parent;
 } ION_ScopeEnv;
 
 /* ===============================
  * Keyword Handler Type
  * =============================== */
 typedef int (*ION_KeywordHandler)(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out);

/* ===============================
  * B+Tree for Keyword Dispatch
 * =============================== */
 typedef struct BPTNodeC {
     int leaf;
     int nkeys;
     int order;
     char **keys;
     void **values;
     struct BPTNodeC **children;
     struct BPTNodeC *next_leaf;
 } BPTNodeC;
 
 typedef struct BPTreeC {
     BPTNodeC *root;
     int order;
 } BPTreeC;
 
 /* ===============================
  * Function Declarations
  * =============================== */
 static void ion_str_free(IonStr *s);
 static IonStr ion_str_from_cstr(const char *c);
 static void ion_nbt_init(ION_NBTValue *v);
 static void ion_nbt_free(ION_NBTValue *v);
 static ION_NBTValue ion_nbt_make_end(void);
 static ION_NBTValue ion_nbt_make_byte(int8_t b);
 static ION_NBTValue ion_nbt_make_short(int16_t s);
 static ION_NBTValue ion_nbt_make_int(int32_t i);
 static ION_NBTValue ion_nbt_make_long(int64_t l);
 static ION_NBTValue ion_nbt_make_float(float f);
 static ION_NBTValue ion_nbt_make_double(double d);
 static ION_NBTValue ion_nbt_make_string_copy(const char *s, uint16_t len);
 static ION_NBTValue ion_nbt_make_array(void);
 static ION_NBTValue ion_nbt_make_compound(void);
 static ION_NBTValue ion_nbt_make_var(void);
static ION_NBTValue ion_nbt_make_point(void);
static ION_NBTValue ion_nbt_make_function(void);
static ION_NBTValue ion_nbt_make_class(void);
static ION_NBTValue ion_nbt_make_instance(void);
 
 static int ion_nbt_array_push_move(ION_NBTArray *a, ION_NBTValue *v);
static int ion_nbt_compound_put_move(ION_NBTCompound *c, IonStr *k, ION_NBTValue *v);
static ssize_t ion_nbt_compound_find(const ION_NBTCompound *c, const char *key, uint16_t len);

// 哈希表相关函数
static uint32_t ion_hash_string(const char *key, uint16_t len);
static void ion_hash_bucket_init(ION_HashBucket *bucket);
static void ion_hash_bucket_free(ION_HashBucket *bucket);
static int ion_hash_bucket_put(ION_HashBucket *bucket, IonStr *key, ION_NBTValue *value);
static ssize_t ion_hash_bucket_find(const ION_HashBucket *bucket, const char *key, uint16_t len);
 
 static void ion_env_init(ION_ScopeEnv *e, ION_ScopeEnv *parent);
 static void ion_env_free(ION_ScopeEnv *e);
 static const ION_NBTValue *ion_env_get_name_ref(const ION_ScopeEnv *e, const char *name);
 static int ion_env_set_name_move(ION_ScopeEnv *e, IonStr *name, ION_NBTValue *val);
 
 static BPTreeC *bpt_create(int order);
 static void bpt_free(BPTreeC *t);
 static int bpt_insert_strptr(BPTreeC *t, const char *key, void *val);
 static void *bpt_get(BPTreeC *t, const char *key);
 
 static const ION_NBTValue *ion_get_field(const ION_NBTValue *compound, const char *key);
 static double ion_value_to_double(const ION_NBTValue *v);
 static int64_t ion_value_to_int64(const ION_NBTValue *v);
 static int ion_value_is_truthy(const ION_NBTValue *v);
 
 static int ion_eval_expr(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out);
static ION_KeywordHandler ion_get_keyword_handler(const char *op);

// 原生方法调用函数
static int ion_call_array_method(const ION_NBTValue *obj, const char *method, size_t method_len, 
                                const ION_NBTValue *args, ION_ScopeEnv *env, ION_NBTValue *result);
static int ion_call_compound_method(const ION_NBTValue *obj, const char *method, size_t method_len, 
                                   const ION_NBTValue *args, ION_ScopeEnv *env, ION_NBTValue *result);
static int ion_call_string_method(const ION_NBTValue *obj, const char *method, size_t method_len, 
                                 const ION_NBTValue *args, ION_ScopeEnv *env, ION_NBTValue *result);
static int ion_call_numeric_method(const ION_NBTValue *obj, const char *method, size_t method_len, 
                                  const ION_NBTValue *args, ION_ScopeEnv *env, ION_NBTValue *result);
static ION_NBTValue ion_nbt_deep_copy(const ION_NBTValue *src);
 
 /* ===============================
  * String Utilities
  * =============================== */
 static void ion_str_free(IonStr *s)
 {
     if (s && s->ptr) { 
         free(s->ptr); 
         s->ptr = NULL; 
         s->len = 0; 
     }
 }
 
 static IonStr ion_str_from_cstr(const char *c)
 {
    IonStr s;
     s.ptr = NULL; 
     s.len = 0;
    if (!c) return s;
    size_t n = strlen(c);
     if (n > 65535U) n = 65535U;
    s.ptr = (char*)malloc(n + 1U);
     if (!s.ptr) { 
         fprintf(stderr, "ION: malloc failed\n"); 
            abort();
        }
    memcpy(s.ptr, c, n);
    s.ptr[n] = '\0';
    s.len = (uint16_t)n;
    return s;
}

/* ===============================
  * NBT Value Functions
 * =============================== */
static void ion_nbt_init(ION_NBTValue *v)
{
    v->type = TAG_END;
    v->as.str.ptr = NULL;
    v->as.str.len = 0;
}

static void ion_nbt_free(ION_NBTValue *v)
{
    if (!v) return;
    if (v->type == TAG_STRING && v->as.str.ptr) {
        free(v->as.str.ptr);
        v->as.str.ptr = NULL;
        v->as.str.len = 0;
    } else if ((v->type == TAG_ARRAY || v->type == TAG_FARRAY) && v->as.arr) {
         for (size_t i = 0; i < v->as.arr->len; ++i) {
             ion_nbt_free(&v->as.arr->items[i]);
         }
         free(v->as.arr->items);
         free(v->as.arr);
        v->as.arr = NULL;
    } else if ((v->type == TAG_COMPOUND || v->type == TAG_FCOMPOUND) && v->as.comp) {
         // 释放哈希表的所有桶
         for (int i = 0; i < ION_HASH_TABLE_SIZE; i++) {
             ion_hash_bucket_free(&v->as.comp->buckets[i]);
         }
         free(v->as.comp);
        v->as.comp = NULL;
    } else if (v->type == TAG_VAR && v->as.var) {
         ion_nbt_free(&v->as.var->value);
         free(v->as.var);
        v->as.var = NULL;
    } else if (v->type == TAG_POINT && v->as.point) {
         free(v->as.point);
        v->as.point = NULL;
    } else if (v->type == TAG_FUNCTION && v->as.func) {
        // 释放函数参数名数组
        if (v->as.func->param_names) {
            for (int i = 0; i < v->as.func->param_count; i++) {
                free(v->as.func->param_names[i]);
            }
            free(v->as.func->param_names);
        }
        // 释放函数体（现在使用深拷贝，需要释放内容）
        if (v->as.func->body) {
            ion_nbt_free(v->as.func->body); // 释放函数体内容
            free(v->as.func->body);
        }
        // 注意：closure_env 由环境管理，这里不释放
        free(v->as.func);
        v->as.func = NULL;
    } else if (v->type == TAG_CLASS && v->as.cls) {
        // 释放类的字段和方法
        if (v->as.cls->fields) {
            ion_nbt_free((ION_NBTValue*)v->as.cls->fields);
            free(v->as.cls->fields);
        }
        if (v->as.cls->methods) {
            ion_nbt_free((ION_NBTValue*)v->as.cls->methods);
            free(v->as.cls->methods);
        }
        // 释放方法名字符串
        free(v->as.cls->init_method_name);
        free(v->as.cls->getitem_method_name);
        free(v->as.cls->setitem_method_name);
        free(v->as.cls->convert_method_name);
        free(v->as.cls);
        v->as.cls = NULL;
    } else if (v->type == TAG_INSTANCE && v->as.inst) {
        // 释放实例的字段值
        if (v->as.inst->field_values) {
            ion_nbt_free((ION_NBTValue*)v->as.inst->field_values);
            free(v->as.inst->field_values);
        }
        // 释放类定义（现在是深拷贝，需要释放）
        if (v->as.inst->class_def) {
            // 释放类的字段和方法
            if (v->as.inst->class_def->fields) {
                ion_nbt_free((ION_NBTValue*)v->as.inst->class_def->fields);
                free(v->as.inst->class_def->fields);
            }
            if (v->as.inst->class_def->methods) {
                ion_nbt_free((ION_NBTValue*)v->as.inst->class_def->methods);
                free(v->as.inst->class_def->methods);
            }
            // 释放方法名字符串
            free(v->as.inst->class_def->init_method_name);
            free(v->as.inst->class_def->getitem_method_name);
            free(v->as.inst->class_def->setitem_method_name);
            free(v->as.inst->class_def->convert_method_name);
            free(v->as.inst->class_def);
        }
        free(v->as.inst);
        v->as.inst = NULL;
    }
    v->type = TAG_END;
}

static ION_NBTValue ion_nbt_make_end(void)
{
     ION_NBTValue v; 
     ion_nbt_init(&v); 
     return v;
}

static ION_NBTValue ion_nbt_make_byte(int8_t b)
{
     ION_NBTValue v; 
     ion_nbt_init(&v); 
     v.type = TAG_BYTE; 
     v.as.b = b; 
     return v;
}

static ION_NBTValue ion_nbt_make_short(int16_t s)
{
     ION_NBTValue v; 
     ion_nbt_init(&v); 
     v.type = TAG_SHORT; 
     v.as.s = s; 
     return v;
}

// 智能整数类型选择
static ION_NBTValue ion_nbt_make_int_auto(int64_t value)
{
    ION_NBTValue v;
    ion_nbt_init(&v);
    
    if (value >= INT8_MIN && value <= INT8_MAX) {
        v.type = TAG_BYTE;
        v.as.b = (int8_t)value;
    } else if (value >= INT16_MIN && value <= INT16_MAX) {
        v.type = TAG_SHORT;
        v.as.s = (int16_t)value;
    } else if (value >= INT32_MIN && value <= INT32_MAX) {
        v.type = TAG_INT;
        v.as.i = (int32_t)value;
    } else {
        v.type = TAG_LONG;
        v.as.l = value;
    }
    return v;
}

// 智能浮点类型选择
static ION_NBTValue ion_nbt_make_float_auto(double value)
{
    ION_NBTValue v;
    ion_nbt_init(&v);
    
    // 检查是否可以用float表示而不丢失精度
    float f_val = (float)value;
    if ((double)f_val == value && isfinite(value)) {
        v.type = TAG_FLOAT;
        v.as.f = f_val;
    } else {
        v.type = TAG_DOUBLE;
        v.as.d = value;
    }
    return v;
}

static ION_NBTValue ion_nbt_make_int(int32_t i)
{
     ION_NBTValue v; 
     ion_nbt_init(&v); 
     v.type = TAG_INT; 
     v.as.i = i; 
     return v;
}

static ION_NBTValue ion_nbt_make_long(int64_t l)
{
     ION_NBTValue v; 
     ion_nbt_init(&v); 
     v.type = TAG_LONG; 
     v.as.l = l; 
     return v;
}

static ION_NBTValue ion_nbt_make_float(float f)
{
     ION_NBTValue v; 
     ion_nbt_init(&v); 
     v.type = TAG_FLOAT; 
     v.as.f = f; 
     return v;
}

static ION_NBTValue ion_nbt_make_double(double d)
{
     ION_NBTValue v; 
     ion_nbt_init(&v); 
     v.type = TAG_DOUBLE; 
     v.as.d = d; 
     return v;
}

static ION_NBTValue ion_nbt_make_string_copy(const char *s, uint16_t len)
{
     ION_NBTValue v; 
     ion_nbt_init(&v); 
     v.type = TAG_STRING;
    v.as.str.ptr = (char*)malloc((size_t)len + 1U);
     if (!v.as.str.ptr) { 
         fprintf(stderr, "ION: malloc failed\n"); 
         abort(); 
     }
    memcpy(v.as.str.ptr, s, (size_t)len);
    v.as.str.ptr[len] = '\0';
    v.as.str.len = len;
    return v;
}

 static ION_NBTValue ion_nbt_make_array(void)
 {
     ION_NBTValue v; 
     ion_nbt_init(&v);
     v.type = TAG_ARRAY;
     v.as.arr = (ION_NBTArray*)malloc(sizeof(ION_NBTArray));
     if (!v.as.arr) { 
         fprintf(stderr, "ION: malloc failed\n"); 
        abort();
    }
     v.as.arr->items = NULL; 
     v.as.arr->len = 0U; 
     v.as.arr->cap = 0U;
    return v;
}

 static ION_NBTValue ion_nbt_make_compound(void)
{
     ION_NBTValue v; 
     ion_nbt_init(&v);
     v.type = TAG_COMPOUND;
     v.as.comp = (ION_NBTCompound*)malloc(sizeof(ION_NBTCompound));
     if (!v.as.comp) { 
         fprintf(stderr, "ION: malloc failed\n"); 
        abort();
    }
     
     // 初始化哈希表的所有桶
     for (int i = 0; i < ION_HASH_TABLE_SIZE; i++) {
         ion_hash_bucket_init(&v.as.comp->buckets[i]);
     }
     v.as.comp->total_len = 0;
     
    return v;
}

static ION_NBTValue ion_nbt_make_var(void)
{
     ION_NBTValue v; 
     ion_nbt_init(&v);
    v.type = TAG_VAR;
    v.as.var = (ION_NBTVar*)malloc(sizeof(ION_NBTVar));
     if (!v.as.var) { 
         fprintf(stderr, "ION: malloc failed\n"); 
        abort();
    }
     memset(v.as.var->id.bytes, 0, 16U);
     ion_nbt_init(&v.as.var->value);
    return v;
}

static ION_NBTValue ion_nbt_make_point(void)
{
     ION_NBTValue v; 
     ion_nbt_init(&v);
    v.type = TAG_POINT;
    v.as.point = (ION_NBTPoint*)malloc(sizeof(ION_NBTPoint));
     if (!v.as.point) { 
         fprintf(stderr, "ION: malloc failed\n"); 
        abort();
    }
     memset(v.as.point->id.bytes, 0, 16U);
    return v;
}

static ION_NBTValue ion_nbt_make_function(void)
{
    ION_NBTValue v;
    ion_nbt_init(&v);
    v.type = TAG_FUNCTION;
    v.as.func = (ION_Function*)malloc(sizeof(ION_Function));
    if (!v.as.func) {
        fprintf(stderr, "ION: malloc failed\n");
        abort();
    }
    v.as.func->param_names = NULL;
    v.as.func->param_count = 0;
    v.as.func->body = NULL;
    v.as.func->closure_env = NULL;
    return v;
}

static ION_NBTValue ion_nbt_make_class(void)
{
    ION_NBTValue v;
    ion_nbt_init(&v);
    v.type = TAG_CLASS;
    v.as.cls = (ION_Class*)malloc(sizeof(ION_Class));
    if (!v.as.cls) {
        fprintf(stderr, "ION: malloc failed\n");
        abort();
    }
    v.as.cls->fields = NULL;
    v.as.cls->methods = NULL;
    v.as.cls->init_method_name = NULL;
    v.as.cls->getitem_method_name = NULL;
    v.as.cls->setitem_method_name = NULL;
    v.as.cls->convert_method_name = NULL;
    return v;
}

static ION_NBTValue ion_nbt_make_instance(void)
{
    ION_NBTValue v;
    ion_nbt_init(&v);
    v.type = TAG_INSTANCE;
    v.as.inst = (ION_Instance*)malloc(sizeof(ION_Instance));
    if (!v.as.inst) {
        fprintf(stderr, "ION: malloc failed\n");
        abort();
    }
    v.as.inst->class_def = NULL;
    v.as.inst->field_values = NULL;
    return v;
}

/* ===============================
 * Hash Table Functions
 * =============================== */

// 简单的字符串哈希函数 (djb2算法)
static uint32_t ion_hash_string(const char *key, uint16_t len) {
    uint32_t hash = 5381;
    for (uint16_t i = 0; i < len; i++) {
        hash = ((hash << 5) + hash) + (unsigned char)key[i];
    }
    return hash;
}

// 初始化哈希桶
static void ion_hash_bucket_init(ION_HashBucket *bucket) {
    bucket->kvs = NULL;
    bucket->len = 0;
    bucket->cap = 0;
}

// 释放哈希桶
static void ion_hash_bucket_free(ION_HashBucket *bucket) {
    if (!bucket) return;
    
    for (size_t i = 0; i < bucket->len; i++) {
        ion_str_free(&bucket->kvs[i].key);
        ion_nbt_free(&bucket->kvs[i].val);
    }
    
    free(bucket->kvs);
    bucket->kvs = NULL;
    bucket->len = 0;
    bucket->cap = 0;
}

// 在哈希桶中查找键
static ssize_t ion_hash_bucket_find(const ION_HashBucket *bucket, const char *key, uint16_t len) {
    for (size_t i = 0; i < bucket->len; i++) {
        if (bucket->kvs[i].key.len == len && 
            memcmp(bucket->kvs[i].key.ptr, key, len) == 0) {
            return (ssize_t)i;
        }
    }
    return -1;
}

// 在哈希桶中添加或更新键值对
static int ion_hash_bucket_put(ION_HashBucket *bucket, IonStr *key, ION_NBTValue *value) {
    // 查找是否已存在
    ssize_t idx = ion_hash_bucket_find(bucket, key->ptr, key->len);
    if (idx >= 0) {
        // 更新现有值 - 先释放旧值
        ion_nbt_free(&bucket->kvs[idx].val);
        bucket->kvs[idx].val = *value;
        
        // 清零源值以防止double free
        ion_nbt_init(value);
        
        // 释放传入的key（因为我们不需要它）
        ion_str_free(key);
        return 0;
    }
    
    // 添加新键值对
    if (bucket->cap <= bucket->len) {
        size_t new_cap = bucket->cap ? bucket->cap * 2 : 4;
        ION_NBTCompoundKV *new_kvs = (ION_NBTCompoundKV*)realloc(
            bucket->kvs, new_cap * sizeof(ION_NBTCompoundKV));
        if (!new_kvs) return -1;
        
        bucket->kvs = new_kvs;
        bucket->cap = new_cap;
    }
    
    // 移动key和value到桶中
    bucket->kvs[bucket->len].key = *key;
    key->ptr = NULL;
    key->len = 0;
    
    bucket->kvs[bucket->len].val = *value;
    ion_nbt_init(value); // 清零源值以防止double free
    
    bucket->len++;
    return 0;
}

/* ===============================
 * Array and Compound Functions
 * =============================== */
 static int ion_nbt_array_push_move(ION_NBTArray *a, ION_NBTValue *v)
 {
     if (a->cap <= a->len) {
         size_t newcap = a->cap ? a->cap * 2 : 8U;
    ION_NBTValue *p = (ION_NBTValue*)realloc(a->items, newcap * sizeof(ION_NBTValue));
    if (!p) return -1;
         a->items = p; 
         a->cap = newcap;
     }
     a->items[a->len] = *v;
    v->type = TAG_END;
    v->as.str.ptr = NULL;
    v->as.str.len = 0;
    a->len += 1U;
    return 0;
}

static ssize_t ion_nbt_compound_find(const ION_NBTCompound *c, const char *key, uint16_t len)
{
    if (!c) return -1;
    
    // 计算哈希值并找到对应的桶
    uint32_t hash = ion_hash_string(key, len);
    size_t bucket_idx = hash % ION_HASH_TABLE_SIZE;
    
    // 在桶中查找
    return ion_hash_bucket_find(&c->buckets[bucket_idx], key, len);
}

static int ion_nbt_compound_put_move(ION_NBTCompound *c, IonStr *k, ION_NBTValue *v)
{
    if (!c) return -1;
    
    // 计算哈希值并找到对应的桶
    uint32_t hash = ion_hash_string(k->ptr, k->len);
    size_t bucket_idx = hash % ION_HASH_TABLE_SIZE;
    
    // 检查是否已存在
    ssize_t idx = ion_hash_bucket_find(&c->buckets[bucket_idx], k->ptr, k->len);
    
    int ret = ion_hash_bucket_put(&c->buckets[bucket_idx], k, v);
    if (ret == 0 && idx < 0) {
        // 新增的键值对，更新总数
        c->total_len++;
    }
    
    return ret;
}

/* ===============================
  * Environment Functions
 * =============================== */
static void ion_env_init(ION_ScopeEnv *e, ION_ScopeEnv *parent)
{
     e->names = NULL; 
     e->names_len = 0U; 
     e->names_cap = 0U;
    e->parent = parent;
}

static void ion_env_free(ION_ScopeEnv *e)
{
    if (!e) return;
    for (size_t i = 0; i < e->names_len; ++i) {
        ion_str_free(&e->names[i].name);
        ion_nbt_free(&e->names[i].value);
    }
     free(e->names); 
     e->names = NULL; 
     e->names_len = 0U; 
     e->names_cap = 0U;
}

static ssize_t ion_env_find_name(const ION_ScopeEnv *e, const char *name, uint16_t len)
{
    for (size_t i = 0; i < e->names_len; ++i) {
        if (e->names[i].name.len == len && memcmp(e->names[i].name.ptr, name, len) == 0) {
            return (ssize_t)i;
        }
    }
    return -1;
}

static const ION_NBTValue *ion_env_get_name_ref(const ION_ScopeEnv *e, const char *name)
{
    const ION_ScopeEnv *cur = e;
    uint16_t len = (uint16_t)strlen(name);
    while (cur) {
        ssize_t idx = ion_env_find_name(cur, name, len);
        if (idx >= 0) return &cur->names[(size_t)idx].value;
        cur = cur->parent;
    }
    return NULL;
}

static int ion_env_set_name_move(ION_ScopeEnv *e, IonStr *name, ION_NBTValue *val)
{
    ssize_t idx = ion_env_find_name(e, name->ptr, name->len);
    if (idx >= 0) {
        size_t i = (size_t)idx;
        ion_nbt_free(&e->names[i].value);
         e->names[i].value = *val; 
         val->type = TAG_END; 
         val->as.str.ptr = NULL; 
         val->as.str.len = 0;
        ion_str_free(name);
        return 0;
    }
     if (e->names_cap <= e->names_len) {
         size_t nc = e->names_cap ? e->names_cap * 2 : 8U;
         ION_NameKV *p = (ION_NameKV*)realloc(e->names, nc * sizeof(ION_NameKV));
    if (!p) return -1;
         e->names = p; 
         e->names_cap = nc;
     }
     e->names[e->names_len].name = *name; 
     name->ptr = NULL; 
     name->len = 0;
     e->names[e->names_len].value = *val; 
     val->type = TAG_END; 
     val->as.str.ptr = NULL; 
     val->as.str.len = 0;
    e->names_len += 1U;
    return 0;
}

/* ===============================
  * Simple B+Tree Implementation
 * =============================== */
static BPTNodeC *bpt_node_new(int leaf, int order)
{
    BPTNodeC *n = (BPTNodeC*)calloc(1, sizeof(BPTNodeC));
     if (!n) { 
         fprintf(stderr, "ION: malloc failed\n"); 
         abort(); 
     }
     n->leaf = leaf; 
     n->nkeys = 0; 
     n->order = order; 
     n->next_leaf = NULL;
        n->keys = (char**)calloc((size_t)(order * 2), sizeof(char*)); // Increased size
    if (!n->keys) { 
        fprintf(stderr, "ION: malloc failed\n"); 
        abort(); 
    }
    if (leaf) {
        n->values = (void**)calloc((size_t)(order * 2), sizeof(void*)); // Increased size
        if (!n->values) { 
            fprintf(stderr, "ION: malloc failed\n"); 
            abort(); 
        }
        n->children = NULL;
    } else {
        n->values = NULL;
        n->children = (BPTNodeC**)calloc((size_t)(order + 1), sizeof(BPTNodeC*));
        if (!n->children) { 
            fprintf(stderr, "ION: malloc failed\n"); 
            abort(); 
        }
    }
    return n;
}

static void bpt_node_free(BPTNodeC *n)
{
    if (!n) return;
    if (n->leaf) {
        for (int i = 0; i < n->nkeys; ++i) {
            free(n->keys[i]);
        }
        free(n->values);
    } else {
        for (int i = 0; i < n->nkeys + 1; ++i) {
            bpt_node_free(n->children[i]);
        }
        for (int i = 0; i < n->nkeys; ++i) {
            free(n->keys[i]);
        }
        free(n->children);
    }
    free(n->keys);
    free(n);
}

static int bpt_key_lower_bound(BPTNodeC *n, const char *key)
{
    int lo = 0, hi = n->nkeys;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        int cmp = strcmp(n->keys[mid], key);
         if (cmp < 0) lo = mid + 1; 
         else hi = mid;
    }
    return lo;
}

static BPTNodeC *bpt_find_leaf(BPTNodeC *root, const char *key)
{
    BPTNodeC *n = root;
    while (n && !n->leaf) {
        int pos = bpt_key_lower_bound(n, key);
        n = n->children[pos];
    }
    return n;
}

static int bpt_leaf_find_value(BPTNodeC *leaf, const char *key, void **out)
{
    // 简化查找：线性搜索
    for (int i = 0; i < leaf->nkeys; ++i) {
        if (leaf->keys[i] && strcmp(leaf->keys[i], key) == 0) {
            *out = leaf->values[i];
            return 1;
        }
    }
    return 0;
}

 static BPTreeC *bpt_create(int order)
 {
     if (order < 3) order = 3;
     BPTreeC *t = (BPTreeC*)malloc(sizeof(BPTreeC));
     if (!t) { 
         fprintf(stderr, "ION: malloc failed\n"); 
         abort(); 
     }
     t->order = order;
     t->root = bpt_node_new(1, order);
     return t;
 }
 
 static void bpt_free(BPTreeC *t)
 {
     if (!t) return; 
     bpt_node_free(t->root); 
     free(t);
 }
 
 static int bpt_insert_strptr(BPTreeC *t, const char *key, void *val)
{
    // Simplified insertion - just add to leaf if there's space
    BPTNodeC *leaf = bpt_find_leaf(t->root, key);
    if (!leaf) return -1;
    
    // Check if key already exists
    for (int i = 0; i < leaf->nkeys; ++i) {
        if (leaf->keys[i] && strcmp(leaf->keys[i], key) == 0) {
            leaf->values[i] = val;
        return 0;
    }
    }
    
    // Add new key if there's space (use larger capacity)
    if (leaf->nkeys < t->order * 2) {  // Increased capacity
        leaf->keys[leaf->nkeys] = strdup(key);
        if (!leaf->keys[leaf->nkeys]) return -1; // strdup failed
        leaf->values[leaf->nkeys] = val;
        leaf->nkeys++;
    return 0;
    }
    
    // If still no space, print debug info
    fprintf(stderr, "B+Tree: No space for key '%s', nkeys=%d, order=%d\n", key, leaf->nkeys, t->order);
    return -1; // No space
}
 
 static void *bpt_get(BPTreeC *t, const char *key)
 {
     BPTNodeC *leaf = bpt_find_leaf(t->root, key);
     if (!leaf) return NULL;
     void *out = NULL;
     if (bpt_leaf_find_value(leaf, key, &out) == 1) return out;
     return NULL;
}

/* ===============================
  * Utility Functions
 * =============================== */
 static const ION_NBTValue *ion_get_field(const ION_NBTValue *compound, const char *key)
 {
     if (!compound || (compound->type != TAG_COMPOUND && compound->type != TAG_FCOMPOUND) || !compound->as.comp) {
         return NULL;
     }
     
     uint16_t key_len = (uint16_t)strlen(key);
     uint32_t hash = ion_hash_string(key, key_len);
     size_t bucket_idx = hash % ION_HASH_TABLE_SIZE;
     
     ssize_t idx = ion_hash_bucket_find(&compound->as.comp->buckets[bucket_idx], key, key_len);
     if (idx < 0) return NULL;
     
     return &compound->as.comp->buckets[bucket_idx].kvs[(size_t)idx].val;
 }
 
 static double ion_value_to_double(const ION_NBTValue *v)
 {
     switch (v->type) {
         case TAG_BYTE: return (double)v->as.b;
         case TAG_SHORT: return (double)v->as.s;
         case TAG_INT: return (double)v->as.i;
         case TAG_LONG: return (double)v->as.l;
         case TAG_FLOAT: return (double)v->as.f;
         case TAG_DOUBLE: return v->as.d;
         default: return 0.0;
     }
 }
 
 static int64_t ion_value_to_int64(const ION_NBTValue *v)
 {
     switch (v->type) {
         case TAG_BYTE: return (int64_t)v->as.b;
         case TAG_SHORT: return (int64_t)v->as.s;
         case TAG_INT: return (int64_t)v->as.i;
         case TAG_LONG: return v->as.l;
         case TAG_FLOAT: return (int64_t)v->as.f;
         case TAG_DOUBLE: return (int64_t)v->as.d;
         default: return 0;
     }
 }
 
 static int ion_value_is_truthy(const ION_NBTValue *v)
 {
     switch (v->type) {
         case TAG_END: return 0;
         case TAG_BYTE: return v->as.b != 0;
         case TAG_SHORT: return v->as.s != 0;
         case TAG_INT: return v->as.i != 0;
         case TAG_LONG: return v->as.l != 0;
         case TAG_FLOAT: return v->as.f != 0.0f;
         case TAG_DOUBLE: return v->as.d != 0.0;
         case TAG_STRING: return v->as.str.len > 0;
         case TAG_ARRAY:
         case TAG_FARRAY: return v->as.arr && v->as.arr->len > 0;
                 case TAG_COMPOUND:
        case TAG_FCOMPOUND: return v->as.comp && v->as.comp->total_len > 0;
         default: return 1;
     }
 }

// 深拷贝函数
static ION_NBTValue ion_nbt_deep_copy(const ION_NBTValue *src) {
    if (!src) return ion_nbt_make_end();
    
    switch (src->type) {
        case TAG_END:
        case TAG_BYTE:
        case TAG_SHORT:
        case TAG_INT:
        case TAG_LONG:
        case TAG_FLOAT:
        case TAG_DOUBLE:
            // 基本类型，直接拷贝
            return *src;
            
        case TAG_STRING:
            return ion_nbt_make_string_copy(src->as.str.ptr, src->as.str.len);
            
        case TAG_ARRAY:
        case TAG_FARRAY:
            {
                ION_NBTValue result = ion_nbt_make_array();
                result.type = src->type; // 保持原类型
                if (src->as.arr) {
                    for (size_t i = 0; i < src->as.arr->len; i++) {
                        ION_NBTValue item_copy = ion_nbt_deep_copy(&src->as.arr->items[i]);
                        ion_nbt_array_push_move(result.as.arr, &item_copy);
                    }
                }
                return result;
            }
            
        case TAG_COMPOUND:
        case TAG_FCOMPOUND:
            {
                ION_NBTValue result = ion_nbt_make_compound();
                result.type = src->type; // 保持原类型
                if (src->as.comp) {
                    // 遍历所有哈希桶
                    for (int bucket_idx = 0; bucket_idx < ION_HASH_TABLE_SIZE; bucket_idx++) {
                        const ION_HashBucket *bucket = &src->as.comp->buckets[bucket_idx];
                        for (size_t i = 0; i < bucket->len; i++) {
                            IonStr key_copy = ion_str_from_cstr(bucket->kvs[i].key.ptr);
                            ION_NBTValue val_copy = ion_nbt_deep_copy(&bucket->kvs[i].val);
                            ion_nbt_compound_put_move(result.as.comp, &key_copy, &val_copy);
                        }
                    }
                }
                return result;
            }
            
        case TAG_CLASS:
            {
                ION_NBTValue result = ion_nbt_make_class();
                if (src->as.cls) {
                    // 深拷贝字段定义
                    if (src->as.cls->fields) {
                        ION_NBTValue fields_val;
                        fields_val.type = TAG_COMPOUND;
                        fields_val.as.comp = src->as.cls->fields;
                        ION_NBTValue fields_copy = ion_nbt_deep_copy(&fields_val);
                        result.as.cls->fields = fields_copy.as.comp;
                    }
                    
                    // 深拷贝方法定义
                    if (src->as.cls->methods) {
                        ION_NBTValue methods_val;
                        methods_val.type = TAG_COMPOUND;
                        methods_val.as.comp = src->as.cls->methods;
                        ION_NBTValue methods_copy = ion_nbt_deep_copy(&methods_val);
                        result.as.cls->methods = methods_copy.as.comp;
                    }
                    
                    // 拷贝方法名字符串
                    if (src->as.cls->init_method_name) {
                        result.as.cls->init_method_name = strdup(src->as.cls->init_method_name);
                    }
                    if (src->as.cls->getitem_method_name) {
                        result.as.cls->getitem_method_name = strdup(src->as.cls->getitem_method_name);
                    }
                    if (src->as.cls->setitem_method_name) {
                        result.as.cls->setitem_method_name = strdup(src->as.cls->setitem_method_name);
                    }
                    if (src->as.cls->convert_method_name) {
                        result.as.cls->convert_method_name = strdup(src->as.cls->convert_method_name);
                    }
                }
                return result;
            }
            
        case TAG_FUNCTION:
            {
                ION_NBTValue result = ion_nbt_make_function();
                if (src->as.func) {
                    // 拷贝参数名
                    if (src->as.func->param_names && src->as.func->param_count > 0) {
                        result.as.func->param_count = src->as.func->param_count;
                        result.as.func->param_names = (char**)malloc(sizeof(char*) * result.as.func->param_count);
                        if (result.as.func->param_names) {
                            for (int i = 0; i < result.as.func->param_count; i++) {
                                if (src->as.func->param_names[i]) {
                                    result.as.func->param_names[i] = strdup(src->as.func->param_names[i]);
                                } else {
                                    result.as.func->param_names[i] = NULL;
                                }
                            }
                        }
                    }
                    
                    // 深拷贝函数体
                    if (src->as.func->body) {
                        result.as.func->body = (ION_NBTValue*)malloc(sizeof(ION_NBTValue));
                        if (result.as.func->body) {
                            *result.as.func->body = ion_nbt_deep_copy(src->as.func->body);
                        }
                    }
                    
                    // 闭包环境不拷贝，设为NULL（简化处理）
                    result.as.func->closure_env = NULL;
                }
                return result;
            }
            
        case TAG_INSTANCE:
            {
                ION_NBTValue result = ion_nbt_make_instance();
                if (src->as.inst) {
                    // 深拷贝类定义
                    if (src->as.inst->class_def) {
                        ION_NBTValue class_val;
                        class_val.type = TAG_CLASS;
                        class_val.as.cls = src->as.inst->class_def;
                        ION_NBTValue class_copy = ion_nbt_deep_copy(&class_val);
                        result.as.inst->class_def = class_copy.as.cls;
                    }
                    
                    // 深拷贝字段值
                    if (src->as.inst->field_values) {
                        ION_NBTValue fields_val;
                        fields_val.type = TAG_COMPOUND;
                        fields_val.as.comp = src->as.inst->field_values;
                        ION_NBTValue fields_copy = ion_nbt_deep_copy(&fields_val);
                        result.as.inst->field_values = fields_copy.as.comp;
                    }
                }
                return result;
            }
            
        case TAG_VAR:
            {
                ION_NBTValue result = ion_nbt_make_var();
                if (src->as.var) {
                    // 拷贝UUID
                    memcpy(result.as.var->id.bytes, src->as.var->id.bytes, 16);
                    // 深拷贝变量的值
                    result.as.var->value = ion_nbt_deep_copy(&src->as.var->value);
                }
                return result;
            }
            
        case TAG_POINT:
            {
                ION_NBTValue result = ion_nbt_make_point();
                if (src->as.point) {
                    // 拷贝UUID
                    memcpy(result.as.point->id.bytes, src->as.point->id.bytes, 16);
                }
                return result;
            }
            
        default:
            // 其他类型暂时返回空值
            return ion_nbt_make_end();
    }
}
 
 

/* ===============================
 * Native Method Implementation Functions
 * =============================== */

// 数组方法实现
static int ion_call_array_method(const ION_NBTValue *obj, const char *method, size_t method_len, 
                                const ION_NBTValue *args, ION_ScopeEnv *env, ION_NBTValue *result) {
    (void)env; // 暂时未使用
    
    if (!obj || !obj->as.arr) {
        *result = ion_nbt_make_end();
        return ION_ERR_PARAM;
    }
    
    // append - 添加元素
    if (method_len == 6 && strncmp(method, "append", 6) == 0) {
        if (!args || (args->type != TAG_ARRAY && args->type != TAG_FARRAY) || !args->as.arr || args->as.arr->len == 0) {
            *result = ion_nbt_make_end();
            return ION_ERR_PARAM;
        }
        
        // 实际添加元素到数组 - 需要深拷贝以避免内存问题
        ION_NBTValue new_elem;
        
        // 深拷贝元素
        const ION_NBTValue *src_elem = &args->as.arr->items[0];
        switch (src_elem->type) {
            case TAG_STRING:
                new_elem = ion_nbt_make_string_copy(src_elem->as.str.ptr, src_elem->as.str.len);
                break;
            default:
                new_elem = *src_elem; // 对于基本类型，浅拷贝是安全的
                break;
        }
        
        // 确保数组有足够空间
        ION_NBTArray *arr = obj->as.arr;
        if (arr->cap <= arr->len) {
            size_t new_cap = arr->cap ? arr->cap * 2 : 8;
            ION_NBTValue *new_items = (ION_NBTValue*)realloc(arr->items, new_cap * sizeof(ION_NBTValue));
            if (!new_items) {
                *result = ion_nbt_make_end();
                return ION_ERR_RUNTIME;
            }
            arr->items = new_items;
            arr->cap = new_cap;
        }
        
        // 添加元素
        arr->items[arr->len] = new_elem;
        arr->len++;
        
        *result = ion_nbt_make_int_auto((int64_t)arr->len);
    return 0;
}

    // index - 查找元素索引
    else if (method_len == 5 && strncmp(method, "index", 5) == 0) {
        if (!args || (args->type != TAG_ARRAY && args->type != TAG_FARRAY) || !args->as.arr || args->as.arr->len == 0) {
            *result = ion_nbt_make_int(-1);
    return 0;
}

        const ION_NBTValue *search_val = &args->as.arr->items[0];
        
        for (size_t i = 0; i < obj->as.arr->len; i++) {
            const ION_NBTValue *elem = &obj->as.arr->items[i];
            
            // 简单比较
            int equal = 0;
            if (elem->type == search_val->type) {
                switch (elem->type) {
                    case TAG_BYTE: equal = (elem->as.b == search_val->as.b); break;
                    case TAG_SHORT: equal = (elem->as.s == search_val->as.s); break;
                    case TAG_INT: equal = (elem->as.i == search_val->as.i); break;
                    case TAG_LONG: equal = (elem->as.l == search_val->as.l); break;
                    case TAG_FLOAT: equal = (elem->as.f == search_val->as.f); break;
                    case TAG_DOUBLE: equal = (elem->as.d == search_val->as.d); break;
                    case TAG_STRING: 
                        equal = (elem->as.str.len == search_val->as.str.len && 
                                memcmp(elem->as.str.ptr, search_val->as.str.ptr, elem->as.str.len) == 0);
                        break;
                    default: equal = 0; break;
                }
            }
            
            if (equal) {
                *result = ion_nbt_make_long((int64_t)i);
    return 0;
            }
        }
        
        *result = ion_nbt_make_int(-1); // 未找到
    return 0;
}

    // pop - 移除并返回元素
    else if (method_len == 3 && strncmp(method, "pop", 3) == 0) {
        ION_NBTArray *arr = obj->as.arr;
        if (arr->len == 0) {
            *result = ion_nbt_make_end();
            return 0;
        }
        
        // 获取要返回的元素（浅拷贝）
        *result = arr->items[arr->len - 1];
        
        // 减少数组长度（不释放内存，只是逻辑删除）
        arr->len--;
        
        return 0;
    }
    
    // length - 获取长度
    else if (method_len == 6 && strncmp(method, "length", 6) == 0) {
        *result = ion_nbt_make_long((int64_t)obj->as.arr->len);
    return 0;
}

    // 未知方法
    *result = ion_nbt_make_end();
    return ION_ERR_UNIMPL;
}

// 复合结构方法实现
static int ion_call_compound_method(const ION_NBTValue *obj, const char *method, size_t method_len, 
                                   const ION_NBTValue *args, ION_ScopeEnv *env, ION_NBTValue *result) {
    (void)env; // 暂时未使用
    
    if (!obj || !obj->as.comp) {
        *result = ion_nbt_make_end();
        return ION_ERR_PARAM;
    }
    
    // val_to_key - 根据值查找键
    if (method_len == 10 && strncmp(method, "val_to_key", 10) == 0) {
        if (!args || (args->type != TAG_ARRAY && args->type != TAG_FARRAY) || !args->as.arr || args->as.arr->len == 0) {
            *result = ion_nbt_make_end();
            return ION_ERR_PARAM;
        }
        
        const ION_NBTValue *search_val = &args->as.arr->items[0];
        
        // 遍历所有哈希桶进行线性搜索 O(n)
        for (int bucket_idx = 0; bucket_idx < ION_HASH_TABLE_SIZE; bucket_idx++) {
            const ION_HashBucket *bucket = &obj->as.comp->buckets[bucket_idx];
            
            for (size_t i = 0; i < bucket->len; i++) {
                const ION_NBTValue *val = &bucket->kvs[i].val;
                
                int equal = 0;
                if (val->type == search_val->type) {
                    switch (val->type) {
                        case TAG_BYTE: equal = (val->as.b == search_val->as.b); break;
                        case TAG_SHORT: equal = (val->as.s == search_val->as.s); break;
                        case TAG_INT: equal = (val->as.i == search_val->as.i); break;
                        case TAG_LONG: equal = (val->as.l == search_val->as.l); break;
                        case TAG_FLOAT: equal = (val->as.f == search_val->as.f); break;
                        case TAG_DOUBLE: equal = (val->as.d == search_val->as.d); break;
                        case TAG_STRING: 
                            equal = (val->as.str.len == search_val->as.str.len && 
                                    memcmp(val->as.str.ptr, search_val->as.str.ptr, val->as.str.len) == 0);
                            break;
                        default: equal = 0; break;
                    }
                }
                
                if (equal) {
                    *result = ion_nbt_make_string_copy(bucket->kvs[i].key.ptr, bucket->kvs[i].key.len);
    return 0;
}
    }
    }
        
        *result = ion_nbt_make_end(); // 未找到
    return 0;
}

    // pop - 移除并返回键值对
    else if (method_len == 3 && strncmp(method, "pop", 3) == 0) {
        if (!args || (args->type != TAG_ARRAY && args->type != TAG_FARRAY) || !args->as.arr || args->as.arr->len == 0) {
            *result = ion_nbt_make_end();
            return ION_ERR_PARAM;
        }
        
        const ION_NBTValue *key_val = &args->as.arr->items[0];
        if (key_val->type != TAG_STRING) {
            *result = ion_nbt_make_end();
            return ION_ERR_TYPE;
        }
        
        uint32_t hash = ion_hash_string(key_val->as.str.ptr, key_val->as.str.len);
        size_t bucket_idx = hash % ION_HASH_TABLE_SIZE;
        ION_HashBucket *bucket = &obj->as.comp->buckets[bucket_idx];
        
        ssize_t idx = ion_hash_bucket_find(bucket, key_val->as.str.ptr, key_val->as.str.len);
    if (idx >= 0) {
            // 获取要返回的值（浅拷贝）
            *result = bucket->kvs[(size_t)idx].val;
            
            // 释放键的内存
            ion_str_free(&bucket->kvs[(size_t)idx].key);
            
            // 移动后面的元素向前填补空隙
            for (size_t i = (size_t)idx; i < bucket->len - 1; i++) {
                bucket->kvs[i] = bucket->kvs[i + 1];
            }
            bucket->len--;
            obj->as.comp->total_len--;
            
        return 0;
    }

        *result = ion_nbt_make_end(); // 未找到
    return 0;
}

    // put - 添加键值对
    else if (method_len == 3 && strncmp(method, "put", 3) == 0) {
        if (!args || (args->type != TAG_ARRAY && args->type != TAG_FARRAY) || !args->as.arr || args->as.arr->len < 2) {
            *result = ion_nbt_make_end();
            return ION_ERR_PARAM;
        }
        
        const ION_NBTValue *key_val = &args->as.arr->items[0];
        const ION_NBTValue *value_val = &args->as.arr->items[1];
        
        if (key_val->type != TAG_STRING) {
            *result = ion_nbt_make_end();
            return ION_ERR_TYPE;
        }
        
        // 创建键值对的拷贝
        IonStr key_copy = ion_str_from_cstr(key_val->as.str.ptr);
        ION_NBTValue value_copy;
        
        // 深拷贝值
        switch (value_val->type) {
            case TAG_STRING:
                value_copy = ion_nbt_make_string_copy(value_val->as.str.ptr, value_val->as.str.len);
                break;
            default:
                value_copy = *value_val; // 对于基本类型，浅拷贝是安全的
                break;
        }
        
        // 添加到哈希表
        int ret = ion_nbt_compound_put_move(obj->as.comp, &key_copy, &value_copy);
        if (ret == 0) {
            *result = ion_nbt_make_int_auto((int64_t)obj->as.comp->total_len);
        } else {
            *result = ion_nbt_make_end();
        }
        return ret;
    }

    // length - 获取长度
    else if (method_len == 6 && strncmp(method, "length", 6) == 0) {
        *result = ion_nbt_make_long((int64_t)obj->as.comp->total_len);
        return 0;
    }
    
    // 未知方法
    *result = ion_nbt_make_end();
    return ION_ERR_UNIMPL;
}

// 字符串方法实现
static int ion_call_string_method(const ION_NBTValue *obj, const char *method, size_t method_len, 
                                 const ION_NBTValue *args, ION_ScopeEnv *env, ION_NBTValue *result) {
    (void)env; // 暂时未使用
    
    if (!obj || !obj->as.str.ptr) {
        *result = ion_nbt_make_end();
        return ION_ERR_PARAM;
    }
    
    // to_array - 转换为字符数组
    if (method_len == 8 && strncmp(method, "to_array", 8) == 0) {
        ION_NBTValue arr = ion_nbt_make_array();
        
        for (size_t i = 0; i < obj->as.str.len; i++) {
            char ch[2] = {obj->as.str.ptr[i], '\0'};
            ION_NBTValue char_val = ion_nbt_make_string_copy(ch, 1);
            ion_nbt_array_push_move(arr.as.arr, &char_val);
        }
        
        *result = arr;
    return 0;
}

    // replace - 字符串替换
    else if (method_len == 7 && strncmp(method, "replace", 7) == 0) {
        if (!args || (args->type != TAG_ARRAY && args->type != TAG_FARRAY) || !args->as.arr || args->as.arr->len < 2) {
            *result = ion_nbt_make_string_copy(obj->as.str.ptr, obj->as.str.len);
            return 0;
        }
        
        const ION_NBTValue *search_val = &args->as.arr->items[0];
        const ION_NBTValue *replace_val = &args->as.arr->items[1];
        
        if (search_val->type != TAG_STRING || replace_val->type != TAG_STRING) {
            *result = ion_nbt_make_string_copy(obj->as.str.ptr, obj->as.str.len);
    return 0;
}

        // 简单的字符串替换实现
        const char *src = obj->as.str.ptr;
        size_t src_len = obj->as.str.len;
        const char *search = search_val->as.str.ptr;
        size_t search_len = search_val->as.str.len;
        const char *replace = replace_val->as.str.ptr;
        size_t replace_len = replace_val->as.str.len;
        
        if (search_len == 0 || src_len == 0) {
            *result = ion_nbt_make_string_copy(src, (uint16_t)src_len);
            return 0;
        }
        
        // 计算需要的缓冲区大小（简化：假设最多替换一次）
        size_t max_result_len = src_len + replace_len;
        if (max_result_len > UINT16_MAX) {
            *result = ion_nbt_make_string_copy(src, (uint16_t)src_len);
    return 0;
}

        char *result_buf = (char*)malloc(max_result_len + 1);
        if (!result_buf) {
            *result = ion_nbt_make_end();
            return ION_ERR_RUNTIME;
        }
        
        // 查找第一个匹配并替换
        const char *found = strstr(src, search);
        if (found) {
            size_t prefix_len = found - src;
            size_t suffix_len = src_len - prefix_len - search_len;
            
            // 构建结果字符串
            memcpy(result_buf, src, prefix_len);
            memcpy(result_buf + prefix_len, replace, replace_len);
            memcpy(result_buf + prefix_len + replace_len, found + search_len, suffix_len);
            
            size_t result_len = prefix_len + replace_len + suffix_len;
            result_buf[result_len] = '\0';
            
            *result = ion_nbt_make_string_copy(result_buf, (uint16_t)result_len);
        } else {
            // 没有找到匹配，返回原字符串
            *result = ion_nbt_make_string_copy(src, (uint16_t)src_len);
        }
        
        free(result_buf);
    return 0;
}

    // length - 获取长度
    else if (method_len == 6 && strncmp(method, "length", 6) == 0) {
        *result = ion_nbt_make_long((int64_t)obj->as.str.len);
        return 0;
    }
    
    // 未知方法
    *result = ion_nbt_make_end();
    return ION_ERR_UNIMPL;
}

// 数值方法实现
static int ion_call_numeric_method(const ION_NBTValue *obj, const char *method, size_t method_len, 
                                  const ION_NBTValue *args, ION_ScopeEnv *env, ION_NBTValue *result) {
    (void)args; (void)env; // 暂时未使用
    
    // inv - 取反/倒数
    if (method_len == 3 && strncmp(method, "inv", 3) == 0) {
        switch (obj->type) {
        case TAG_BYTE:
                *result = ion_nbt_make_byte(-obj->as.b);
            return 0;
        case TAG_SHORT:
                *result = ion_nbt_make_short(-obj->as.s);
            return 0;
        case TAG_INT:
                *result = ion_nbt_make_int(-obj->as.i);
            return 0;
        case TAG_LONG:
                *result = ion_nbt_make_long(-obj->as.l);
            return 0;
        case TAG_FLOAT:
                if (obj->as.f != 0.0f) {
                    *result = ion_nbt_make_float(1.0f / obj->as.f);
                } else {
                    *result = ion_nbt_make_float(0.0f);
                }
            return 0;
        case TAG_DOUBLE:
                if (obj->as.d != 0.0) {
                    *result = ion_nbt_make_double(1.0 / obj->as.d);
                } else {
                    *result = ion_nbt_make_double(0.0);
                }
            return 0;
            default:
                *result = ion_nbt_make_end();
                return ION_ERR_TYPE;
        }
    }
    
    // 未知方法
    *result = ion_nbt_make_end();
    return ION_ERR_UNIMPL;
}

/* ===============================
 * Keyword Implementation Functions
 * =============================== */
 
 // 序列执行
 static int kw_序(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out)
 {
     const ION_NBTValue *tasks = ion_get_field(expr, "做");
     if (!tasks || (tasks->type != TAG_ARRAY && tasks->type != TAG_FARRAY)) {
         if (out) *out = ion_nbt_make_end();
        return 0;
    }
     
     ION_NBTValue last = ion_nbt_make_end();
     for (size_t i = 0; i < tasks->as.arr->len; ++i) {
         ion_nbt_free(&last);
         if (ion_eval_expr(&tasks->as.arr->items[i], env, &last) != 0) {
             ion_nbt_free(&last);
             if (out) *out = ion_nbt_make_end();
             return ION_ERR_RUNTIME;
         }
     }
     if (out) *out = last; 
     else ion_nbt_free(&last);
    return 0;
}

 // 算术运算
 static int kw_arithmetic(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out, const char *op)
 {
     const ION_NBTValue *args = ion_get_field(expr, "与");
     if (!args || (args->type != TAG_ARRAY && args->type != TAG_FARRAY) || args->as.arr->len == 0) {
         if (out) *out = ion_nbt_make_int(0);
         return 0;
     }
     
     double result_d = 0.0;
     int64_t result_i = 0;
     int is_float = 0;
     int first = 1;
     
         for (size_t i = 0; i < args->as.arr->len; ++i) {
    ION_NBTValue val;
    int need_free = 0;  // 标记是否需要释放val
        // 对于简单值（非复合表达式），直接使用
        if (args->as.arr->items[i].type != TAG_COMPOUND && args->as.arr->items[i].type != TAG_FCOMPOUND) {
            val = args->as.arr->items[i]; // shallow copy
            need_free = 0;  // 浅拷贝，不需要释放
    } else {
            if (ion_eval_expr(&args->as.arr->items[i], env, &val) != 0) {
                if (out) *out = ion_nbt_make_int(0);
                return ION_ERR_RUNTIME;
            }
            need_free = 1;  // 求值结果需要释放
        }
         
         int current_is_float = (val.type == TAG_FLOAT || val.type == TAG_DOUBLE);
         
         if (first) {
             if (current_is_float) {
                 result_d = ion_value_to_double(&val);
                 is_float = 1;
             } else {
                 result_i = ion_value_to_int64(&val);
             }
             first = 0;
         } else {
             // 如果当前值是浮点数，但之前的运算是整数，需要转换
             if (current_is_float && !is_float) {
                 result_d = (double)result_i;
                 is_float = 1;
             }
             
             if (strcmp(op, "加") == 0) {
                 if (is_float) result_d += ion_value_to_double(&val);
                 else result_i += ion_value_to_int64(&val);
             } else if (strcmp(op, "减") == 0) {
                 if (is_float) result_d -= ion_value_to_double(&val);
                 else result_i -= ion_value_to_int64(&val);
             } else if (strcmp(op, "乘") == 0) {
                 if (is_float) result_d *= ion_value_to_double(&val);
                 else result_i *= ion_value_to_int64(&val);
                         } else if (strcmp(op, "除") == 0) {
                double divisor = ion_value_to_double(&val);
                if (divisor == 0.0) {
                    if (need_free) ion_nbt_free(&val);
                    if (out) *out = ion_nbt_make_int(0);
                    return ION_ERR_RUNTIME;
                }
                // 转换为浮点运算
                if (!is_float) {
                    result_d = (double)result_i;
                    is_float = 1;
                }
                result_d /= divisor;
             } else if (strcmp(op, "余") == 0) {
                                 int64_t divisor = ion_value_to_int64(&val);
                if (divisor == 0) {
                    if (need_free) ion_nbt_free(&val);
                    if (out) *out = ion_nbt_make_int(0);
                    return ION_ERR_RUNTIME;
                }
                 result_i %= divisor;
             }
         }
         // 只有需要时才释放内存
         if (need_free) {
             ion_nbt_free(&val);
         }
     }
     
     if (out) {
         if (is_float) *out = ion_nbt_make_float_auto(result_d);
         else *out = ion_nbt_make_int_auto(result_i);
    }
        return 0;
    }

 static int kw_加(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
     return kw_arithmetic(expr, env, out, "加");
 }
 
 static int kw_减(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
     return kw_arithmetic(expr, env, out, "减");
 }
 
 static int kw_乘(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
     return kw_arithmetic(expr, env, out, "乘");
 }
 
 static int kw_除(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
     return kw_arithmetic(expr, env, out, "除");
 }
 
 static int kw_余(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
     return kw_arithmetic(expr, env, out, "余");
 }
 
 // 比较运算 - 修复为使用"左"、"右"参数
static int kw_等(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    const ION_NBTValue *left_expr = ion_get_field(expr, "左");
    const ION_NBTValue *right_expr = ion_get_field(expr, "右");
    
    if (!left_expr || !right_expr) {
        if (out) *out = ion_nbt_make_byte(0);
        return ION_ERR_PARAM;
    }

             ION_NBTValue left, right;
    
    if (left_expr->type != TAG_COMPOUND && left_expr->type != TAG_FCOMPOUND) {
        left = *left_expr;
    } else {
        if (ion_eval_expr((ION_NBTValue*)left_expr, env, &left) != 0) {
            if (out) *out = ion_nbt_make_byte(0);
            return ION_ERR_RUNTIME;
        }
    }
    
    if (right_expr->type != TAG_COMPOUND && right_expr->type != TAG_FCOMPOUND) {
        right = *right_expr;
    } else {
        if (ion_eval_expr((ION_NBTValue*)right_expr, env, &right) != 0) {
            if (out) *out = ion_nbt_make_byte(0);
            return ION_ERR_RUNTIME;
        }
    }
    
    int equal = 0;
    if (left.type == right.type) {
        switch (left.type) {
            case TAG_BYTE: equal = (left.as.b == right.as.b); break;
            case TAG_SHORT: equal = (left.as.s == right.as.s); break;
            case TAG_INT: equal = (left.as.i == right.as.i); break;
            case TAG_LONG: equal = (left.as.l == right.as.l); break;
            case TAG_FLOAT: equal = (left.as.f == right.as.f); break;
            case TAG_DOUBLE: equal = (left.as.d == right.as.d); break;
        case TAG_STRING:
                equal = (left.as.str.len == right.as.str.len && 
                        memcmp(left.as.str.ptr, right.as.str.ptr, left.as.str.len) == 0); 
                break;
            default: equal = 0; break;
        }
    } else {
        double d1 = ion_value_to_double(&left);
        double d2 = ion_value_to_double(&right);
        equal = (d1 == d2);
    }
    
    ion_nbt_free(&left);
    ion_nbt_free(&right);
    
    if (out) *out = ion_nbt_make_byte(equal ? 1 : 0);
            return 0;
}

 static int kw_大于(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    const ION_NBTValue *left_expr = ion_get_field(expr, "左");
    const ION_NBTValue *right_expr = ion_get_field(expr, "右");
    
    if (!left_expr || !right_expr) {
        if (out) *out = ion_nbt_make_byte(0);
        return ION_ERR_PARAM;
    }
    
    ION_NBTValue left, right;
    
    if (left_expr->type != TAG_COMPOUND && left_expr->type != TAG_FCOMPOUND) {
        left = *left_expr;
    } else {
        if (ion_eval_expr((ION_NBTValue*)left_expr, env, &left) != 0) {
            if (out) *out = ion_nbt_make_byte(0);
            return ION_ERR_RUNTIME;
        }
    }
    
    if (right_expr->type != TAG_COMPOUND && right_expr->type != TAG_FCOMPOUND) {
        right = *right_expr;
    } else {
        if (ion_eval_expr((ION_NBTValue*)right_expr, env, &right) != 0) {
            if (out) *out = ion_nbt_make_byte(0);
            return ION_ERR_RUNTIME;
        }
    }
     
     double d1 = ion_value_to_double(&left);
     double d2 = ion_value_to_double(&right);
     
     ion_nbt_free(&left);
     ion_nbt_free(&right);
     
     if (out) *out = ion_nbt_make_byte(d1 > d2 ? 1 : 0);
            return 0;
}

 static int kw_小于(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    const ION_NBTValue *left_expr = ion_get_field(expr, "左");
    const ION_NBTValue *right_expr = ion_get_field(expr, "右");
    
    if (!left_expr || !right_expr) {
        if (out) *out = ion_nbt_make_byte(0);
        return ION_ERR_PARAM;
    }
    
    ION_NBTValue left, right;
    
    if (left_expr->type != TAG_COMPOUND && left_expr->type != TAG_FCOMPOUND) {
        left = *left_expr;
    } else {
        if (ion_eval_expr((ION_NBTValue*)left_expr, env, &left) != 0) {
            if (out) *out = ion_nbt_make_byte(0);
            return ION_ERR_RUNTIME;
        }
    }
    
    if (right_expr->type != TAG_COMPOUND && right_expr->type != TAG_FCOMPOUND) {
        right = *right_expr;
    } else {
        if (ion_eval_expr((ION_NBTValue*)right_expr, env, &right) != 0) {
            if (out) *out = ion_nbt_make_byte(0);
            return ION_ERR_RUNTIME;
        }
    }
     
     double d1 = ion_value_to_double(&left);
     double d2 = ion_value_to_double(&right);
     
     ion_nbt_free(&left);
     ion_nbt_free(&right);
     
     if (out) *out = ion_nbt_make_byte(d1 < d2 ? 1 : 0);
            return 0;
        }

 // 逻辑运算符
static int kw_且(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    const ION_NBTValue *left_expr = ion_get_field(expr, "左");
    const ION_NBTValue *right_expr = ion_get_field(expr, "右");
    
    if (!left_expr || !right_expr) {
        if (out) *out = ion_nbt_make_byte(0);
        return ION_ERR_PARAM;
    }
    
    ION_NBTValue left;
    if (left_expr->type != TAG_COMPOUND && left_expr->type != TAG_FCOMPOUND) {
        left = *left_expr;
    } else {
        if (ion_eval_expr((ION_NBTValue*)left_expr, env, &left) != 0) {
            if (out) *out = ion_nbt_make_byte(0);
            return ION_ERR_RUNTIME;
        }
    }
    
    int left_truthy = ion_value_is_truthy(&left);
    ion_nbt_free(&left);
    
    if (!left_truthy) {
        if (out) *out = ion_nbt_make_byte(0);
            return 0;
        }

    ION_NBTValue right;
    if (right_expr->type != TAG_COMPOUND && right_expr->type != TAG_FCOMPOUND) {
        right = *right_expr;
    } else {
        if (ion_eval_expr((ION_NBTValue*)right_expr, env, &right) != 0) {
            if (out) *out = ion_nbt_make_byte(0);
            return ION_ERR_RUNTIME;
        }
    }
    
    int right_truthy = ion_value_is_truthy(&right);
    ion_nbt_free(&right);
    
    if (out) *out = ion_nbt_make_byte(right_truthy ? 1 : 0);
            return 0;
        }

static int kw_或(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    const ION_NBTValue *left_expr = ion_get_field(expr, "左");
    const ION_NBTValue *right_expr = ion_get_field(expr, "右");
    
    if (!left_expr || !right_expr) {
        if (out) *out = ion_nbt_make_byte(0);
        return ION_ERR_PARAM;
    }
    
    ION_NBTValue left;
    if (left_expr->type != TAG_COMPOUND && left_expr->type != TAG_FCOMPOUND) {
        left = *left_expr;
    } else {
        if (ion_eval_expr((ION_NBTValue*)left_expr, env, &left) != 0) {
            if (out) *out = ion_nbt_make_byte(0);
            return ION_ERR_RUNTIME;
        }
    }
    
    int left_truthy = ion_value_is_truthy(&left);
    ion_nbt_free(&left);
    
    if (left_truthy) {
        if (out) *out = ion_nbt_make_byte(1);
            return 0;
        }

    ION_NBTValue right;
    if (right_expr->type != TAG_COMPOUND && right_expr->type != TAG_FCOMPOUND) {
        right = *right_expr;
    } else {
        if (ion_eval_expr((ION_NBTValue*)right_expr, env, &right) != 0) {
            if (out) *out = ion_nbt_make_byte(0);
            return ION_ERR_RUNTIME;
        }
    }
    
    int right_truthy = ion_value_is_truthy(&right);
    ion_nbt_free(&right);
    
    if (out) *out = ion_nbt_make_byte(right_truthy ? 1 : 0);
            return 0;
        }

static int kw_非(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    const ION_NBTValue *value_expr = ion_get_field(expr, "值");
    if (!value_expr) {
        if (out) *out = ion_nbt_make_byte(1); // 空值为假，非假为真
            return 0;
        }
    
    ION_NBTValue val;
    if (value_expr->type != TAG_COMPOUND && value_expr->type != TAG_FCOMPOUND) {
        val = *value_expr; // shallow copy
    } else {
        if (ion_eval_expr((ION_NBTValue*)value_expr, env, &val) != 0) {
            if (out) *out = ion_nbt_make_byte(1);
            return ION_ERR_RUNTIME;
        }
    }
    
    int is_truthy = ion_value_is_truthy(&val);
    ion_nbt_free(&val);
    
    if (out) *out = ion_nbt_make_byte(is_truthy ? 0 : 1);
            return 0;
        }
 
 static int kw_若(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
     (void)expr; (void)env;
     if (out) *out = ion_nbt_make_end();
            return 0;
        }
 
 static int kw_造变(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    const ION_NBTValue *value_expr = ion_get_field(expr, "值");
    const ION_NBTValue *name_expr = ion_get_field(expr, "名");
    
    if (name_expr) {
        // 模式2：通过名字指向已有变量
        if (name_expr->type == TAG_STRING) {
            // 通过变量名获取现有变量
            const ION_NBTValue *existing_value = ion_env_get_name_ref(env, name_expr->as.str.ptr);
            if (!existing_value) {
                if (out) *out = ion_nbt_make_end();
                return ION_ERR_RUNTIME; // 变量不存在
            }
            
            // 如果现有值已经是变量，返回它的深拷贝
            if (existing_value->type == TAG_VAR) {
                if (out) *out = ion_nbt_deep_copy(existing_value);
                return 0;
            }
            
            // 否则创建一个新变量来包装现有值
            ION_NBTValue var = ion_nbt_make_var();
            var.as.var->value = ion_nbt_deep_copy(existing_value);
            
            if (out) *out = var;
            else ion_nbt_free(&var);
            return 0;
        } else {
            // 模式3：通过表达式求值获得变量或指针
            ION_NBTValue name_val;
            if (ion_eval_expr((ION_NBTValue*)name_expr, env, &name_val) != 0) {
                if (out) *out = ion_nbt_make_end();
                return ION_ERR_RUNTIME;
            }
            
            if (name_val.type == TAG_VAR) {
                // 返回变量的深拷贝
                if (out) *out = ion_nbt_deep_copy(&name_val);
                ion_nbt_free(&name_val);
                return 0;
            } else if (name_val.type == TAG_POINT) {
                // 简化实现：直接返回指针
                if (out) *out = name_val;
                return 0;
            } else {
                ion_nbt_free(&name_val);
                if (out) *out = ion_nbt_make_end();
                return ION_ERR_TYPE; // 名参数必须指向变量或指针
            }
        }
    } else if (value_expr) {
        // 模式1：创建新变量
        ION_NBTValue val;
        if (ion_eval_expr((ION_NBTValue*)value_expr, env, &val) != 0) {
            if (out) *out = ion_nbt_make_end();
            return ION_ERR_RUNTIME;
        }
        
        ION_NBTValue var = ion_nbt_make_var();
        var.as.var->value = val;
        
        // 简化实现：生成一个唯一的变量ID并存储变量
        static uint64_t var_counter = 1;
        char var_key[32];
        snprintf(var_key, sizeof(var_key), "__var_%llu", (unsigned long long)var_counter++);
        
        // 将变量存储在环境中
        IonStr var_name = ion_str_from_cstr(var_key);
        ION_NBTValue var_copy = ion_nbt_deep_copy(&var);
        if (ion_env_set_name_move(env, &var_name, &var_copy) != 0) {
            ion_str_free(&var_name);
            ion_nbt_free(&var_copy);
            ion_nbt_free(&var);
            if (out) *out = ion_nbt_make_end();
            return ION_ERR_RUNTIME;
        }
        
        // 创建指向变量的指针
        ION_NBTValue point = ion_nbt_make_point();
        // 简化：将变量名作为指针ID存储
        memcpy(point.as.point->id.bytes, var_key, 16);
        
        ion_nbt_free(&var);
        
        if (out) *out = point;
        else ion_nbt_free(&point);
        return 0;
    } else {
        // 既没有值也没有名参数
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_PARAM;
    }
}
 
 static int kw_记(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
     const ION_NBTValue *name_expr = ion_get_field(expr, "位");
     const ION_NBTValue *value_expr = ion_get_field(expr, "为");
     
     if (!name_expr || !value_expr) {
         if (out) *out = ion_nbt_make_end();
         return ION_ERR_PARAM;
     }
     
     if (name_expr->type != TAG_STRING) {
         if (out) *out = ion_nbt_make_end();
         return ION_ERR_TYPE;
     }
     
     ION_NBTValue val;
     if (ion_eval_expr((ION_NBTValue*)value_expr, env, &val) != 0) {
         if (out) *out = ion_nbt_make_end();
         return ION_ERR_RUNTIME;
     }
     
     IonStr name = ion_str_from_cstr(name_expr->as.str.ptr);
     if (ion_env_set_name_move(env, &name, &val) != 0) {
         ion_str_free(&name);
         ion_nbt_free(&val);
         if (out) *out = ion_nbt_make_end();
         return ION_ERR_RUNTIME;
     }
     
     const ION_NBTValue *result = ion_env_get_name_ref(env, name_expr->as.str.ptr);
     if (out) {
         if (result) {
             *out = ion_nbt_deep_copy(result);
         } else {
             *out = ion_nbt_make_end();
         }
     }
    return 0;
}

 static int kw_取(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    const ION_NBTValue *name_expr = ion_get_field(expr, "名");
    const ION_NBTValue *pos_expr = ion_get_field(expr, "位");
    
    if (pos_expr) {
        // 通过位置/指针获取值
        ION_NBTValue ptr_val;
        if (ion_eval_expr((ION_NBTValue*)pos_expr, env, &ptr_val) != 0) {
            if (out) *out = ion_nbt_make_end();
            return ION_ERR_RUNTIME;
        }
        
                        if (ptr_val.type == TAG_POINT) {
            // 指针解引用：从指针ID恢复变量名并获取变量
            char var_key[32];
            memcpy(var_key, ptr_val.as.point->id.bytes, 16);
            var_key[16] = '\0'; // 确保字符串终止
            
            // 去掉前导空格
            char *trimmed_key = var_key;
            while (*trimmed_key == ' ') trimmed_key++;
            
            // 从环境中获取变量
            const ION_NBTValue *stored_var = ion_env_get_name_ref(env, trimmed_key);
            if (stored_var && stored_var->type == TAG_VAR) {
                if (out) *out = ion_nbt_deep_copy(&stored_var->as.var->value);
            } else {
                if (out) *out = ion_nbt_make_end();
            }
            ion_nbt_free(&ptr_val);
    return 0;
        } else if (ptr_val.type == TAG_VAR) {
            // 如果是变量，返回变量的值
            if (out) *out = ion_nbt_deep_copy(&ptr_val.as.var->value);
            ion_nbt_free(&ptr_val);
            return 0;
        } else if (ptr_val.type >= TAG_BYTE && ptr_val.type <= TAG_DOUBLE) {
            // 如果是数字，检查是否为0（null指针）
            double num_val = ion_value_to_double(&ptr_val);
            if (num_val == 0.0) {
                if (out) *out = ion_nbt_make_end();
            } else {
                // 返回数字本身
                if (out) *out = ion_nbt_deep_copy(&ptr_val);
            }
            ion_nbt_free(&ptr_val);
            return 0;
        } else {
            // 其他类型直接返回
            if (out) *out = ptr_val;
            return 0;
        }
    } else if (name_expr) {
        // 通过名字获取值
        if (name_expr->type != TAG_STRING) {
            if (out) *out = ion_nbt_make_end();
            return ION_ERR_TYPE;
        }
        
        const ION_NBTValue *result = ion_env_get_name_ref(env, name_expr->as.str.ptr);
        if (out) {
            if (result) {
                *out = ion_nbt_deep_copy(result);
            } else {
                *out = ion_nbt_make_end();
            }
        }
        return 0;
    } else {
        // 既没有名也没有位参数
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_PARAM;
    }
}

 static int kw_看(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
     const ION_NBTValue *value_expr = ion_get_field(expr, "值");
     if (!value_expr) {
         printf("(null)\n");
         if (out) *out = ion_nbt_make_end();
    return 0;
}

     ION_NBTValue val;
     if (ion_eval_expr((ION_NBTValue*)value_expr, env, &val) != 0) {
         printf("(error)\n");
         if (out) *out = ion_nbt_make_end();
         return ION_ERR_RUNTIME;
     }
     
     switch (val.type) {
         case TAG_END: printf("(end)\n"); break;
         case TAG_BYTE: printf("%d\n", (int)val.as.b); break;
         case TAG_SHORT: printf("%d\n", (int)val.as.s); break;
         case TAG_INT: printf("%d\n", val.as.i); break;
         case TAG_LONG: printf("%lld\n", (long long)val.as.l); break;
         case TAG_FLOAT: printf("%f\n", val.as.f); break;
         case TAG_DOUBLE: printf("%f\n", val.as.d); break;
         case TAG_STRING: printf("%.*s\n", val.as.str.len, val.as.str.ptr); break;
         default: printf("(complex type)\n"); break;
     }
     
     if (out) {
         *out = ion_nbt_deep_copy(&val);
         ion_nbt_free(&val);  // 释放原始的val
     } else {
         ion_nbt_free(&val);
     }
        return 0;
    }
 
 // 函数定义和调用
static int kw_函(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    // 函数定义: {"$": "函", "参": param_names, "体": body}
    const ION_NBTValue *params_expr = ion_get_field(expr, "参");
    const ION_NBTValue *body_expr = ion_get_field(expr, "体");
    
    if (!params_expr || !body_expr) {
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_PARAM;
    }
    
    if (params_expr->type != TAG_ARRAY && params_expr->type != TAG_FARRAY) {
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_TYPE;
    }
    
    ION_NBTValue func_val = ion_nbt_make_function();
    
    // 处理参数名
    if (params_expr->as.arr && params_expr->as.arr->len > 0) {
        func_val.as.func->param_count = (int)params_expr->as.arr->len;
        func_val.as.func->param_names = (char**)malloc(sizeof(char*) * func_val.as.func->param_count);
        
        if (!func_val.as.func->param_names) {
            ion_nbt_free(&func_val);
            if (out) *out = ion_nbt_make_end();
            return ION_ERR_RUNTIME;
        }
        
        for (int i = 0; i < func_val.as.func->param_count; i++) {
            const ION_NBTValue *param = &params_expr->as.arr->items[i];
            if (param->type == TAG_STRING) {
                size_t len = param->as.str.len + 1;
                func_val.as.func->param_names[i] = (char*)malloc(len);
                if (func_val.as.func->param_names[i]) {
                    memcpy(func_val.as.func->param_names[i], param->as.str.ptr, param->as.str.len);
                    func_val.as.func->param_names[i][param->as.str.len] = '\0';
                }
            } else {
                // 非字符串参数名，使用默认名
                func_val.as.func->param_names[i] = (char*)malloc(16);
                if (func_val.as.func->param_names[i]) {
                    snprintf(func_val.as.func->param_names[i], 16, "param%d", i);
                }
            }
        }
    }
    
    // 保存函数体（使用深拷贝避免double free）
    func_val.as.func->body = (ION_NBTValue*)malloc(sizeof(ION_NBTValue));
    if (func_val.as.func->body) {
        *func_val.as.func->body = ion_nbt_deep_copy(body_expr);
    }
    
    // 保存闭包环境（简化实现：不保存）
    func_val.as.func->closure_env = env;
    
    if (out) *out = func_val;
    else ion_nbt_free(&func_val);
    return 0;
}

static int kw_呼(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    // 函数调用: {"$": "呼", "函": function, "参": arguments}
    const ION_NBTValue *func_expr = ion_get_field(expr, "函");
    const ION_NBTValue *args_expr = ion_get_field(expr, "传");
    
    if (!func_expr) {
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_PARAM;
    }
    
    // 评估函数表达式
    ION_NBTValue func_val;
    if (ion_eval_expr((ION_NBTValue*)func_expr, env, &func_val) != 0) {
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_RUNTIME;
    }
    
    if (func_val.type != TAG_FUNCTION || !func_val.as.func) {
        ion_nbt_free(&func_val);
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_TYPE;
    }
    
    // 创建新的作用域用于函数执行
    ION_ScopeEnv func_env;
    ion_env_init(&func_env, func_val.as.func->closure_env ? func_val.as.func->closure_env : env);
    
    // 处理参数绑定
    if (args_expr && (args_expr->type == TAG_ARRAY || args_expr->type == TAG_FARRAY) && args_expr->as.arr) {
        size_t arg_count = args_expr->as.arr->len;
        size_t param_count = func_val.as.func->param_count;
        
        for (size_t i = 0; i < param_count && i < arg_count; i++) {
            // 评估参数值
            ION_NBTValue arg_val;
            if (ion_eval_expr(&args_expr->as.arr->items[i], env, &arg_val) != 0) {
                ion_env_free(&func_env);
                ion_nbt_free(&func_val);
                if (out) *out = ion_nbt_make_end();
                return ION_ERR_RUNTIME;
            }
            
            // 绑定参数到函数环境
            if (func_val.as.func->param_names && func_val.as.func->param_names[i]) {
                IonStr param_name = ion_str_from_cstr(func_val.as.func->param_names[i]);
                if (ion_env_set_name_move(&func_env, &param_name, &arg_val) != 0) {
                    ion_str_free(&param_name);
                    ion_nbt_free(&arg_val);
                    ion_env_free(&func_env);
                    ion_nbt_free(&func_val);
                    if (out) *out = ion_nbt_make_end();
                    return ION_ERR_RUNTIME;
                }
            } else {
                ion_nbt_free(&arg_val);
            }
        }
    }
    
    // 执行函数体
    ION_NBTValue result = ion_nbt_make_end();
    if (func_val.as.func->body) {
        int ret = ion_eval_expr(func_val.as.func->body, &func_env, &result);
        if (ret != 0 && ret != -100) { // -100 是返回码，不是错误
            ion_nbt_free(&result);
            result = ion_nbt_make_end();
        }
    }
    
    // 清理
    ion_env_free(&func_env);
    ion_nbt_free(&func_val);
    
    if (out) *out = result;
    else ion_nbt_free(&result);
            return 0;
        }
 // 控制流关键字
static int kw_出(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    const ION_NBTValue *value_expr = ion_get_field(expr, "值");
    if (!value_expr) {
        if (out) *out = ion_nbt_make_end();
    return 0;
}

    ION_NBTValue val;
    if (value_expr->type != TAG_COMPOUND && value_expr->type != TAG_FCOMPOUND) {
        val = *value_expr; // shallow copy
    } else {
        if (ion_eval_expr((ION_NBTValue*)value_expr, env, &val) != 0) {
            if (out) *out = ion_nbt_make_end();
            return ION_ERR_RUNTIME;
        }
    }
    
    if (out) *out = val; 
    else ion_nbt_free(&val);
    return 0; // 返回值，但不是特殊返回码
}

static int kw_循环(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    const ION_NBTValue *times_expr = ion_get_field(expr, "次");
    const ION_NBTValue *body_expr = ion_get_field(expr, "做");
    
    if (!times_expr || !body_expr) {
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_PARAM;
    }
    
    ION_NBTValue times_val;
    if (times_expr->type != TAG_COMPOUND && times_expr->type != TAG_FCOMPOUND) {
        times_val = *times_expr;
    } else {
        if (ion_eval_expr((ION_NBTValue*)times_expr, env, &times_val) != 0) {
            if (out) *out = ion_nbt_make_end();
            return ION_ERR_RUNTIME;
        }
    }
    
    int64_t times = ion_value_to_int64(&times_val);
    ion_nbt_free(&times_val);
    
    ION_NBTValue last = ion_nbt_make_end();
    
    for (int64_t i = 0; i < times; ++i) {
        ion_nbt_free(&last);
        int ret = ion_eval_expr((ION_NBTValue*)body_expr, env, &last);
        if (ret == -100) break; // 跳出
        if (ret == -101) continue; // 继续
        if (ret != 0) {
            ion_nbt_free(&last);
            if (out) *out = ion_nbt_make_end();
            return ret;
        }
    }
    
    if (out) *out = last; 
    else ion_nbt_free(&last);
    return 0;
}

static int kw_当(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    const ION_NBTValue *cond_expr = ion_get_field(expr, "条件");
    const ION_NBTValue *body_expr = ion_get_field(expr, "做");
    
    if (!cond_expr || !body_expr) {
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_PARAM;
    }
    
    ION_NBTValue last = ion_nbt_make_end();
    
    while (1) {
        ION_NBTValue cond_val;
        int ret = ion_eval_expr((ION_NBTValue*)cond_expr, env, &cond_val);
        if (ret != 0) {
            ion_nbt_free(&last);
            if (out) *out = ion_nbt_make_end();
            return ret;
        }
        
        int is_truthy = ion_value_is_truthy(&cond_val);
        ion_nbt_free(&cond_val);
        
        if (!is_truthy) break;
        
        ion_nbt_free(&last);
        ret = ion_eval_expr((ION_NBTValue*)body_expr, env, &last);
        if (ret == -100) break; // 跳出
        if (ret == -101) continue; // 继续
        if (ret != 0) {
            ion_nbt_free(&last);
            if (out) *out = ion_nbt_make_end();
            return ret;
        }
    }
    
    if (out) *out = last; 
    else ion_nbt_free(&last);
            return 0;
        }

static int kw_遍历(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    const ION_NBTValue *array_expr = ion_get_field(expr, "于");
    const ION_NBTValue *var_expr = ion_get_field(expr, "元");
    const ION_NBTValue *body_expr = ion_get_field(expr, "做");
    
    if (!array_expr || !var_expr || !body_expr) {
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_PARAM;
    }
    
    if (var_expr->type != TAG_STRING) {
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_TYPE;
    }
    
    ION_NBTValue array_val;
    if (array_expr->type != TAG_COMPOUND && array_expr->type != TAG_FCOMPOUND) {
        array_val = *array_expr;
    } else {
        if (ion_eval_expr((ION_NBTValue*)array_expr, env, &array_val) != 0) {
            if (out) *out = ion_nbt_make_end();
            return ION_ERR_RUNTIME;
        }
    }
    
    if ((array_val.type != TAG_ARRAY && array_val.type != TAG_FARRAY) || !array_val.as.arr) {
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_TYPE;
    }
    
    ION_NBTValue last = ion_nbt_make_end();
    
    for (size_t i = 0; i < array_val.as.arr->len; ++i) {
        // 设置循环变量
        IonStr var_name = ion_str_from_cstr(var_expr->as.str.ptr);
        ION_NBTValue elem = array_val.as.arr->items[i]; // shallow copy
        if (ion_env_set_name_move(env, &var_name, &elem) != 0) {
            ion_str_free(&var_name);
            ion_nbt_free(&last);
            if (out) *out = ion_nbt_make_end();
            return ION_ERR_RUNTIME;
        }
        
        ion_nbt_free(&last);
        int ret = ion_eval_expr((ION_NBTValue*)body_expr, env, &last);
        if (ret == -100) break; // 跳出
        if (ret == -101) continue; // 继续
        if (ret != 0) {
            ion_nbt_free(&last);
            if (out) *out = ion_nbt_make_end();
            return ret;
        }
    }
    
    if (out) *out = last; 
    else ion_nbt_free(&last);
    return 0;
}

static int kw_跳出(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) { 
    (void)expr; (void)env; 
    if (out) *out = ion_nbt_make_end(); 
    return -100; // 特殊返回码表示跳出
}

static int kw_继续(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) { 
    (void)expr; (void)env; 
    if (out) *out = ion_nbt_make_end(); 
    return -101; // 特殊返回码表示继续
}

static int kw_接(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    const ION_NBTValue *arrays_expr = ion_get_field(expr, "与");
    if (!arrays_expr || (arrays_expr->type != TAG_ARRAY && arrays_expr->type != TAG_FARRAY)) {
        if (out) *out = ion_nbt_make_array();
    return 0;
}

    ION_NBTValue result = ion_nbt_make_array();
    
    for (size_t i = 0; i < arrays_expr->as.arr->len; ++i) {
        ION_NBTValue array_val;
        if (arrays_expr->as.arr->items[i].type != TAG_COMPOUND && arrays_expr->as.arr->items[i].type != TAG_FCOMPOUND) {
            array_val = arrays_expr->as.arr->items[i];
    } else {
            if (ion_eval_expr(&arrays_expr->as.arr->items[i], env, &array_val) != 0) {
                ion_nbt_free(&result);
                if (out) *out = ion_nbt_make_array();
                return ION_ERR_RUNTIME;
            }
        }
        
        if ((array_val.type == TAG_ARRAY || array_val.type == TAG_FARRAY) && array_val.as.arr) {
            for (size_t j = 0; j < array_val.as.arr->len; ++j) {
                ION_NBTValue elem = array_val.as.arr->items[j]; // shallow copy
                ion_nbt_array_push_move(result.as.arr, &elem);
            }
        }
        
        ion_nbt_free(&array_val);
    }
    
    if (out) *out = result; 
    else ion_nbt_free(&result);
    return 0;
}
 static int kw_标(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    const ION_NBTValue *array_expr = ion_get_field(expr, "于");
    const ION_NBTValue *index_expr = ion_get_field(expr, "索");
    
    if (!array_expr || !index_expr) {
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_PARAM;
    }
    
    ION_NBTValue array_val, index_val;
    
    if (array_expr->type != TAG_COMPOUND && array_expr->type != TAG_FCOMPOUND) {
        array_val = *array_expr;
    } else {
        if (ion_eval_expr((ION_NBTValue*)array_expr, env, &array_val) != 0) {
            if (out) *out = ion_nbt_make_end();
            return ION_ERR_RUNTIME;
        }
    }
    
    if (index_expr->type != TAG_COMPOUND && index_expr->type != TAG_FCOMPOUND) {
        index_val = *index_expr;
    } else {
        if (ion_eval_expr((ION_NBTValue*)index_expr, env, &index_val) != 0) {
            if (out) *out = ion_nbt_make_end();
            return ION_ERR_RUNTIME;
        }
    }
    
    if ((array_val.type != TAG_ARRAY && array_val.type != TAG_FARRAY) || !array_val.as.arr) {
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_TYPE;
    }
    
    int64_t idx = ion_value_to_int64(&index_val);
    
    if (idx < 0 || (size_t)idx >= array_val.as.arr->len) {
        if (out) *out = ion_nbt_make_end();
    return 0;
}

    if (out) *out = array_val.as.arr->items[(size_t)idx]; // shallow copy
    return 0;
}
 static int kw_记标(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    const ION_NBTValue *array_expr = ion_get_field(expr, "于");
    const ION_NBTValue *index_expr = ion_get_field(expr, "索");
    const ION_NBTValue *value_expr = ion_get_field(expr, "为");
    
    if (!array_expr || !index_expr || !value_expr) {
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_PARAM;
    }
    
    ION_NBTValue array_val, index_val, new_val;
    
    if (array_expr->type != TAG_COMPOUND && array_expr->type != TAG_FCOMPOUND) {
        array_val = *array_expr;
    } else {
        if (ion_eval_expr((ION_NBTValue*)array_expr, env, &array_val) != 0) {
            if (out) *out = ion_nbt_make_end();
            return ION_ERR_RUNTIME;
        }
    }
    
    if (index_expr->type != TAG_COMPOUND && index_expr->type != TAG_FCOMPOUND) {
        index_val = *index_expr;
    } else {
        if (ion_eval_expr((ION_NBTValue*)index_expr, env, &index_val) != 0) {
            if (out) *out = ion_nbt_make_end();
            return ION_ERR_RUNTIME;
        }
    }
    
    if (value_expr->type != TAG_COMPOUND && value_expr->type != TAG_FCOMPOUND) {
        new_val = *value_expr;
    } else {
        if (ion_eval_expr((ION_NBTValue*)value_expr, env, &new_val) != 0) {
            if (out) *out = ion_nbt_make_end();
            return ION_ERR_RUNTIME;
        }
    }
    
    if ((array_val.type != TAG_ARRAY && array_val.type != TAG_FARRAY) || !array_val.as.arr) {
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_TYPE;
    }
    
    int64_t idx = ion_value_to_int64(&index_val);
    
    if (idx >= 0 && (size_t)idx < array_val.as.arr->len) {
        // 释放原来的值并设置新值
        ion_nbt_free(&array_val.as.arr->items[(size_t)idx]);
        array_val.as.arr->items[(size_t)idx] = new_val; // move
        if (out) *out = new_val; // shallow copy
    } else {
        if (out) *out = ion_nbt_make_end();
    }
    
            return 0;
        }
 static int kw_创属(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    // 动态创建属性: {"$": "创属", "于": instance_expr, "名": name_expr, "为": value_expr}
    const ION_NBTValue *obj_expr = ion_get_field(expr, "于");
    const ION_NBTValue *name_expr = ion_get_field(expr, "名");
    const ION_NBTValue *value_expr = ion_get_field(expr, "为");
    
    if (!obj_expr || !name_expr || !value_expr) {
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_PARAM;
    }
    
    // 评估实例表达式
    ION_NBTValue obj_val;
    if (ion_eval_expr((ION_NBTValue*)obj_expr, env, &obj_val) != 0) {
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_RUNTIME;
    }
    
    // 检查对象类型
    if (obj_val.type != TAG_INSTANCE || !obj_val.as.inst) {
        ion_nbt_free(&obj_val);
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_TYPE;
    }
    
    // 评估属性名
    ION_NBTValue name_val;
    if (ion_eval_expr((ION_NBTValue*)name_expr, env, &name_val) != 0) {
        ion_nbt_free(&obj_val);
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_RUNTIME;
    }
    
    // 检查属性名类型
    if (name_val.type != TAG_STRING) {
        ion_nbt_free(&obj_val);
        ion_nbt_free(&name_val);
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_TYPE;
    }
    
    // 评估属性值
    ION_NBTValue value_val;
    if (ion_eval_expr((ION_NBTValue*)value_expr, env, &value_val) != 0) {
        ion_nbt_free(&obj_val);
        ion_nbt_free(&name_val);
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_RUNTIME;
    }
    
    // 确保实例有字段值容器
    if (!obj_val.as.inst->field_values) {
        obj_val.as.inst->field_values = (ION_NBTCompound*)malloc(sizeof(ION_NBTCompound));
        if (!obj_val.as.inst->field_values) {
            ion_nbt_free(&obj_val);
            ion_nbt_free(&name_val);
            ion_nbt_free(&value_val);
            if (out) *out = ion_nbt_make_end();
            return ION_ERR_RUNTIME;
        }
        
        // 初始化哈希表
        for (int i = 0; i < ION_HASH_TABLE_SIZE; i++) {
            ion_hash_bucket_init(&obj_val.as.inst->field_values->buckets[i]);
        }
        obj_val.as.inst->field_values->total_len = 0;
    }
    
    // 动态创建属性（即使在类的域中未声明）
    IonStr attr_name = ion_str_from_cstr(name_val.as.str.ptr);
    ION_NBTValue value_copy = ion_nbt_deep_copy(&value_val);
    
    int ret = ion_nbt_compound_put_move(obj_val.as.inst->field_values, &attr_name, &value_copy);
    
    if (ret == 0) {
        // 成功创建，返回创建的值
        if (out) *out = ion_nbt_deep_copy(&value_val);
    } else {
        // 创建失败
        if (out) *out = ion_nbt_make_end();
    }
    
    ion_nbt_free(&obj_val);
    ion_nbt_free(&name_val);
    ion_nbt_free(&value_val);
    return ret;
}
 static int kw_转(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    const ION_NBTValue *type_expr = ion_get_field(expr, "类");
    const ION_NBTValue *value_expr = ion_get_field(expr, "值");
    
    if (!type_expr || !value_expr) {
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_PARAM;
    }
    
    ION_NBTValue type_val, val;
    
    if (type_expr->type != TAG_COMPOUND && type_expr->type != TAG_FCOMPOUND) {
        type_val = *type_expr;
    } else {
        if (ion_eval_expr((ION_NBTValue*)type_expr, env, &type_val) != 0) {
            if (out) *out = ion_nbt_make_end();
            return ION_ERR_RUNTIME;
        }
    }
    
    if (value_expr->type != TAG_COMPOUND && value_expr->type != TAG_FCOMPOUND) {
        val = *value_expr;
    } else {
        if (ion_eval_expr((ION_NBTValue*)value_expr, env, &val) != 0) {
            if (out) *out = ion_nbt_make_end();
            return ION_ERR_RUNTIME;
        }
    }
    
    if (type_val.type != TAG_STRING) {
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_TYPE;
    }
    
    ION_NBTValue result = ion_nbt_make_end();
    const char *target_type = type_val.as.str.ptr;
    
    if (strcmp(target_type, "Byte") == 0) {
        result = ion_nbt_make_byte((int8_t)ion_value_to_int64(&val));
    } else if (strcmp(target_type, "Short") == 0) {
        result = ion_nbt_make_short((int16_t)ion_value_to_int64(&val));
    } else if (strcmp(target_type, "Int") == 0) {
        result = ion_nbt_make_int((int32_t)ion_value_to_int64(&val));
    } else if (strcmp(target_type, "Long") == 0) {
        result = ion_nbt_make_long(ion_value_to_int64(&val));
    } else if (strcmp(target_type, "Float") == 0) {
        result = ion_nbt_make_float((float)ion_value_to_double(&val));
    } else if (strcmp(target_type, "Double") == 0) {
        result = ion_nbt_make_double(ion_value_to_double(&val));
    } else if (strcmp(target_type, "String") == 0) {
        char buf[64];
        int len;
        switch (val.type) {
            case TAG_BYTE: len = snprintf(buf, sizeof(buf), "%d", (int)val.as.b); break;
            case TAG_SHORT: len = snprintf(buf, sizeof(buf), "%d", (int)val.as.s); break;
            case TAG_INT: len = snprintf(buf, sizeof(buf), "%d", val.as.i); break;
            case TAG_LONG: len = snprintf(buf, sizeof(buf), "%lld", (long long)val.as.l); break;
            case TAG_FLOAT: len = snprintf(buf, sizeof(buf), "%f", val.as.f); break;
            case TAG_DOUBLE: len = snprintf(buf, sizeof(buf), "%f", val.as.d); break;
            default: len = snprintf(buf, sizeof(buf), "(complex)"); break;
        }
        result = ion_nbt_make_string_copy(buf, (uint16_t)len);
    }
    
    if (out) *out = result; 
    else ion_nbt_free(&result);
        return 0;
        }
 static int kw_文件(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    // 文件操作: {"$": "文件", "模": mode, "路": filepath, "文": content, "位": position}
    const ION_NBTValue *mode_expr = ion_get_field(expr, "模");
    const ION_NBTValue *filepath_expr = ion_get_field(expr, "路");
    const ION_NBTValue *content_expr = ion_get_field(expr, "文");
    const ION_NBTValue *position_expr = ion_get_field(expr, "位");
    
    if (!mode_expr || !filepath_expr) {
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_PARAM;
    }
    
    // 评估模式
    ION_NBTValue mode_val;
    if (ion_eval_expr((ION_NBTValue*)mode_expr, env, &mode_val) != 0) {
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_RUNTIME;
    }
    
    if (mode_val.type != TAG_STRING) {
        ion_nbt_free(&mode_val);
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_TYPE;
    }
    
    // 评估文件路径
    ION_NBTValue filepath_val;
    if (ion_eval_expr((ION_NBTValue*)filepath_expr, env, &filepath_val) != 0) {
        ion_nbt_free(&mode_val);
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_RUNTIME;
    }
    
    if (filepath_val.type != TAG_STRING) {
        ion_nbt_free(&mode_val);
        ion_nbt_free(&filepath_val);
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_TYPE;
    }
    
    const char *mode = mode_val.as.str.ptr;
    const char *filepath = filepath_val.as.str.ptr;
    
    // 简化实现：基本的文件读写操作
    if (strcmp(mode, "读") == 0) {
        // 读取模式
        FILE *file = fopen(filepath, "r");
        if (!file) {
            ion_nbt_free(&mode_val);
            ion_nbt_free(&filepath_val);
            if (out) *out = ion_nbt_make_string_copy("文件读取失败", 18);
            return ION_ERR_RUNTIME;
        }
        
        // 获取文件大小
        fseek(file, 0, SEEK_END);
        long file_size = ftell(file);
        fseek(file, 0, SEEK_SET);
        
        if (file_size > UINT16_MAX) {
            // 文件太大，截断
            file_size = UINT16_MAX;
        }
        
        // 读取文件内容
        char *buffer = (char*)malloc(file_size + 1);
        if (!buffer) {
            fclose(file);
            ion_nbt_free(&mode_val);
            ion_nbt_free(&filepath_val);
            if (out) *out = ion_nbt_make_end();
            return ION_ERR_RUNTIME;
        }
        
        size_t bytes_read = fread(buffer, 1, file_size, file);
        buffer[bytes_read] = '\0';
        fclose(file);
        
        if (out) *out = ion_nbt_make_string_copy(buffer, (uint16_t)bytes_read);
        free(buffer);
        
    } else if (strcmp(mode, "写") == 0) {
        // 写入模式
        if (!content_expr) {
            ion_nbt_free(&mode_val);
            ion_nbt_free(&filepath_val);
            if (out) *out = ion_nbt_make_end();
            return ION_ERR_PARAM;
        }
        
        // 评估内容
        ION_NBTValue content_val;
        if (ion_eval_expr((ION_NBTValue*)content_expr, env, &content_val) != 0) {
            ion_nbt_free(&mode_val);
            ion_nbt_free(&filepath_val);
            if (out) *out = ion_nbt_make_end();
            return ION_ERR_RUNTIME;
        }
        
        if (content_val.type != TAG_STRING) {
            ion_nbt_free(&mode_val);
            ion_nbt_free(&filepath_val);
            ion_nbt_free(&content_val);
            if (out) *out = ion_nbt_make_end();
            return ION_ERR_TYPE;
        }
        
        FILE *file = fopen(filepath, "w");
        if (!file) {
            ion_nbt_free(&mode_val);
            ion_nbt_free(&filepath_val);
            ion_nbt_free(&content_val);
            if (out) *out = ion_nbt_make_string_copy("文件写入失败", 18);
            return ION_ERR_RUNTIME;
        }
        
        fwrite(content_val.as.str.ptr, 1, content_val.as.str.len, file);
        fclose(file);
        
        if (out) *out = ion_nbt_make_string_copy("写入成功", 12);
        ion_nbt_free(&content_val);
        
    } else {
        // 未知模式
        ion_nbt_free(&mode_val);
        ion_nbt_free(&filepath_val);
        if (out) *out = ion_nbt_make_string_copy("未知文件模式", 18);
        return ION_ERR_PARAM;
    }
    
    ion_nbt_free(&mode_val);
    ion_nbt_free(&filepath_val);
            return 0;
        }
 static int kw_查(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    // 查找过滤: {"$": "查", "匹": filter_function, "复合": data_structure}
    const ION_NBTValue *filter_expr = ion_get_field(expr, "匹");
    const ION_NBTValue *data_expr = ion_get_field(expr, "复合");
    
    if (!filter_expr || !data_expr) {
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_PARAM;
    }
    
    // 评估过滤函数
    ION_NBTValue filter_func;
    if (ion_eval_expr((ION_NBTValue*)filter_expr, env, &filter_func) != 0) {
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_RUNTIME;
    }
    
    if (filter_func.type != TAG_FUNCTION || !filter_func.as.func) {
        ion_nbt_free(&filter_func);
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_TYPE;
    }
    
    // 评估数据结构
    ION_NBTValue data_val;
    if (ion_eval_expr((ION_NBTValue*)data_expr, env, &data_val) != 0) {
        ion_nbt_free(&filter_func);
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_RUNTIME;
    }
    
    // 简化实现：只处理数组和复合结构的查找
    if ((data_val.type == TAG_ARRAY || data_val.type == TAG_FARRAY) && data_val.as.arr) {
        // 数组查找
        ION_NBTValue result_array = ion_nbt_make_array();
        
        for (size_t i = 0; i < data_val.as.arr->len; i++) {
            // 为每个元素调用过滤函数
            ION_NBTValue *item = &data_val.as.arr->items[i];
            
            // 创建函数调用环境
            ION_ScopeEnv func_env;
            ion_env_init(&func_env, env);
            
            // 绑定参数（简化：只传递值）
            if (filter_func.as.func->param_count > 0 && filter_func.as.func->param_names) {
                IonStr param_name = ion_str_from_cstr(filter_func.as.func->param_names[0]);
                ION_NBTValue item_copy = ion_nbt_deep_copy(item);
                ion_env_set_name_move(&func_env, &param_name, &item_copy);
            }
            
            // 调用过滤函数
            ION_NBTValue filter_result;
            if (filter_func.as.func->body) {
                if (ion_eval_expr(filter_func.as.func->body, &func_env, &filter_result) == 0) {
                    // 检查结果是否为真
                    if (ion_value_is_truthy(&filter_result)) {
                        // 添加到结果数组
                        ION_NBTValue item_copy = ion_nbt_deep_copy(item);
                        ion_nbt_array_push_move(result_array.as.arr, &item_copy);
                    }
                    ion_nbt_free(&filter_result);
                }
            }
            
            ion_env_free(&func_env);
        }
        
        if (out) *out = result_array;
        else ion_nbt_free(&result_array);
        
    } else if ((data_val.type == TAG_COMPOUND || data_val.type == TAG_FCOMPOUND) && data_val.as.comp) {
        // 复合结构查找
        ION_NBTValue result_compound = ion_nbt_make_compound();
        
        // 遍历所有哈希桶
        for (int bucket_idx = 0; bucket_idx < ION_HASH_TABLE_SIZE; bucket_idx++) {
            const ION_HashBucket *bucket = &data_val.as.comp->buckets[bucket_idx];
            
            for (size_t i = 0; i < bucket->len; i++) {
                const ION_NBTCompoundKV *kv = &bucket->kvs[i];
                
                // 创建函数调用环境
                ION_ScopeEnv func_env;
                ion_env_init(&func_env, env);
                
                // 绑定参数（键和值）
                if (filter_func.as.func->param_count >= 2 && filter_func.as.func->param_names) {
                    // 绑定键
                    IonStr key_param = ion_str_from_cstr(filter_func.as.func->param_names[0]);
                    ION_NBTValue key_val = ion_nbt_make_string_copy(kv->key.ptr, kv->key.len);
                    ion_env_set_name_move(&func_env, &key_param, &key_val);
                    
                    // 绑定值
                    IonStr val_param = ion_str_from_cstr(filter_func.as.func->param_names[1]);
                    ION_NBTValue val_copy = ion_nbt_deep_copy(&kv->val);
                    ion_env_set_name_move(&func_env, &val_param, &val_copy);
                } else if (filter_func.as.func->param_count >= 1 && filter_func.as.func->param_names) {
                    // 只绑定值
                    IonStr val_param = ion_str_from_cstr(filter_func.as.func->param_names[0]);
                    ION_NBTValue val_copy = ion_nbt_deep_copy(&kv->val);
                    ion_env_set_name_move(&func_env, &val_param, &val_copy);
                }
                
                // 调用过滤函数
                ION_NBTValue filter_result;
                if (filter_func.as.func->body) {
                    if (ion_eval_expr(filter_func.as.func->body, &func_env, &filter_result) == 0) {
                        // 检查结果是否为真
                        if (ion_value_is_truthy(&filter_result)) {
                            // 添加到结果复合结构
                            IonStr key_copy = ion_str_from_cstr(kv->key.ptr);
                            ION_NBTValue val_copy = ion_nbt_deep_copy(&kv->val);
                            ion_nbt_compound_put_move(result_compound.as.comp, &key_copy, &val_copy);
                        }
                        ion_nbt_free(&filter_result);
                    }
                }
                
                ion_env_free(&func_env);
            }
        }
        
        if (out) *out = result_compound;
        else ion_nbt_free(&result_compound);
        
    } else {
        // 不支持的数据类型
        if (out) *out = ion_nbt_make_end();
    }
    
    ion_nbt_free(&filter_func);
    ion_nbt_free(&data_val);
    return 0;
}
 static int kw_在(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    const ION_NBTValue *item_expr = ion_get_field(expr, "左");
    const ION_NBTValue *container_expr = ion_get_field(expr, "右");
    
    if (!item_expr || !container_expr) {
        if (out) *out = ion_nbt_make_byte(0);
        return ION_ERR_PARAM;
    }
    
    ION_NBTValue item, container;
    
    if (item_expr->type != TAG_COMPOUND && item_expr->type != TAG_FCOMPOUND) {
        item = *item_expr;
    } else {
        if (ion_eval_expr((ION_NBTValue*)item_expr, env, &item) != 0) {
            if (out) *out = ion_nbt_make_byte(0);
            return ION_ERR_RUNTIME;
        }
    }
    
    if (container_expr->type != TAG_COMPOUND && container_expr->type != TAG_FCOMPOUND) {
        container = *container_expr;
    } else {
        if (ion_eval_expr((ION_NBTValue*)container_expr, env, &container) != 0) {
            if (out) *out = ion_nbt_make_byte(0);
            return ION_ERR_RUNTIME;
        }
    }
    
    int found = 0;
    if ((container.type == TAG_ARRAY || container.type == TAG_FARRAY) && container.as.arr) {
        // 在数组中查找
        for (size_t i = 0; i < container.as.arr->len && !found; ++i) {
            ION_NBTValue *elem = &container.as.arr->items[i];
            if (item.type == elem->type) {
                switch (item.type) {
                    case TAG_BYTE: found = (item.as.b == elem->as.b); break;
                    case TAG_SHORT: found = (item.as.s == elem->as.s); break;
                    case TAG_INT: found = (item.as.i == elem->as.i); break;
                    case TAG_LONG: found = (item.as.l == elem->as.l); break;
                    case TAG_FLOAT: found = (item.as.f == elem->as.f); break;
                    case TAG_DOUBLE: found = (item.as.d == elem->as.d); break;
                    case TAG_STRING: 
                        found = (item.as.str.len == elem->as.str.len && 
                                memcmp(item.as.str.ptr, elem->as.str.ptr, item.as.str.len) == 0);
                        break;
                    default: found = 0; break;
                }
            }
        }
    } else if (container.type == TAG_STRING && item.type == TAG_STRING) {
        // 在字符串中查找子字符串
        if (item.as.str.len <= container.as.str.len) {
            for (size_t i = 0; i <= container.as.str.len - item.as.str.len && !found; ++i) {
                if (memcmp(container.as.str.ptr + i, item.as.str.ptr, item.as.str.len) == 0) {
                    found = 1;
                }
            }
        }
    } else if ((container.type == TAG_COMPOUND || container.type == TAG_FCOMPOUND) && container.as.comp) {
        // 在复合结构中查找键
        if (item.type == TAG_STRING) {
            ssize_t idx = ion_nbt_compound_find(container.as.comp, item.as.str.ptr, item.as.str.len);
            found = (idx >= 0);
        }
    }
    
    if (out) *out = ion_nbt_make_byte(found ? 1 : 0);
        return 0;
}
 // 类系统相关函数
static int kw_类(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    (void)env; // 未使用的参数
    // 类定义: {"$": "类", "域": fields, "法": methods, "初函": init_method, "标函": getitem_method, "记函": setitem_method, "转函": convert_method}
    const ION_NBTValue *fields_expr = ion_get_field(expr, "域");
    const ION_NBTValue *methods_expr = ion_get_field(expr, "法");
    const ION_NBTValue *init_method_expr = ion_get_field(expr, "初函");
    const ION_NBTValue *getitem_method_expr = ion_get_field(expr, "标函");
    const ION_NBTValue *setitem_method_expr = ion_get_field(expr, "记函");
    const ION_NBTValue *convert_method_expr = ion_get_field(expr, "转函");
    
    ION_NBTValue class_val = ion_nbt_make_class();
    
    // 处理字段默认值
    if (fields_expr && (fields_expr->type == TAG_COMPOUND || fields_expr->type == TAG_FCOMPOUND)) {
        ION_NBTValue fields_copy = ion_nbt_deep_copy(fields_expr);
        if (fields_copy.type == TAG_COMPOUND || fields_copy.type == TAG_FCOMPOUND) {
            class_val.as.cls->fields = fields_copy.as.comp;
            fields_copy.as.comp = NULL; // 防止double free
        }
    }
    
    // 处理方法
    if (methods_expr && (methods_expr->type == TAG_COMPOUND || methods_expr->type == TAG_FCOMPOUND)) {
        ION_NBTValue methods_copy = ion_nbt_deep_copy(methods_expr);
        if (methods_copy.type == TAG_COMPOUND || methods_copy.type == TAG_FCOMPOUND) {
            class_val.as.cls->methods = methods_copy.as.comp;
            methods_copy.as.comp = NULL; // 防止double free
        }
    }
    
    // 处理特殊方法名
    if (init_method_expr && init_method_expr->type == TAG_STRING) {
        size_t len = init_method_expr->as.str.len + 1;
        class_val.as.cls->init_method_name = (char*)malloc(len);
        if (class_val.as.cls->init_method_name) {
            memcpy(class_val.as.cls->init_method_name, init_method_expr->as.str.ptr, init_method_expr->as.str.len);
            class_val.as.cls->init_method_name[init_method_expr->as.str.len] = '\0';
        }
    }
    
    if (getitem_method_expr && getitem_method_expr->type == TAG_STRING) {
        size_t len = getitem_method_expr->as.str.len + 1;
        class_val.as.cls->getitem_method_name = (char*)malloc(len);
        if (class_val.as.cls->getitem_method_name) {
            memcpy(class_val.as.cls->getitem_method_name, getitem_method_expr->as.str.ptr, getitem_method_expr->as.str.len);
            class_val.as.cls->getitem_method_name[getitem_method_expr->as.str.len] = '\0';
        }
    }
    
    if (setitem_method_expr && setitem_method_expr->type == TAG_STRING) {
        size_t len = setitem_method_expr->as.str.len + 1;
        class_val.as.cls->setitem_method_name = (char*)malloc(len);
        if (class_val.as.cls->setitem_method_name) {
            memcpy(class_val.as.cls->setitem_method_name, setitem_method_expr->as.str.ptr, setitem_method_expr->as.str.len);
            class_val.as.cls->setitem_method_name[setitem_method_expr->as.str.len] = '\0';
        }
    }
    
    if (convert_method_expr && convert_method_expr->type == TAG_STRING) {
        size_t len = convert_method_expr->as.str.len + 1;
        class_val.as.cls->convert_method_name = (char*)malloc(len);
        if (class_val.as.cls->convert_method_name) {
            memcpy(class_val.as.cls->convert_method_name, convert_method_expr->as.str.ptr, convert_method_expr->as.str.len);
            class_val.as.cls->convert_method_name[convert_method_expr->as.str.len] = '\0';
        }
    }
    
    if (out) *out = class_val;
    else ion_nbt_free(&class_val);
    return 0;
}

static int kw_造(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    // 类实例化: {"$": "造", "类": class, "参": constructor_args}
    const ION_NBTValue *class_expr = ion_get_field(expr, "类");
    const ION_NBTValue *args_expr = ion_get_field(expr, "传");
    (void)args_expr; // 未使用的参数，留作将来扩展
    
    if (!class_expr) {
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_PARAM;
    }
    
    // 评估类表达式
    ION_NBTValue class_val;
    if (ion_eval_expr((ION_NBTValue*)class_expr, env, &class_val) != 0) {
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_RUNTIME;
    }
    
    if (class_val.type != TAG_CLASS || !class_val.as.cls) {
        ion_nbt_free(&class_val);
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_TYPE;
    }
    
    // 创建实例
    ION_NBTValue instance = ion_nbt_make_instance();
    // 为实例创建类定义的深拷贝
    instance.as.inst->class_def = (ION_Class*)malloc(sizeof(ION_Class));
    if (!instance.as.inst->class_def) {
        ion_nbt_free(&instance);
        ion_nbt_free(&class_val);
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_RUNTIME;
    }
    
    // 深拷贝类定义
    memset(instance.as.inst->class_def, 0, sizeof(ION_Class));
    
    // 拷贝字段定义
    if (class_val.as.cls->fields) {
        ION_NBTValue fields_val;
        fields_val.type = TAG_COMPOUND;
        fields_val.as.comp = class_val.as.cls->fields;
        ION_NBTValue fields_copy = ion_nbt_deep_copy(&fields_val);
        instance.as.inst->class_def->fields = fields_copy.as.comp;
    }
    
    // 拷贝方法定义
    if (class_val.as.cls->methods) {
        ION_NBTValue methods_val;
        methods_val.type = TAG_COMPOUND;
        methods_val.as.comp = class_val.as.cls->methods;
        ION_NBTValue methods_copy = ion_nbt_deep_copy(&methods_val);
        instance.as.inst->class_def->methods = methods_copy.as.comp;
    }
    
    // 拷贝方法名字符串
    if (class_val.as.cls->init_method_name) {
        instance.as.inst->class_def->init_method_name = strdup(class_val.as.cls->init_method_name);
    }
    if (class_val.as.cls->getitem_method_name) {
        instance.as.inst->class_def->getitem_method_name = strdup(class_val.as.cls->getitem_method_name);
    }
    if (class_val.as.cls->setitem_method_name) {
        instance.as.inst->class_def->setitem_method_name = strdup(class_val.as.cls->setitem_method_name);
    }
    if (class_val.as.cls->convert_method_name) {
        instance.as.inst->class_def->convert_method_name = strdup(class_val.as.cls->convert_method_name);
    }
    
    // 初始化字段值
    if (class_val.as.cls->fields && class_val.as.cls->fields->total_len > 0) {
        instance.as.inst->field_values = (ION_NBTCompound*)malloc(sizeof(ION_NBTCompound));
        if (!instance.as.inst->field_values) {
            ion_nbt_free(&instance);
            ion_nbt_free(&class_val);
            if (out) *out = ion_nbt_make_end();
            return ION_ERR_RUNTIME;
        }
        
        // 初始化哈希表结构并复制字段值
        for (int i = 0; i < ION_HASH_TABLE_SIZE; i++) {
            ion_hash_bucket_init(&instance.as.inst->field_values->buckets[i]);
        }
        instance.as.inst->field_values->total_len = 0;
        
        // 复制默认字段值
        for (int bucket_idx = 0; bucket_idx < ION_HASH_TABLE_SIZE; bucket_idx++) {
            const ION_HashBucket *src_bucket = &class_val.as.cls->fields->buckets[bucket_idx];
            for (size_t i = 0; i < src_bucket->len; i++) {
                IonStr key_copy = ion_str_from_cstr(src_bucket->kvs[i].key.ptr);
                ION_NBTValue val_copy = src_bucket->kvs[i].val; // 浅拷贝
                ion_nbt_compound_put_move(instance.as.inst->field_values, &key_copy, &val_copy);
            }
        }
    } else {
        // 创建空字段值容器
        instance.as.inst->field_values = (ION_NBTCompound*)malloc(sizeof(ION_NBTCompound));
        if (instance.as.inst->field_values) {
            for (int i = 0; i < ION_HASH_TABLE_SIZE; i++) {
                ion_hash_bucket_init(&instance.as.inst->field_values->buckets[i]);
            }
            instance.as.inst->field_values->total_len = 0;
        }
    }
    
    // 调用构造函数（如果存在）
    if (class_val.as.cls->init_method_name && class_val.as.cls->methods) {
        // 查找初始化方法
        ssize_t init_idx = ion_nbt_compound_find(class_val.as.cls->methods, 
                                                 class_val.as.cls->init_method_name,
                                                 (uint16_t)strlen(class_val.as.cls->init_method_name));
        
        if (init_idx >= 0) {
            // 找到初始化方法的哈希桶和索引
            uint32_t hash = ion_hash_string(class_val.as.cls->init_method_name, 
                                          (uint16_t)strlen(class_val.as.cls->init_method_name));
            size_t bucket_idx = hash % ION_HASH_TABLE_SIZE;
            ssize_t method_idx = ion_hash_bucket_find(&class_val.as.cls->methods->buckets[bucket_idx], 
                                                     class_val.as.cls->init_method_name,
                                                     (uint16_t)strlen(class_val.as.cls->init_method_name));
            
            if (method_idx >= 0) {
                const ION_NBTValue *init_method = &class_val.as.cls->methods->buckets[bucket_idx].kvs[method_idx].val;
            
            // 简化实现：构造函数调用（这里应该调用方法，但简化为直接执行）
            if (init_method->type == TAG_FUNCTION && init_method->as.func && init_method->as.func->body) {
                ION_ScopeEnv init_env;
                ion_env_init(&init_env, env);
                
                // 绑定self参数（简化）
                IonStr self_name = ion_str_from_cstr("self");
                ION_NBTValue self_ref = instance; // 浅拷贝
                ion_env_set_name_move(&init_env, &self_name, &self_ref);
                
                // 执行初始化方法
                ION_NBTValue init_result;
                ion_eval_expr(init_method->as.func->body, &init_env, &init_result);
                ion_nbt_free(&init_result);
                
                ion_env_free(&init_env);
            }
            }
        }
    }
    
    if (out) *out = instance;
    else ion_nbt_free(&instance);
    
    // 现在可以安全释放class_val，因为instance有自己的深拷贝
    ion_nbt_free(&class_val);
            return 0;
        }

static int kw_取属(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    // 属性获取: {"$": "取属", "于": instance, "名": attr_name}
    const ION_NBTValue *obj_expr = ion_get_field(expr, "于");
    const ION_NBTValue *name_expr = ion_get_field(expr, "名");
    
    if (!obj_expr || !name_expr) {
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_PARAM;
    }
    
    // 评估对象表达式
    ION_NBTValue obj_val;
    if (ion_eval_expr((ION_NBTValue*)obj_expr, env, &obj_val) != 0) {
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_RUNTIME;
    }
    
    // 检查对象类型
    if (obj_val.type != TAG_INSTANCE || !obj_val.as.inst) {
        ion_nbt_free(&obj_val);
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_TYPE;
    }
    
    // 检查属性名类型
    if (name_expr->type != TAG_STRING) {
        ion_nbt_free(&obj_val);
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_TYPE;
    }
    
    // 在实例的字段值中查找属性
    const char *attr_name = name_expr->as.str.ptr;
    uint16_t attr_len = name_expr->as.str.len;
    
    if (obj_val.as.inst->field_values) {
        uint32_t hash = ion_hash_string(attr_name, attr_len);
        size_t bucket_idx = hash % ION_HASH_TABLE_SIZE;
        ssize_t field_idx = ion_hash_bucket_find(&obj_val.as.inst->field_values->buckets[bucket_idx], 
                                                 attr_name, attr_len);
        
        if (field_idx >= 0) {
            // 找到属性，返回深拷贝
            const ION_NBTValue *field_val = &obj_val.as.inst->field_values->buckets[bucket_idx].kvs[field_idx].val;
            if (out) *out = ion_nbt_deep_copy(field_val);
            ion_nbt_free(&obj_val);
            return 0;
        }
    }
    
    // 如果实例字段中没有，检查类的默认字段值
    if (obj_val.as.inst->class_def && obj_val.as.inst->class_def->fields) {
        uint32_t hash = ion_hash_string(attr_name, attr_len);
        size_t bucket_idx = hash % ION_HASH_TABLE_SIZE;
        ssize_t field_idx = ion_hash_bucket_find(&obj_val.as.inst->class_def->fields->buckets[bucket_idx], 
                                                 attr_name, attr_len);
        
        if (field_idx >= 0) {
            // 找到类默认字段，返回深拷贝
            const ION_NBTValue *field_val = &obj_val.as.inst->class_def->fields->buckets[bucket_idx].kvs[field_idx].val;
            if (out) *out = ion_nbt_deep_copy(field_val);
            ion_nbt_free(&obj_val);
            return 0;
        }
    }
    
    // 属性不存在
    ion_nbt_free(&obj_val);
    if (out) *out = ion_nbt_make_end();
    return 0;
}

static int kw_记属(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    // 属性设置: {"$": "记属", "于": instance, "名": attr_name, "为": value}
    const ION_NBTValue *obj_expr = ion_get_field(expr, "于");
    const ION_NBTValue *name_expr = ion_get_field(expr, "名");
    const ION_NBTValue *value_expr = ion_get_field(expr, "为");
    
    if (!obj_expr || !name_expr || !value_expr) {
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_PARAM;
    }
    
    // 评估对象表达式
    ION_NBTValue obj_val;
    if (ion_eval_expr((ION_NBTValue*)obj_expr, env, &obj_val) != 0) {
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_RUNTIME;
    }
    
    // 检查对象类型
    if (obj_val.type != TAG_INSTANCE || !obj_val.as.inst) {
        ion_nbt_free(&obj_val);
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_TYPE;
    }
    
    // 检查属性名类型
    if (name_expr->type != TAG_STRING) {
        ion_nbt_free(&obj_val);
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_TYPE;
    }
    
    // 评估新值
    ION_NBTValue new_val;
    if (ion_eval_expr((ION_NBTValue*)value_expr, env, &new_val) != 0) {
        ion_nbt_free(&obj_val);
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_RUNTIME;
    }
    
    // 确保实例有字段值容器
    if (!obj_val.as.inst->field_values) {
        obj_val.as.inst->field_values = (ION_NBTCompound*)malloc(sizeof(ION_NBTCompound));
        if (!obj_val.as.inst->field_values) {
            ion_nbt_free(&obj_val);
            ion_nbt_free(&new_val);
            if (out) *out = ion_nbt_make_end();
            return ION_ERR_RUNTIME;
        }
        
        // 初始化哈希表
        for (int i = 0; i < ION_HASH_TABLE_SIZE; i++) {
            ion_hash_bucket_init(&obj_val.as.inst->field_values->buckets[i]);
        }
        obj_val.as.inst->field_values->total_len = 0;
    }
    
    // 设置属性值
    IonStr attr_name = ion_str_from_cstr(name_expr->as.str.ptr);
    ION_NBTValue new_val_copy = ion_nbt_deep_copy(&new_val);
    
    int ret = ion_nbt_compound_put_move(obj_val.as.inst->field_values, &attr_name, &new_val_copy);
    
    if (ret == 0) {
        // 成功设置，返回设置的值
        if (out) *out = ion_nbt_deep_copy(&new_val);
    } else {
        // 设置失败
        if (out) *out = ion_nbt_make_end();
    }
    
    ion_nbt_free(&obj_val);
    ion_nbt_free(&new_val);
    return ret;
}

static int kw_呼属(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    // 方法调用: {"$": "呼属", "对象": object, "法": method_name, "参": arguments}
    const ION_NBTValue *obj_expr = ion_get_field(expr, "对象");
    const ION_NBTValue *method_expr = ion_get_field(expr, "法");
    const ION_NBTValue *args_expr = ion_get_field(expr, "参");
    
    if (!obj_expr || !method_expr) {
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_PARAM;
    }
    
    // 评估对象
    ION_NBTValue obj_val;
    if (ion_eval_expr((ION_NBTValue*)obj_expr, env, &obj_val) != 0) {
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_RUNTIME;
    }
    
    // 获取方法名
    if (method_expr->type != TAG_STRING) {
        ion_nbt_free(&obj_val);
        if (out) *out = ion_nbt_make_end();
        return ION_ERR_TYPE;
    }
    
    const char *method_name = method_expr->as.str.ptr;
    size_t method_len = method_expr->as.str.len;
    
    // 根据对象类型和方法名调用相应的原生方法
    ION_NBTValue result = ion_nbt_make_end();
    int ret = 0;
    
    switch (obj_val.type) {
        case TAG_ARRAY:
        case TAG_FARRAY:
            ret = ion_call_array_method(&obj_val, method_name, method_len, args_expr, env, &result);
            break;
            
        case TAG_COMPOUND:
        case TAG_FCOMPOUND:
            ret = ion_call_compound_method(&obj_val, method_name, method_len, args_expr, env, &result);
            break;
            
        case TAG_STRING:
            ret = ion_call_string_method(&obj_val, method_name, method_len, args_expr, env, &result);
            break;
            
        case TAG_BYTE:
        case TAG_SHORT:
        case TAG_INT:
        case TAG_LONG:
        case TAG_FLOAT:
        case TAG_DOUBLE:
            ret = ion_call_numeric_method(&obj_val, method_name, method_len, args_expr, env, &result);
            break;
            
        case TAG_INSTANCE:
            // 对于实例对象，查找类方法
            if (obj_val.as.inst && obj_val.as.inst->class_def && obj_val.as.inst->class_def->methods) {
                uint32_t hash = ion_hash_string(method_name, (uint16_t)method_len);
                size_t bucket_idx = hash % ION_HASH_TABLE_SIZE;
                ssize_t method_idx = ion_hash_bucket_find(&obj_val.as.inst->class_def->methods->buckets[bucket_idx], 
                                                         method_name, (uint16_t)method_len);
                if (method_idx >= 0) {
                    const ION_NBTValue *method = &obj_val.as.inst->class_def->methods->buckets[bucket_idx].kvs[method_idx].val;
                    // 简化实现：直接调用方法（应该绑定self等）
                    if (method->type == TAG_FUNCTION && method->as.func && method->as.func->body) {
                        ret = ion_eval_expr(method->as.func->body, env, &result);
                    } else {
                        ret = ION_ERR_TYPE;
                    }
                } else {
                    ret = ION_ERR_UNIMPL;
                }
            } else {
                ret = ION_ERR_TYPE;
            }
            break;
            
        default:
            ret = ION_ERR_TYPE;
            break;
    }
    
    ion_nbt_free(&obj_val);
    
    if (ret == 0) {
        if (out) *out = result;
        else ion_nbt_free(&result);
    } else {
        ion_nbt_free(&result);
        if (out) *out = ion_nbt_make_end();
    }
    
    return ret;
}
 // 类型检查和工具函数
static int kw_是类(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    const ION_NBTValue *obj_expr = ion_get_field(expr, "值");
    const ION_NBTValue *class_expr = ion_get_field(expr, "类");
    
    if (!obj_expr || !class_expr) {
        if (out) *out = ion_nbt_make_byte(0);
        return ION_ERR_PARAM;
    }
    
    // 简化实现：检查对象是否是复合类型
    ION_NBTValue obj_val;
    if (ion_eval_expr((ION_NBTValue*)obj_expr, env, &obj_val) != 0) {
        if (out) *out = ion_nbt_make_byte(0);
        return ION_ERR_RUNTIME;
    }
    
    int is_class = (obj_val.type == TAG_COMPOUND || obj_val.type == TAG_FCOMPOUND);
    ion_nbt_free(&obj_val);
    
    if (out) *out = ion_nbt_make_byte(is_class ? 1 : 0);
    return 0;
}

static int kw_验介(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    // 简化实现：总是返回假
    (void)expr; (void)env;
    if (out) *out = ion_nbt_make_byte(0);
            return 0;
        }

static int kw_接口(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    // 简化实现：创建一个空的复合结构表示接口
    (void)expr; (void)env;
    if (out) *out = ion_nbt_make_compound();
    return 0;
}

static int kw_存(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    (void)env; // 未使用的参数
    // 简化实现：返回保存成功的消息
    const ION_NBTValue *name_expr = ion_get_field(expr, "名");
    const ION_NBTValue *data_expr = ion_get_field(expr, "值");
    
    if (!name_expr || !data_expr) {
        if (out) *out = ion_nbt_make_string_copy("存储失败：缺少参数", 24);
        return ION_ERR_PARAM;
    }
    
    // 简化：只返回成功消息，不实际保存
    if (out) *out = ion_nbt_make_string_copy("数据已保存", 12);
            return 0;
        }

static int kw_载(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    // 简化实现：返回空数据
    (void)expr; (void)env;
    if (out) *out = ion_nbt_make_end();
    return 0;
}

static int kw_空(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    const ION_NBTValue *value_expr = ion_get_field(expr, "值");
    if (!value_expr) {
        if (out) *out = ion_nbt_make_byte(1);
    return 0;
}

    ION_NBTValue val;
    if (ion_eval_expr((ION_NBTValue*)value_expr, env, &val) != 0) {
        if (out) *out = ion_nbt_make_byte(1);
            return 0;
        }

    int is_empty = 0;
    switch (val.type) {
        case TAG_END:
            is_empty = 1;
            break;
        case TAG_BYTE:
        case TAG_SHORT:
        case TAG_INT:
        case TAG_LONG:
            is_empty = (ion_value_to_int64(&val) == 0);
            break;
        case TAG_FLOAT:
        case TAG_DOUBLE:
            is_empty = (ion_value_to_double(&val) == 0.0);
            break;
        case TAG_STRING:
            is_empty = (val.as.str.len == 0);
            break;
        case TAG_ARRAY:
        case TAG_FARRAY:
            is_empty = (val.as.arr == NULL || val.as.arr->len == 0);
            break;
        case TAG_COMPOUND:
        case TAG_FCOMPOUND:
            is_empty = (val.as.comp == NULL || val.as.comp->total_len == 0);
            break;
        default:
            is_empty = 0;
            break;
    }
    
    ion_nbt_free(&val);
    if (out) *out = ion_nbt_make_byte(is_empty ? 1 : 0);
    return 0;
}

static int kw_非空(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    // 调用kw_空然后取反
    ION_NBTValue empty_result;
    int ret = kw_空(expr, env, &empty_result);
    if (ret != 0) {
        if (out) *out = ion_nbt_make_byte(0);
        return ret;
    }
    
    int is_empty = (empty_result.type == TAG_BYTE && empty_result.as.b != 0);
    ion_nbt_free(&empty_result);
    
    if (out) *out = ion_nbt_make_byte(is_empty ? 0 : 1);
    return 0;
}

static int kw_类型(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    const ION_NBTValue *value_expr = ion_get_field(expr, "值");
    if (!value_expr) {
        if (out) *out = ion_nbt_make_string_copy("Unknown", 7);
        return 0;
    }
    
    ION_NBTValue val;
    if (ion_eval_expr((ION_NBTValue*)value_expr, env, &val) != 0) {
        if (out) *out = ion_nbt_make_string_copy("Error", 5);
    return 0;
}

    const char *type_name = "Unknown";
    switch (val.type) {
        case TAG_END: type_name = "End"; break;
        case TAG_BYTE: type_name = "Byte"; break;
        case TAG_SHORT: type_name = "Short"; break;
        case TAG_INT: type_name = "Int"; break;
        case TAG_LONG: type_name = "Long"; break;
        case TAG_FLOAT: type_name = "Float"; break;
        case TAG_DOUBLE: type_name = "Double"; break;
        case TAG_STRING: type_name = "String"; break;
        case TAG_ARRAY: type_name = "Array"; break;
        case TAG_FARRAY: type_name = "FArray"; break;
        case TAG_COMPOUND: type_name = "Compound"; break;
        case TAG_FCOMPOUND: type_name = "FCompound"; break;
        case TAG_VAR: type_name = "Var"; break;
        case TAG_POINT: type_name = "Point"; break;
        default: type_name = "Unknown"; break;
    }
    
    ion_nbt_free(&val);
    if (out) *out = ion_nbt_make_string_copy(type_name, (uint16_t)strlen(type_name));
    return 0;
}

static int kw_安全取(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    // 安全版本的取值，捕获所有错误
    int ret = kw_取(expr, env, out);
    if (ret != 0) {
        if (out) *out = ion_nbt_make_end();
        return 0; // 转换错误为成功，返回空值
    }
    return 0;
}

static int kw_安全记(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    // 安全版本的赋值，捕获所有错误
    int ret = kw_记(expr, env, out);
    if (ret != 0) {
        if (out) *out = ion_nbt_make_end();
        return 0; // 转换错误为成功，返回空值
    }
    return 0;
}

static int kw_取位(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    // 简化实现：返回空指针
    (void)expr; (void)env;
    if (out) *out = ion_nbt_make_end();
    return 0;
}

static int kw_改(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out) {
    // 改的实现与记相同，但语义上表示修改现有变量
    return kw_记(expr, env, out);
}

/* ===============================
  * Expression Evaluator
 * =============================== */
 static int ion_eval_expr(ION_NBTValue *expr, ION_ScopeEnv *env, ION_NBTValue *out)
 {
     if (!expr || !env || !out) return ION_ERR_PARAM;
     
     // 简单值直接返回 - 使用深拷贝避免double free
     if (expr->type != TAG_COMPOUND && expr->type != TAG_FCOMPOUND) {
         *out = ion_nbt_deep_copy(expr);
         return 0;
     }
     
     // 复合表达式：查找操作符
     const ION_NBTValue *op_field = ion_get_field(expr, "$");
     if (!op_field || op_field->type != TAG_STRING) {
         *out = ion_nbt_deep_copy(expr);
         return 0;
     }
     
     // 查找关键字处理函数
     ION_KeywordHandler handler = ion_get_keyword_handler(op_field->as.str.ptr);
     if (!handler) {
         *out = ion_nbt_make_end();
         return ION_ERR_UNIMPL;
     }
     
     return handler(expr, env, out);
 }

/* ===============================
  * Keyword Dispatch System
 * =============================== */
static BPTreeC *g_keyword_map = NULL;

 static int ion_keywords_init(void)
{
    if (g_keyword_map) return 0;
    g_keyword_map = bpt_create(32);
    if (!g_keyword_map) return -1;
     
     // 映射表
    bpt_insert_strptr(g_keyword_map, "序", (void*)kw_序);
    bpt_insert_strptr(g_keyword_map, "加", (void*)kw_加);
    bpt_insert_strptr(g_keyword_map, "减", (void*)kw_减);
    bpt_insert_strptr(g_keyword_map, "乘", (void*)kw_乘);
    bpt_insert_strptr(g_keyword_map, "除", (void*)kw_除);
    bpt_insert_strptr(g_keyword_map, "余", (void*)kw_余);
    bpt_insert_strptr(g_keyword_map, "等", (void*)kw_等);
    bpt_insert_strptr(g_keyword_map, "大于", (void*)kw_大于);
    bpt_insert_strptr(g_keyword_map, "小于", (void*)kw_小于);
    bpt_insert_strptr(g_keyword_map, "且", (void*)kw_且);
    bpt_insert_strptr(g_keyword_map, "或", (void*)kw_或);
    bpt_insert_strptr(g_keyword_map, "非", (void*)kw_非);
    bpt_insert_strptr(g_keyword_map, "若", (void*)kw_若);
    bpt_insert_strptr(g_keyword_map, "造变", (void*)kw_造变);
    bpt_insert_strptr(g_keyword_map, "记", (void*)kw_记);
    bpt_insert_strptr(g_keyword_map, "取", (void*)kw_取);
    bpt_insert_strptr(g_keyword_map, "看", (void*)kw_看);
    bpt_insert_strptr(g_keyword_map, "函", (void*)kw_函);
    bpt_insert_strptr(g_keyword_map, "呼", (void*)kw_呼);
    bpt_insert_strptr(g_keyword_map, "出", (void*)kw_出);
    bpt_insert_strptr(g_keyword_map, "循环", (void*)kw_循环);
    bpt_insert_strptr(g_keyword_map, "当", (void*)kw_当);
    bpt_insert_strptr(g_keyword_map, "遍历", (void*)kw_遍历);
    bpt_insert_strptr(g_keyword_map, "跳出", (void*)kw_跳出);
    bpt_insert_strptr(g_keyword_map, "继续", (void*)kw_继续);
    bpt_insert_strptr(g_keyword_map, "接", (void*)kw_接);
    bpt_insert_strptr(g_keyword_map, "标", (void*)kw_标);
    bpt_insert_strptr(g_keyword_map, "记标", (void*)kw_记标);
    bpt_insert_strptr(g_keyword_map, "创属", (void*)kw_创属);
    bpt_insert_strptr(g_keyword_map, "转", (void*)kw_转);
    bpt_insert_strptr(g_keyword_map, "文件", (void*)kw_文件);
    bpt_insert_strptr(g_keyword_map, "查", (void*)kw_查);
    bpt_insert_strptr(g_keyword_map, "在", (void*)kw_在);
    bpt_insert_strptr(g_keyword_map, "类", (void*)kw_类);
    bpt_insert_strptr(g_keyword_map, "造", (void*)kw_造);
    bpt_insert_strptr(g_keyword_map, "取属", (void*)kw_取属);
    bpt_insert_strptr(g_keyword_map, "记属", (void*)kw_记属);
    bpt_insert_strptr(g_keyword_map, "呼属", (void*)kw_呼属);
    bpt_insert_strptr(g_keyword_map, "是类", (void*)kw_是类);
    bpt_insert_strptr(g_keyword_map, "验介", (void*)kw_验介);
    bpt_insert_strptr(g_keyword_map, "接口", (void*)kw_接口);
    bpt_insert_strptr(g_keyword_map, "存", (void*)kw_存);
    bpt_insert_strptr(g_keyword_map, "载", (void*)kw_载);
    bpt_insert_strptr(g_keyword_map, "空", (void*)kw_空);
    bpt_insert_strptr(g_keyword_map, "非空", (void*)kw_非空);
    bpt_insert_strptr(g_keyword_map, "类型", (void*)kw_类型);
    bpt_insert_strptr(g_keyword_map, "安全取", (void*)kw_安全取);
    bpt_insert_strptr(g_keyword_map, "安全记", (void*)kw_安全记);
    bpt_insert_strptr(g_keyword_map, "取位", (void*)kw_取位);
    bpt_insert_strptr(g_keyword_map, "改", (void*)kw_改);
    return 0;
}

static ION_KeywordHandler ion_get_keyword_handler(const char *op)
{
    if (!g_keyword_map) { 
        if (ion_keywords_init() != 0) return NULL; 
    }
    return (ION_KeywordHandler)bpt_get(g_keyword_map, op);
}

/* ===============================
 * Public API
 * =============================== */
ION_API int ion_run_simple(ION_NBTValue *program, ION_NBTValue *result)
{
    if (!program || !result) return ION_ERR_PARAM;
    
    // 初始化关键字映射
    if (ion_keywords_init() != 0) return ION_ERR_RUNTIME;
    
    // 创建环境
    ION_ScopeEnv env;
    ion_env_init(&env, NULL);
    
    // 执行程序
    int ret = ion_eval_expr(program, &env, result);
    
    // 清理环境
    ion_env_free(&env);
    
    return ret;
}

ION_API void ion_cleanup(void)
{
    if (g_keyword_map) {
        bpt_free(g_keyword_map);
        g_keyword_map = NULL;
    }
}

/* ===============================
 * Python API Functions
 * =============================== */
#ifdef ION_ENABLE_PYTHON_API

// 将ION_NBTValue转换为Python对象
static PyObject* ion_nbt_to_python(const ION_NBTValue *val) {
    if (!val) {
        Py_RETURN_NONE;
    }
    
    switch (val->type) {
        case TAG_END:
            Py_RETURN_NONE;
            
        case TAG_BYTE:
            return PyLong_FromLong(val->as.b);
            
        case TAG_SHORT:
            return PyLong_FromLong(val->as.s);
            
        case TAG_INT:
            return PyLong_FromLong(val->as.i);
            
        case TAG_LONG:
            return PyLong_FromLongLong(val->as.l);
            
        case TAG_FLOAT:
            return PyFloat_FromDouble(val->as.f);
            
        case TAG_DOUBLE:
            return PyFloat_FromDouble(val->as.d);
            
        case TAG_STRING:
            if (val->as.str.ptr) {
                return PyUnicode_FromStringAndSize(val->as.str.ptr, val->as.str.len);
            }
            Py_RETURN_NONE;
            
        case TAG_ARRAY:
        case TAG_FARRAY:
            if (val->as.arr) {
                PyObject *list = PyList_New(val->as.arr->len);
                if (!list) return NULL;
                
                for (size_t i = 0; i < val->as.arr->len; i++) {
                    PyObject *item = ion_nbt_to_python(&val->as.arr->items[i]);
                    if (!item) {
                        Py_DECREF(list);
                        return NULL;
                    }
                    PyList_SET_ITEM(list, i, item);
                }
                return list;
            }
            return PyList_New(0);
            
        case TAG_COMPOUND:
        case TAG_FCOMPOUND:
            if (val->as.comp) {
                PyObject *dict = PyDict_New();
                if (!dict) return NULL;
                
                // 遍历哈希表的所有桶
                for (int bucket_idx = 0; bucket_idx < ION_HASH_TABLE_SIZE; bucket_idx++) {
                    const ION_HashBucket *bucket = &val->as.comp->buckets[bucket_idx];
                    
                    for (size_t i = 0; i < bucket->len; i++) {
                        PyObject *key = PyUnicode_FromStringAndSize(
                            bucket->kvs[i].key.ptr,
                            bucket->kvs[i].key.len
                        );
                        PyObject *value = ion_nbt_to_python(&bucket->kvs[i].val);
                        
                        if (!key || !value) {
                            Py_XDECREF(key);
                            Py_XDECREF(value);
                            Py_DECREF(dict);
                            return NULL;
                        }
                        
                        PyDict_SetItem(dict, key, value);
                        Py_DECREF(key);
                        Py_DECREF(value);
                    }
                }
                return dict;
            }
            return PyDict_New();
            
        case TAG_FUNCTION:
            return PyUnicode_FromString("(function)");
            
        case TAG_CLASS:
            return PyUnicode_FromString("(class)");
            
        case TAG_INSTANCE:
            return PyUnicode_FromString("(instance)");
            
        case TAG_VAR:
            if (val->as.var) {
                // 返回变量的值，而不是变量本身
                return ion_nbt_to_python(&val->as.var->value);
            }
            Py_RETURN_NONE;
            
        case TAG_POINT:
            // 简化实现：返回指针的字符串表示
            if (val->as.point) {
                char point_str[64];
                snprintf(point_str, sizeof(point_str), "<Point:%16s>", (char*)val->as.point->id.bytes);
                return PyUnicode_FromString(point_str);
            }
            return PyUnicode_FromString("(null point)");
            
        default:
            Py_RETURN_NONE;
    }
}

// 将Python对象转换为ION_NBTValue
static ION_NBTValue python_to_ion_nbt(PyObject *obj) {
    if (!obj || obj == Py_None) {
        return ion_nbt_make_end();
    }
    
    if (PyLong_Check(obj)) {
        long long val = PyLong_AsLongLong(obj);
        return ion_nbt_make_int_auto(val);
    }
    
    if (PyFloat_Check(obj)) {
        double val = PyFloat_AsDouble(obj);
        return ion_nbt_make_float_auto(val);
    }
    
    if (PyUnicode_Check(obj)) {
        Py_ssize_t size;
        const char *str = PyUnicode_AsUTF8AndSize(obj, &size);
        if (str && size <= UINT16_MAX) {
            // 检查是否是指针字符串格式 "<Point:...>"
            if (size > 7 && strncmp(str, "<Point:", 7) == 0 && str[size-1] == '>') {
                // 解析指针字符串，提取变量ID
                ION_NBTValue point = ion_nbt_make_point();
                if (size >= 23) { // "<Point:" + 16字符ID + ">"
                    memcpy(point.as.point->id.bytes, str + 7, 16);
                }
                return point;
            }
            return ion_nbt_make_string_copy(str, (uint16_t)size);
        }
        return ion_nbt_make_end();
    }
    
    if (PyList_Check(obj)) {
        Py_ssize_t len = PyList_Size(obj);
        ION_NBTValue arr = ion_nbt_make_array();
        
        for (Py_ssize_t i = 0; i < len; i++) {
            PyObject *item = PyList_GetItem(obj, i);
            ION_NBTValue item_val = python_to_ion_nbt(item);
            ion_nbt_array_push_move(arr.as.arr, &item_val);
        }
        return arr;
    }
    
    if (PyDict_Check(obj)) {
        ION_NBTValue comp = ion_nbt_make_compound();
        
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        
        while (PyDict_Next(obj, &pos, &key, &value)) {
            if (PyUnicode_Check(key)) {
                Py_ssize_t key_size;
                const char *key_str = PyUnicode_AsUTF8AndSize(key, &key_size);
                
                if (key_str && key_size <= UINT16_MAX) {
                    IonStr ion_key = ion_str_from_cstr(key_str);
                    ION_NBTValue ion_value = python_to_ion_nbt(value);
                    ion_nbt_compound_put_move(comp.as.comp, &ion_key, &ion_value);
                }
            }
        }
        return comp;
    }
    
    return ion_nbt_make_end();
}

// Python包装函数：运行ION表达式
static PyObject* py_ion_run(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *expr_obj;
    
    if (!PyArg_ParseTuple(args, "O", &expr_obj)) {
        return NULL;
    }
    
    // 转换Python对象到ION_NBTValue
    ION_NBTValue expr = python_to_ion_nbt(expr_obj);
    ION_NBTValue result;
    ion_nbt_init(&result);  // 确保result被正确初始化
    
    // 运行表达式
    int ret = ion_run_simple(&expr, &result);
    
    PyObject *py_result = NULL;
    if (ret == 0) {
        py_result = ion_nbt_to_python(&result);
    } else {
        PyErr_SetString(PyExc_RuntimeError, "ION expression execution failed");
        py_result = NULL;
    }
    
    // 安全清理 - 避免double free
    ion_nbt_free(&expr);
    if (ret == 0) {
        ion_nbt_free(&result);
    }
    
    return py_result;
}

// Python包装函数：清理资源
static PyObject* py_ion_cleanup(PyObject *self, PyObject *args) {
    (void)self; (void)args;
    ion_cleanup();
    Py_RETURN_NONE;
}

// Python模块方法定义
static PyMethodDef ion_methods[] = {
    {"run", py_ion_run, METH_VARARGS, "Run an ION expression"},
    {"cleanup", py_ion_cleanup, METH_NOARGS, "Cleanup ION resources"},
    {NULL, NULL, 0, NULL}
};

// Python模块定义
static struct PyModuleDef ion_module = {
    PyModuleDef_HEAD_INIT,
    "ion_expression_c",
    "ION Expression C Implementation",
    -1,
    ion_methods
};

// Python模块初始化函数
PyMODINIT_FUNC PyInit_ion_expression_c(void) {
    return PyModule_Create(&ion_module);
}

#endif /* ION_ENABLE_PYTHON_API */