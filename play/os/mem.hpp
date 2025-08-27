#pragma once

#include "play/core/common.hpp"

struct play_new_wrapper {};

inline void *
operator new(size_t size, play_new_wrapper, void *ptr)
{
    return ptr;
}
inline void
operator delete(void *, play_new_wrapper, void *ptr) noexcept {}

namespace play
{

struct api_mem
{
    void *(*malloc)(size_t size);
    void (*free)(void *ptr);
    void *(*realloc)(void *ptr, size_t size);
    void *(*calloc)(size_t count, size_t size);
};

extern api_mem g_mem;

template <typename T> inline T *
new_1()
{
    PLAY_ASSERT(g_mem.malloc != nullptr);
    return new (play_new_wrapper(), g_mem.malloc(sizeof(T))) T;
}

template <typename T> inline T *
emplace_1(T* ptr)
{
    return new (play_new_wrapper(), ptr) T;
}

template <typename T> inline void
delete_1(T *&ptr)
{
    if (ptr != nullptr)
    {
        ptr->~T();
        g_mem.free(ptr);
        ptr = nullptr;
    }
}

template <typename T> inline T *
new_n(size_t count)
{
    PLAY_ASSERT(g_mem.calloc != nullptr);
    return new (play_new_wrapper(), g_mem.calloc(count, sizeof(T)))
           T[count];
}

template <typename T> inline T *
emplace_n(T* ptr, size_t count)
{
    PLAY_ASSERT(g_mem.calloc != nullptr);
    return new (play_new_wrapper(), ptr) T[count];
}

template <typename T> inline void
delete_n(T *ptr, size_t count)
{
    if (ptr != nullptr)
    {
        for (size_t i = 0; i < count; ++i)
        {
            ptr[i].~T();
        }
        PLAY_ASSERT(g_mem.free != nullptr);
        g_mem.free(ptr);
    }
}

}