#pragma once

#include <stdint.h>

#include "play/os/mem.hpp"

namespace play
{

template<typename T>
struct slice
{
    T* data;
    size_t size;
};

template<typename T>
struct dyn_array
{
    T* buffer;
    size_t count;
    size_t capacity;

    typedef dyn_array<T> self_t;

    typedef T value_type;
    typedef value_type* iterator;
    typedef const value_type* const_iterator;

    inline
    dyn_array()
    { buffer = nullptr; count = 0; capacity = 0; }

    inline
    dyn_array(const dyn_array<T>& src) : dyn_array()
    { operator=(src); }

    inline
    dyn_array(const T* _start, const T* _end) : dyn_array()
    { push_back(_start, _end); }


    inline
    ~dyn_array()
    { g_mem.free(buffer); }

    inline dyn_array<T>&
    operator=(const dyn_array<T>& src)
    {
        clear();
        resize(src.count);
        if(src.buffer)
            memcpy(buffer, src.buffer, sizeof(T) * src.count);
        return *this;
    }

    T&
    operator[](size_t index) { PLAY_ASSERT(index < count); return buffer[index]; }

    const T&
    operator[](size_t index) const { PLAY_ASSERT(index < count); return buffer[index]; }

    inline self_t&
    clear()
    {
        if(buffer)
        {
            g_mem.free(buffer);
            buffer = nullptr;
        }
        count = 0;
        capacity = 0;
        return *this;
    }

    inline size_t
    _grow_capacity(size_t new_size) const
    {
        // Increase by 1.5x each time
        size_t new_capacity = capacity ? (capacity + (capacity >> 1)) : 8;
        return new_capacity > new_size ? new_capacity : new_size;
    }

    inline self_t&
    resize(size_t new_size, T* init_value = nullptr)
    {
        if (new_size > capacity)
        {
            reserve(_grow_capacity(new_size));
        }
        if(new_size > count && init_value != nullptr)
        {
            T value = *init_value;
            for(size_t i = count; i < new_size; ++i)
            {
                buffer[i] = value;
            }
        }
        count = new_size;
        return *this;
    }

    inline self_t&
    reserve(size_t new_capacity)
    {
        if(new_capacity < capacity)
        {
            return;
        }

        T* new_buffer = static_cast<T*>(g_mem.malloc(sizeof(T) * new_capacity));
        if(buffer)
        {
            memcpy(new_buffer, buffer, sizeof(T) * count);
            g_mem.free(buffer);
        }
        buffer = new_buffer;
        capacity = new_capacity;
        return *this;
    }

    inline self_t&
    erase(size_t start, size_t end)
    {
        PLAY_ASSERT(start <= end);
        PLAY_ASSERT(end <= count);

        if (start == end) return *this;

        size_t num_elements = end - start;
        memmove(&buffer[start], &buffer[end], sizeof(T) * (count - end));
        count -= num_elements;
        return *this;
    }

    inline T*
    begin() { return buffer; }

    inline const T*
    begin() const { return buffer; }

    inline T*
    end() { return buffer + count; }

    inline const T*
    end() const { return buffer + count; }

    inline self_t&
    shrink(size_t new_size)
    {
        PLAY_ASSERT(new_size <= count);
        count = new_size;
        return *this;
    }

    inline self_t&
    push_back(const T& value)
    {
        if(count == capacity)
            reserve(_grow_capacity(count + 1));
        memcpy(&buffer[count], &value, sizeof(T));
        count++;
        return *this;
    }

    inline self_t&
    push_back(const T* _start, const T* _end)
    {
        if(capacity < count + (_end - _start))
            reserve(_grow_capacity(count + (_end - _start)));
        memcpy(&buffer[count], _start, sizeof(T) * (_end - _start));
        count += (_end - _start);
        return *this;
    }

    inline self_t&
    erase_back(size_t n = 1)
    {
        PLAY_ASSERT(n <= count);
        count -= n;
        return *this;
    }

    // Pops the elements into another array in actual order
    inline size_t
    pop_back(T* out, size_t n = 1)
    {
        if (n > count) n = count;
        if (out)
        {
            memcpy(out, &buffer[count - n], sizeof(T) * n);
        }
        count -= n;
        return n;
    }

    inline T*
    erase(const T* it_begin)
    {
        PLAY_ASSERT(it_begin >= buffer && it_begin < buffer + count);
        size_t index = it_begin - buffer;
        memmove(&buffer[index], &buffer[index + 1], sizeof(T) * (count - index - 1));
        count--;
        return buffer + index;
    }

    inline T*
    erase(const T* it_begin, const T* it_end)
    {
        PLAY_ASSERT(it_begin >= buffer && it_begin < buffer + count);
        PLAY_ASSERT(it_end > it_begin && it_end <= buffer + count);
        size_t index_begin = it_begin - buffer;
        size_t index_end = it_end - buffer;
        size_t num_elements = index_end - index_begin;
        memmove(&buffer[index_begin], &buffer[index_end], sizeof(T) * (count - index_end));
        count -= num_elements;
        return buffer + index_begin;
    }

    inline T*
    insert_n(const T* it_pos, const T& value)
    {
        PLAY_ASSERT(it_pos >= buffer && it_pos < buffer + count);
        size_t index = it_pos - buffer;
        if (count == capacity)
            reserve(_grow_capacity(count + 1));
        memmove(&buffer[index + 1], &buffer[index], sizeof(T) * (count - index));
        buffer[index] = value;
        count++;
        return buffer + index;
    }

};

struct string_buffer
{
    dyn_array<char> data;
    dyn_array<size_t> offsets;

};

}
