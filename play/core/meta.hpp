#pragma once

#include <string.h>
#include <stdint.h>
#include <ctype.h>

/**
 * Cross-compiler meta information
 * credits to @SandersMertens from the flecs library
 * https://github.com/SanderMertens/flecs/blob/4cafd278bdb648c277982cf2ff117cfbfb16f8bc/include/flecs/addons/cpp/component.hpp#L30
 * and
 * https://blog.molecular-matters.com/2015/12/11/getting-the-type-of-a-template-argument-as-string-without-rtti/
 */
namespace play {

#define PLAY_META_PREFIX_CONST "const "
#define PLAY_META_PREFIX_STRUCT "struct "
#define PLAY_META_PREFIX_CLASS "class "
#define PLAY_META_PREFIX_ENUM "enum "

#define PLAY_META_PREFIX_LEN_CONST (sizeof(PLAY_META_PREFIX_CONST) - 1)
#define PLAY_META_PREFIX_LEN_STRUCT (sizeof(PLAY_META_PREFIX_STRUCT) - 1)
#define PLAY_META_PREFIX_LEN_CLASS (sizeof(PLAY_META_PREFIX_CLASS) - 1)
#define PLAY_META_PREFIX_LEN_ENUM (sizeof(PLAY_META_PREFIX_ENUM) - 1)

#if defined(__clang__)
    #define PLAY_META_FUNC_NAME_FRONT(type, name)                                                   \
        ((sizeof(#type) + sizeof(" play::() [T = ") + sizeof(#name)) - 3u)
    #define PLAY_META_FUNC_NAME_BACK (sizeof("]") - 1u)
    #define PLAY_META_FUNC_NAME      __PRETTY_FUNCTION__
#elif defined(__GNUC__)
    #define PLAY_META_FUNC_NAME_FRONT(type, name)                                                   \
        ((sizeof(#type) + sizeof(" play::() [with T = ") + sizeof(#name)) - 3u)
    #define PLAY_META_FUNC_NAME_BACK (sizeof("]") - 1u)
    #define PLAY_META_FUNC_NAME      __PRETTY_FUNCTION__
#elif defined(_WIN32)
    #define PLAY_META_FUNC_NAME_FRONT(type, name)                                                   \
        ((sizeof(#type) + sizeof(" __cdecl play::<") + sizeof(#name)) - 3u)
    #define PLAY_META_FUNC_NAME_BACK (sizeof(">(void)") - 1u)
    #define PLAY_META_FUNC_NAME      __FUNCSIG__
#else
    #error "implicit component registration not supported"
#endif

#define PLAY_META_FUNC_TYPE_LEN(type, name, str)                                                    \
    (play::const_strlen(str) - (PLAY_META_FUNC_NAME_FRONT(type, name) + PLAY_META_FUNC_NAME_BACK))

#define PLAY_META_MEMBER_PAIR(type, member) #member, &type::member

template <size_t N>
static constexpr size_t
const_strlen(char const (&)[N])
{
    return N - 1; // -1 for null terminator
}


static int32_t
meta_strip_prefix(char* typeName, int32_t len, const char* prefix, int32_t prefix_len)
{
    if ((len > prefix_len) && !strncmp(typeName, prefix, prefix_len))
    {
        memmove(typeName, typeName + prefix_len, len - prefix_len);
        typeName[len - prefix_len]  = '\0';
        len                        -= prefix_len;
    }
    return len;
}

static void
meta_trim_type_name(char* typeName)
{
    int32_t len = strlen(typeName);

    len = meta_strip_prefix(typeName, len, PLAY_META_PREFIX_CONST, PLAY_META_PREFIX_LEN_CONST);
    len = meta_strip_prefix(typeName, len, PLAY_META_PREFIX_STRUCT, PLAY_META_PREFIX_LEN_STRUCT);
    len = meta_strip_prefix(typeName, len, PLAY_META_PREFIX_CLASS, PLAY_META_PREFIX_LEN_CLASS);
    len = meta_strip_prefix(typeName, len, PLAY_META_PREFIX_ENUM, PLAY_META_PREFIX_LEN_ENUM);

    while (typeName[len - 1] == ' ' || typeName[len - 1] == '&' || typeName[len - 1] == '*')
    {
        len--;
        typeName[len] = '\0';
    }

    /* Remove const at end of string */
    if (len > PLAY_META_PREFIX_LEN_CONST)
    {
        if (!strncmp(&typeName[len - PLAY_META_PREFIX_LEN_CONST], " const", PLAY_META_PREFIX_LEN_CONST))
        {
            typeName[len - PLAY_META_PREFIX_LEN_CONST] = '\0';
        }
        len -= PLAY_META_PREFIX_LEN_CONST;
    }

    /* Check if there are any remaining "struct " strings, which can happen
     * if this is a template type on msvc. */
    if (len > PLAY_META_PREFIX_LEN_STRUCT)
    {
        char* ptr = typeName;
        while ((ptr = strstr(ptr + 1, PLAY_META_PREFIX_STRUCT)) != 0)
        {
            /* Make sure we're not matched with part of a longer identifier
             * that contains 'struct' */
            if (ptr[-1] == '<' || ptr[-1] == ',' || isspace(ptr[-1]))
            {
                memmove(ptr, ptr + PLAY_META_PREFIX_LEN_STRUCT, strlen(ptr + PLAY_META_PREFIX_LEN_STRUCT) + 1);
                len -= PLAY_META_PREFIX_LEN_STRUCT;
            }
        }
    }
}

static char*
meta_get_type_name(char* type_name, const char* func_name, size_t len, size_t front_len)
{
    memcpy(type_name, func_name + front_len, len);
    type_name[len] = '\0'; // null terminate
    meta_trim_type_name(type_name);
    return type_name;
}

static char*
meta_get_symbol_name(char* symbol_name, const char* type_name, size_t len)
{
    const char* ptr;
    size_t      i;
    for (i = 0, ptr = type_name; i < len && *ptr; i++, ptr++)
    {
        if (*ptr == ':')
        {
            symbol_name[i] = '/';
            ptr++;
        }
        else
        {
            symbol_name[i] = *ptr;
        }
    }

    symbol_name[i] = '\0';

    return symbol_name;
}

template <typename T>
inline const char*
meta_type_name()
{
    static const size_t len = PLAY_META_FUNC_TYPE_LEN(const char*, meta_type_name, PLAY_META_FUNC_NAME);
    static char         result[len + 1] = {};
    static const size_t front_len       = PLAY_META_FUNC_NAME_FRONT(const char*, meta_type_name);
    static const char*  cppmeta_type_name     = meta_get_type_name(result, PLAY_META_FUNC_NAME, len, front_len);
    return cppmeta_type_name;
}

template <typename T>
inline const char*
meta_name()
{
    static const size_t len = PLAY_META_FUNC_TYPE_LEN(const char*, meta_name, PLAY_META_FUNC_NAME);
    static char         result[len + 1] = {};
    static const char*  cppSymbolName   = meta_get_symbol_name(result, meta_type_name<T>(), len);
    return cppSymbolName;
}


}