
#pragma once

#include <stdarg.h>

#include "play/os/process.hpp"

namespace play
{

struct binds;

using mod_fn_bind = void(*)(play::binds* binds);
using mod_fn_load = bool (*)();
using mod_fn_unload = void (*)();
using mod_fn_update = void (*)();


struct mod
{
    char name[256];
    void* library;
    mod_fn_bind fn_bind;
    mod_fn_load fn_load;
    mod_fn_unload fn_unload;
    mod_fn_update fn_update;

    uint64_t build_started;
    process build_process;
    
    
};

struct api_mod
{
    void* (*load)(const char* name);
    void* (*get_function)(void* handle, const char* name);
    void  (*unload)(void* handle);
};

extern api_mod g_mod;

}