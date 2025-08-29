#pragma once

#include "common.hpp"

#include "play/os/mod.hpp"

namespace play
{

struct context
{
    bool should_close = false;
    shared_object mod;
};

extern context* g_play;

context* create_context();
void     destroy_context(context* ctx = nullptr);

context* get_context();
void     set_context(context* ctx);

void update();
inline bool
is_running() { return g_play != nullptr && !g_play->should_close; }
void close();

bool load_module(const char* name);
bool unload_module(const char* name);

}
