#pragma once

#include "play/play.hpp"

namespace play
{

struct binds
{
    api_log log;
    api_mem mem;
    api_mod mod;
    context* context;
    test_context* root_tests;
    test_context* current_tests;
};

inline void
get_binds(binds* b)
{
    b->log = g_log;
    b->mem = g_mem;
    b->mod = g_mod;
    b->context = get_context();
    b->root_tests = &g_root_tests;
    b->current_tests = g_test;
}

inline void
set_binds(binds* b)
{
    g_log = b->log;
    g_mem = b->mem;
    g_mod = b->mod;
    g_test = b->current_tests;
    g_root_tests = *b->root_tests;
    set_context(b->context);
}

}