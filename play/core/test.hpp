#pragma once

#include "play/os/log.hpp"

#ifdef PLAY_TEST_CATCH_STD_EXCEPTIONS
#include <exception>
#endif
namespace play
{

static inline const int MAX_TESTS_PER_CASE = 512;

enum test_status
{
    TEST_STATUS_SUCCESS,
    TEST_STATUS_FAILED,
    TEST_STATUS_EXCEPTION
};

struct test_context
{
    const char* name = nullptr;
    const char* descriptions[MAX_TESTS_PER_CASE] = {};
    test_status status[MAX_TESTS_PER_CASE] = {};
    int succeeded = 0;
    int count = 0;
    int depth = 0;
};

extern test_context g_root_tests;
extern test_context* g_test;


template<typename TFunc>
inline void
test_case(const char* name, TFunc func)
{
    test_context* old_test = g_test;
    test_context group;
    group.name = name;
    group.count = 0;

    group.depth = g_test ? g_test->depth + 1 : 0;
    g_test = &group;

    play::log_info("%*s[case] '%s' begin", group.depth * 4, "", name);

    try
    {
        func();
    }
#ifdef PLAY_TEST_CATCH_STD_EXCEPTIONS
    catch (const std::exception& e)
    {
        play::log_error("Exeption caught: %s", e.what());
        g_test->test_status[g_test->test_count] = test_status_exception;
    }
#endif
    catch (...)
    {
        play::log_error("Unknown exception caught");
        g_test->status[g_test->count] = TEST_STATUS_EXCEPTION;
    }


    // Summerize group
    play::log_info("%*s[case] '%s' (%i/%i) passed", group.depth * 4, "", name, g_test->succeeded, g_test->count);

    g_test = old_test;
}

inline void
test_assert(const char* name, bool condition)
{
    if (!condition)
    {
        play::log_error("%*s[assert] '%s' failed", (g_test->depth + 1) * 4, "", name);
        g_test->status[g_test->count] = TEST_STATUS_FAILED;
    }
    else
    {
        play::log_debug("%*s[assert] '%s' passed", (g_test->depth + 1) * 4, "", name);
        g_test->status[g_test->count] = TEST_STATUS_SUCCESS;
        g_test->succeeded++;
    }

    g_test->descriptions[g_test->count] = name;
    g_test->count++;
}

}
