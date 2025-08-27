#pragma once

#include <assert.h>

#define inline_eval(...) ([&](){ __VA_ARGS__ })()

#define PLAY_ASSERT(x) assert(x)
#define PLAY_STATIC_ASSERT(cond, str) static_assert(cond, str)

#define PLAY_SIZEOF(x) (sizeof(x))
#define PLAY_COUNTOF(x) (sizeof(x) / sizeof(x[0]))

#if defined(__MINGW32__) && !defined(__clang__)
#define PLAY_ARGS_FMT(FMT) __attribute__((format(gnu_printf, FMT, FMT + 1)))
#define PLAY_ARGS_FMTLIST(FMT) __attribute__((format(gnu_printf, FMT, 0)))
#elif (defined(__clang__) || defined(__GNUC__))
#define PLAY_ARGS_FMT(FMT) __attribute__((format(printf, FMT, FMT + 1)))
#define PLAY_ARGS_FMTLIST(FMT) __attribute__((format(printf, FMT, 0)))
#else
#define PLAY_ARGS_FMT(FMT)
#define PLAY_ARGS_FMTLIST(FMT)
#endif

#ifndef PLAY_USE_DEFAULT_APIS
#define PLAY_USE_DEFAULT_APIS 1
#endif




