
#pragma once

#include <stdarg.h>

#include "play/core/common.hpp"

namespace play
{

enum log_level
{
    LOG_LEVEL_CRITICAL,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_WARNING,
    LOG_LEVEL_INFO,
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_TRACE
};

struct api_log
{
    void(*log_format_v)(log_level level, const char* format, va_list args);
};

extern api_log g_log;

void log_error(const char* format, ...) PLAY_ARGS_FMT(1);
void log_info(const char* format, ...) PLAY_ARGS_FMT(1);
void log_debug(const char* format, ...) PLAY_ARGS_FMT(1);
void log_warn(const char* format, ...) PLAY_ARGS_FMT(1);
void log(log_level level, const char* format, ...) PLAY_ARGS_FMT(2);


}