#include "play/os/log.hpp"

#include <stdio.h>

namespace play
{
#if PLAY_USE_DEFAULT_APIS
void
api_log_format_v(log_level level, const char* format, va_list args)
{
    // Default log implementation can be overridden
    const char* start_color =
        (level == LOG_LEVEL_CRITICAL) ? ("\33[0;31m") :
        (level == LOG_LEVEL_ERROR) ? ("\33[0;31m") :
        (level == LOG_LEVEL_WARNING) ? ("\33[0;33m") :
        (level == LOG_LEVEL_INFO) ? ("\33[0;37m") :
        (level == LOG_LEVEL_DEBUG) ? ("\33[0;32m") :
        (level == LOG_LEVEL_TRACE) ? ("\33[0;36m") :
        "\33[0;37m";

    const char* end_color = "\33[0;37m";
    fputs(start_color, stderr);
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    fputs(end_color, stderr);
}
#endif

#if PLAY_USE_DEFAULT_APIS
api_log g_log =
{
    .log_format_v = api_log_format_v
};
#else
api_log g_log =
{
    .log_format_v = nullptr
};
#endif

void
log_error(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    g_log.log_format_v(LOG_LEVEL_ERROR, format, args);
    va_end(args);
}

void
log_info(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    g_log.log_format_v(LOG_LEVEL_INFO, format, args);
    va_end(args);
}

void
log_debug(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    g_log.log_format_v(LOG_LEVEL_DEBUG, format, args);
    va_end(args);
}

void
log_warn(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    g_log.log_format_v(LOG_LEVEL_WARNING, format, args);
    va_end(args);
}


void
log(log_level level, const char* format, ...)
{
    va_list args;
    va_start(args, format);
    g_log.log_format_v(level, format, args);
    va_end(args);
}
}