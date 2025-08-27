#include "play/play.hpp"

#include <SDL3/SDL.h>

namespace play_sdl3
{

void
api_log_format_v(play::log_level level, const char* format, va_list args)
{
    SDL_LogPriority priority =
        (level == play::LOG_LEVEL_CRITICAL) ? SDL_LOG_PRIORITY_CRITICAL :
        (level == play::LOG_LEVEL_ERROR) ? SDL_LOG_PRIORITY_ERROR :
        (level == play::LOG_LEVEL_WARNING) ? SDL_LOG_PRIORITY_WARN :
        (level == play::LOG_LEVEL_INFO) ? SDL_LOG_PRIORITY_INFO :
        (level == play::LOG_LEVEL_DEBUG) ? SDL_LOG_PRIORITY_DEBUG :
        (level == play::LOG_LEVEL_TRACE) ? SDL_LOG_PRIORITY_TRACE :
        SDL_LOG_PRIORITY_INVALID;

    SDL_LogMessageV(SDL_LOG_CATEGORY_APPLICATION, priority, format, args);
}

void*
api_mod_load(const char* path)
{
    return static_cast<void*>(SDL_LoadObject(path));
}

void
api_mod_unload(void* lib)
{
    SDL_UnloadObject(static_cast<SDL_SharedObject*>(lib));
}

void*
api_mod_get_function(void* lib, const char* address)
{
    return reinterpret_cast<void*>(SDL_LoadFunction(reinterpret_cast<SDL_SharedObject*>(lib), address));
}

void
setup_os_apis()
{
    //play::g_log.log_format_v = api_log_format_v;

    play::g_mem.malloc = SDL_malloc;
    play::g_mem.calloc = SDL_calloc;
    play::g_mem.realloc = SDL_realloc;
    play::g_mem.free = SDL_free;

    play::g_mod.load = api_mod_load;
    play::g_mod.unload = api_mod_unload;
    play::g_mod.get_function = api_mod_get_function;

}

}