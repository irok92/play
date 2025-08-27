#include "play/os/mem.hpp"

#include <stdlib.h>


namespace play
{

#if PLAY_USE_DEFAULT_APIS
api_mem g_mem =
{
    .malloc = malloc,
    .free = free,
    .realloc = realloc,
    .calloc = calloc
};
#else
api_mem g_mem =
{
    .malloc = nullptr,
    .free = nullptr,
    .realloc = nullptr,
    .calloc = nullptr
};
#endif
}