#define SDL_MAIN_USE_CALLBACKS 1
#include <SDL3/SDL_main.h>
#include <SDL3/SDL.h>

#include "play_sdl3/play_sdl3.hpp"
#include "play/play.hpp"

SDL_AppResult
SDL_AppInit(void** state, int argc, char** argv)
{
    SDL_SetLogPriorities(SDL_LOG_PRIORITY_TRACE);
    if(!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS))
    {
        return SDL_APP_FAILURE;
    }

    play_sdl3::setup_apis();

    if(!play::create_context())
    {
        return SDL_APP_FAILURE;
    }

    if(!play::load_module("./libplay_tests.so"))
    {
        return SDL_APP_FAILURE;
    }

    return SDL_APP_CONTINUE;
}

static Uint64 timestamp = 0;

SDL_AppResult
SDL_AppIterate(void* state)
{

    SDL_Delay(1);
    if(SDL_GetTicks() > timestamp + 5000)
    {
        timestamp = SDL_GetTicks();
        play::log_info("%lu seconds have passed!", (timestamp / 1000));
    }

    play::update();

    return play::is_running() ? SDL_APP_CONTINUE : SDL_APP_SUCCESS;
}


SDL_AppResult
SDL_AppEvent(void* state, SDL_Event* event)
{
    if(event->type == SDL_EVENT_QUIT)
    {
        play::close();
    }

    play_sdl3::push_event(event);
    return SDL_APP_CONTINUE;
}



void
SDL_AppQuit(void* state, SDL_AppResult result)
{
    play::unload_module("play_tests");
    play::destroy_context();
}
