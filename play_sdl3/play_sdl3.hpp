#pragma once

union SDL_Event;

namespace play_sdl3
{
void setup_os_apis();
void setup_apis();
void push_event(SDL_Event* event);
}