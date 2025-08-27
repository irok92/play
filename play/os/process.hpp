#pragma once

namespace play
{

using process = void*;

process create_process(const char* command);
void update_process(process proc);
void destroy_process(process proc);

struct api_process
{
    process (*create)(const char* command);
    void (*update)(process proc);
    void (*destroy)(process proc);
};
}