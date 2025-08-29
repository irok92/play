#include "play/play_bind.hpp"

#include <string.h>


namespace play
{
context* g_play = nullptr;

void
file_changed_cb(
    watch_id id,
    const char* dir,
    const char* filename,
    int filename_length,
    watch_action action,
    void* user_data
)
{
    switch(action)
    {
    case WATCH_ACTION_ADDED:
        log_info("File added %s/%s", dir, filename);
        break;
    case WATCH_ACTION_DELETED:
        log_info("File deleted %s/%s", dir, filename);
        break;
    case WATCH_ACTION_MODIFIED:
        log_info("File modified %s/%s", dir, filename);
        break;
    default:
        break;
    }
}


void
initialize_context()
{
    watcher_init();
    watcher_watch("./play/", file_changed_cb, get_context());
}

void
shutdown_context()
{
    watcher_destroy();
}


context*
create_context()
{
    context* ctx = new_1<context>();
    context* old_ctx = get_context();
    set_context(ctx);
    initialize_context();
    if(old_ctx != nullptr)
    {
        set_context(old_ctx);
    }
    return ctx;
}

context*
get_context()
{
    return g_play;
}


void
set_context(context* ctx)
{
    g_play = ctx;
}

void
destroy_context(context* ctx)
{
    shutdown_context();
    if (ctx == g_play)
    {
        ctx = g_play;
        g_play = nullptr;
    }

    if (ctx != nullptr)
    {
        delete_1(ctx);
    }

}


void
update()
{
    watcher_update();
}

void
close()
{
    g_play->should_close = true;
}


bool
load_module(const char *name)
{
    PLAY_ASSERT(g_play != nullptr);
    shared_object& m = g_play->mod;

    bool succeeded = true;

    strncpy(m.name, name, sizeof(m.name) - 1);
    m.library = g_mod.load(m.name);

    if(!m.library)
    {
        log_error("Failed to load library at %s", m.name);
        m = {};
        return false;
    }

    m.fn_bind = reinterpret_cast<mod_fn_bind>(g_mod.get_function(m.library, "play_mod_bind"));
    m.fn_load = reinterpret_cast<mod_fn_load>(g_mod.get_function(m.library, "play_mod_load"));
    m.fn_update = reinterpret_cast<mod_fn_update>(g_mod.get_function(m.library, "play_mod_update"));
    m.fn_unload = reinterpret_cast<mod_fn_unload>(g_mod.get_function(m.library, "play_mod_unload"));

    if(!m.fn_bind)
    {
        log_error("Failed to get bind function for %s", m.name);
        succeeded = false;
    }

    if(!m.fn_load)
    {
        log_error("Failed to get load function for %s", m.name);
        succeeded = false;
    }

    if(!m.fn_update)
    {
        log_error("Failed to get update function for %s", m.name);
        succeeded = false;
    }

    if(!m.fn_unload)
    {
        log_error("Failed to get unload function for %s", m.name);
        succeeded = false;
    }

    if(!succeeded)
    {
        m = {};
        log_error("Failed to load all functions from %s", m.name);
        return false;
    }

    log_info("Loading library %s ...", m.name);

    binds bindings = {};
    play::get_binds(&bindings);
    m.fn_bind(&bindings);

    if(!m.fn_load())
    {
        log_error("Failed to load library %s", m.name);
        return false;
    }

    log_info("Loaded library %s successfully", m.name);
    return true;



}

bool
unload_module(const char *name)
{
    PLAY_ASSERT(g_play != nullptr);
    shared_object& m = g_play->mod;

    if(m.library != nullptr)
    {
        if(m.fn_unload)
        {
            m.fn_unload();
        }

        g_mod.unload(m.library);
        log_info("Unloaded library %s", m.name);
        m = {};
        return true;
    }

    return true;
}

}
