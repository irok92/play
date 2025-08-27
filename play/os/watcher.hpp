#pragma once


namespace play
{

using watch_id = int;

struct watcher;

extern watcher* g_watcher;

enum watch_action
{
    WATCH_ACTION_ADDED = 1,
    WATCH_ACTION_DELETED = 2,
    WATCH_ACTION_MODIFIED = 4
};

using fn_watcher_callback =
    void(*)(
        watch_id id,
        const char* dir,
        const char* filename,
        int filename_length,
        watch_action action,
        void* user_data
    );


void watcher_init();
void watcher_update();
watch_id watcher_watch(const char* dir, fn_watcher_callback cb, void* user_data = nullptr);
void watcher_unwatch(watch_id& id);
void watcher_destroy();

}