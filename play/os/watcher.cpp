#include "play/play.hpp"


#define _WIN32_WINNT 0x0550
#include <windows.h>

#if defined(_MSC_VER)
#pragma comment(lib, "comctl32.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "ole32.lib")

// disable secure warnings
#pragma warning(disable : 4996)
#endif



namespace play
{
struct watcher;

// TODO: Add ifdefs for a Linux/Unix variant
struct watch_instance
{
    OVERLAPPED overlapped;
    HANDLE dir_handle;
    LPARAM lparam;
    DWORD notify_filter;
    bool stop_now;
    char* dir_name;
    BYTE buffer[32*1024];
    watch_id id;
    fn_watcher_callback callback;
    void* user_data;
};

struct watcher
{
    watch_instance* watchers[16] = {};
    int count = 0;
};

watcher* g_watcher = nullptr;

bool
watcher_refresh_dir(watch_instance* watch, bool clear = false);


void
watcher_handle_action(
    watch_instance* instance,
    const char* filename,
    size_t filename_length,
    unsigned long action,
    void* user_data
)
{
    watch_action local_action = WATCH_ACTION_ADDED;

    switch(action)
    {
    case FILE_ACTION_RENAMED_NEW_NAME:
    case FILE_ACTION_ADDED:
        local_action = WATCH_ACTION_ADDED;
        break;
    case FILE_ACTION_RENAMED_OLD_NAME:
    case FILE_ACTION_REMOVED:
        local_action = WATCH_ACTION_DELETED;
        break;
    case FILE_ACTION_MODIFIED:
        local_action = WATCH_ACTION_MODIFIED;
        break;
    default:
        play::log_warn("Unknown file action: %lu", action);
        return;
    }

    if(instance->callback != nullptr)
    {
        instance->callback(
            instance->id,
            instance->dir_name,
            filename,
            filename_length,
            local_action,
            user_data
        );
    }
    else
    {
        play::log_warn("No callback was set for file_watcher %d", instance->id);
    }
}

void
watcher_init()
{
    g_watcher = play::new_1<watcher>();
}


void
watcher_update()
{
    DWORD result = MsgWaitForMultipleObjectsEx(0, NULL, 0, QS_ALLINPUT, MWMO_ALERTABLE);
}

watch_id
watcher_watch(
    const char* dir,
    fn_watcher_callback cb,
    void* user_data
)
{
    PLAY_ASSERT(g_watcher != nullptr);

    if(g_watcher->count >= (PLAY_COUNTOF(g_watcher->watchers) - 1))
    {
        play::log_error("Failed to create watcher at %s, maximum watchers reached.", dir);
        return -1;
    }
    // TODO: possibly add a nullptr fill in for "released" watchers.
    // 0 is a valid id to keep it aligned with the array.
    watch_id id = g_watcher->count;
    // Append count afterwards.
    g_watcher->count++;

    watch_instance* i = nullptr;
    size_t i_size = sizeof(*i);
    i = static_cast<watch_instance*>(HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, i_size));
    *i = {};
    i->id = id;
    i->stop_now = false;
    i->callback = cb;
    i->user_data = user_data;
    i->dir_name = strdup(dir);

    i->dir_handle = CreateFile(
                        i->dir_name,
                        FILE_LIST_DIRECTORY,
                        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                        NULL,
                        OPEN_EXISTING,
                        FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OVERLAPPED,
                        NULL
                    );


    if(i->dir_handle != INVALID_HANDLE_VALUE)
    {
        i->overlapped.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
        i->notify_filter = FILE_NOTIFY_CHANGE_CREATION | FILE_NOTIFY_CHANGE_SIZE | FILE_NOTIFY_CHANGE_FILE_NAME;

        if(i->overlapped.hEvent == NULL)
        {
            log_error("Failed to create event for directory '%s'. Error: %lu", i->dir_name, GetLastError());
            CloseHandle(i->dir_handle);
            HeapFree(GetProcessHeap(), 0, i);
            return -1;
        }

        if(watcher_refresh_dir(i))
        {
            log_info("Started watching directory '%s' with ID %i", i->dir_name, id);
            g_watcher->watchers[id] = i;
            return id;
        }

        CloseHandle(i->overlapped.hEvent);
        CloseHandle(i->dir_handle);
        HeapFree(GetProcessHeap(), 0, i);
    }


    log_info("Failed to open directory '%s'. Error: %lu", dir, GetLastError());


    return -1;
}

void
watcher_unwatch(watch_id& id)
{
    PLAY_ASSERT(g_watcher != nullptr);

    if(id < 0 || id >= PLAY_COUNTOF(g_watcher->watchers))
    {
        log_warn("watch_id invalid or out of range %i", id);
        return;
    }

    watch_instance* i = g_watcher->watchers[id];
    if(!i)
    {
        log_warn("Couldn't free instance at id %i, was null.", id);
        return;
    }

    i->stop_now = true;
    CancelIo(i->dir_handle);
    watcher_refresh_dir(i, true);
    if(!HasOverlappedIoCompleted(&i->overlapped))
    {
        SleepEx(5, TRUE);
    }

    CloseHandle(i->overlapped.hEvent);
    CloseHandle(i->dir_handle);
    free(i->dir_name);
    i->~watch_instance();
    HeapFree(GetProcessHeap(), 0, i);
    g_watcher->watchers[id] = nullptr;
    id = -1;
}

void
CALLBACK
watcher_callback(
    DWORD dwErrorCode,
    DWORD dwNumberOfBytesTransfered,
    LPOVERLAPPED lpOverlapped)
{
    TCHAR szFile[MAX_PATH];
    PFILE_NOTIFY_INFORMATION pNotify;
    play::watch_instance* i = reinterpret_cast<play::watch_instance*>(lpOverlapped);
    size_t offset = 0;


    if(dwNumberOfBytesTransfered == 0)
    {
        return;
    }

    if(dwErrorCode == ERROR_SUCCESS)
    {
        do
        {
            pNotify = (PFILE_NOTIFY_INFORMATION)&i->buffer[offset];
            offset = pNotify->NextEntryOffset;
#if defined(UNICODE)
            size_t length = min(MAX_PATH, pNotify->FileNameLength / sizeof(WCHAR) + 1);
            lstrcpynW(szFile, pNotify->FileName, length);
            szFile[length] = TEXT('\0');

#else
            size_t length = WideCharToMultiByte(
                                CP_ACP,
                                0,
                                pNotify->FileName,
                                pNotify->FileNameLength / sizeof(WCHAR),
                                szFile,
                                MAX_PATH - 1,
                                NULL,
                                NULL
                            );
            szFile[length] = TEXT('\0');
#endif
            watcher_handle_action(i, szFile, length, pNotify->Action, i->user_data);
        }
        while(pNotify->NextEntryOffset != 0);
    }

    if(!i->stop_now)
    {
        watcher_refresh_dir(i);
    }
}

bool
watcher_refresh_dir(watch_instance* i, bool _clear)
{
    log_info("Refreshing dir %p %s", i, _clear ? "true" : "false");

    int res = ReadDirectoryChangesW(
                  i->dir_handle,
                  i->buffer,
                  sizeof(i->buffer),
                  TRUE,
                  i->notify_filter,
                  NULL,
                  &i->overlapped,
                  _clear ? 0 : watcher_callback
              );

    return res != 0;
}

void
watcher_destroy()
{
    PLAY_ASSERT(g_watcher != nullptr);

    for(int i = 0; i < PLAY_COUNTOF(g_watcher->watchers); i++)
    {
        if(g_watcher->watchers[i] != nullptr)
        {
            watcher_unwatch(i);
        }
    }
}

}