#include "play/play_bind.hpp"

extern "C"
void
play_mod_bind(play::binds* binds)
{
    play::set_binds(binds);
    play::log_info("Bound globals in module");
}

extern "C"
bool
play_mod_load()
{
    play::log_info("Inside play_mod_load!");

    play::test_case("Random Test!", []
    {
        play::test_assert("Apple!", 10 > 10);
        play::test_assert("Crabapple!", 10 > 10);
        play::test_assert("Pineapple!", 10 == 10);
        play::test_case("Random Test Inside!", []
        {
            play::test_assert("Apple!", 10 > 10);
            play::test_assert("Crabapple!", 10 > 10);
            play::test_assert("Pineapple!", 10 == 10);
        });
    });

    return true;
}

extern "C"
void
play_mod_update()
{

    //play::close();
}

extern "C"
void
play_mod_unload()
{
    play::log_info("Inside play_mod_unload");
}
