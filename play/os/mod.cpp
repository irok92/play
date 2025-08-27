#include "play/os/mod.hpp"


namespace play
{

api_mod g_mod =
{
    .load= nullptr,
    .get_function =  nullptr,
    .unload = nullptr,
};

}