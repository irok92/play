#pragma once 


inline void test_dyn_array()
{
    
    play::test_case("Dynamic Arrays", []{
        const int a[] = {
            1, 2, 3, 4, 5, 6, 7
        }

        play::dyn_array<int> dyn_array(a);
        dyn_array.push_back(a, a + sizeof(a) / sizeof(a[0]));
    });
}