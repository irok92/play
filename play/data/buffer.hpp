#pragma once

#include "play/os/mem.hpp"

namespace play
{
  struct allocator {
    
  };
  
  struct buffer {
      allocator* owner = nullptr;
      void* data = nullptr;
      size_t count = 0;
      size_t capacity = 0;
  };
  
  
  

}
