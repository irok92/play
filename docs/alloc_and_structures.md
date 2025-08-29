# Play Structure

Play uses a table-graph data storage using buffers as its base data structure

```cpp

struct buffer {
  void* buf;
  uint32_t size;
  uint32_t capacity;
};

```
