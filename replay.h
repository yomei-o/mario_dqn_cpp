#pragma once
// Fixed-capacity experience replay buffer (ring). Stores transitions and hands
// back uniform random minibatches -- the "off-policy" heart of DQN.
#include "autograd.h"
#include <vector>

namespace rl {

struct Transition {
    std::vector<float> s, ns;   // state, next state
    int a;                      // action taken
    float r;                    // reward
    bool done;
};

class Replay {
public:
    std::vector<Transition> buf;
    size_t cap, head = 0;

    explicit Replay(size_t capacity) : cap(capacity) { buf.reserve(capacity); }

    void push(Transition t) {
        if (buf.size() < cap) buf.push_back(std::move(t));
        else buf[head] = std::move(t);
        head = (head + 1) % cap;
    }

    size_t size() const { return buf.size(); }

    std::vector<int> sample(int n) const {
        std::vector<int> idx(n);
        for (int i = 0; i < n; ++i) idx[i] = (int)(ag::randf() * buf.size()) % (int)buf.size();
        return idx;
    }
};

}  // namespace rl
