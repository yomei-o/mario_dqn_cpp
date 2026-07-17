// DQN trainer for World 1-3 (thin main over the generic trainer).
// Usage: mario13_dqn [ROM] [seed] [out.bin] [warm6.bin] [eps] [lr]
#include "train_common.hpp"
#include "mario13.h"
int main(int argc, char** argv) {
    return trn::train_level<mario13::Env>(argc, argv, "1-3", "mario13_best.bin");
}
