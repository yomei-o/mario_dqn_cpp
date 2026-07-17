// DQN trainer for World 1-4 (thin main over the generic trainer).
// Usage: mario14_dqn [ROM] [seed] [out.bin] [warm6.bin] [eps] [lr]
#include "train_common.hpp"
#include "mario14.h"
int main(int argc, char** argv) {
    return trn::train_level<mario14::Env>(argc, argv, "1-4", "mario14_best.bin");
}
