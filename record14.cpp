// Greedy 1-4 run recorder (thin main over the generic recorder).
// Usage: mario14_rec [ROM] [net.bin] [out.bin]
#include "record_common.hpp"
#include "mario14.h"
int main(int argc, char** argv){ return rec::record_level<mario14::Env>(argc, argv, "web/run_1-4.bin"); }
