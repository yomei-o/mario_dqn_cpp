// Greedy 1-3 run recorder (thin main over the generic recorder).
// Usage: mario13_rec [ROM] [net.bin] [out.bin]
#include "record_common.hpp"
#include "mario13.h"
int main(int argc, char** argv){ return rec::record_level<mario13::Env>(argc, argv, "web/run_1-3.bin"); }
