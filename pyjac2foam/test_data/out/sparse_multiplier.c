#include "sparse_multiplier.h"

void sparse_multiplier(const double * A, const double * Vm, double* w) {
  w[0] =  A[0] * Vm[0] +  A[9] * Vm[1] +  A[18] * Vm[2] +  A[27] * Vm[3] +  A[36] * Vm[4] +  A[45] * Vm[5] +  A[54] * Vm[6] +  A[63] * Vm[7] +  A[72] * Vm[8];
  w[1] =  A[1] * Vm[0] +  A[10] * Vm[1] +  A[19] * Vm[2] +  A[28] * Vm[3] +  A[37] * Vm[4] +  A[46] * Vm[5] +  A[55] * Vm[6] +  A[64] * Vm[7] +  A[73] * Vm[8];
  w[2] =  A[2] * Vm[0] +  A[11] * Vm[1] +  A[20] * Vm[2] +  A[29] * Vm[3] +  A[38] * Vm[4] +  A[47] * Vm[5] +  A[56] * Vm[6] +  A[65] * Vm[7] +  A[74] * Vm[8];
  w[3] =  A[3] * Vm[0] +  A[12] * Vm[1] +  A[21] * Vm[2] +  A[30] * Vm[3] +  A[39] * Vm[4] +  A[48] * Vm[5] +  A[57] * Vm[6] +  A[66] * Vm[7] +  A[75] * Vm[8];
  w[4] =  A[4] * Vm[0] +  A[13] * Vm[1] +  A[22] * Vm[2] +  A[31] * Vm[3] +  A[40] * Vm[4] +  A[49] * Vm[5] +  A[58] * Vm[6] +  A[67] * Vm[7] +  A[76] * Vm[8];
  w[5] =  A[5] * Vm[0] +  A[14] * Vm[1] +  A[23] * Vm[2] +  A[32] * Vm[3] +  A[41] * Vm[4] +  A[50] * Vm[5] +  A[59] * Vm[6] +  A[68] * Vm[7] +  A[77] * Vm[8];
  w[6] =  A[6] * Vm[0] +  A[15] * Vm[1] +  A[24] * Vm[2] +  A[33] * Vm[3] +  A[42] * Vm[4] +  A[51] * Vm[5] +  A[60] * Vm[6] +  A[69] * Vm[7] +  A[78] * Vm[8];
  w[7] =  A[7] * Vm[0] +  A[16] * Vm[1] +  A[25] * Vm[2] +  A[34] * Vm[3] +  A[43] * Vm[4] +  A[52] * Vm[5] +  A[61] * Vm[6] +  A[70] * Vm[7] +  A[79] * Vm[8];
  w[8] =  A[8] * Vm[0] +  A[17] * Vm[1] +  A[26] * Vm[2] +  A[35] * Vm[3] +  A[44] * Vm[4] +  A[53] * Vm[5] +  A[62] * Vm[6] +  A[71] * Vm[7] +  A[80] * Vm[8];
}
