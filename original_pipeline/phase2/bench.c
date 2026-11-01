#include <stdio.h>
#include <time.h>
#include "ann_predict.h"
int main(void){
    float x[ANN_N_IN]; int i,k; for(i=0;i<ANN_N_IN;i++) x[i]=0.5f;
    boolean_T f[ANN_N_OUT]; const int N=200000;
    struct timespec a,b; clock_gettime(CLOCK_MONOTONIC,&a);
    for(k=0;k<N;k++){ x[k%ANN_N_IN]=(float)(k%100)/100.0f; ann_predict(x,f); }
    clock_gettime(CLOCK_MONOTONIC,&b);
    double us=((b.tv_sec-a.tv_sec)*1e9+(b.tv_nsec-a.tv_nsec))/N/1000.0;
    printf("%s : %.3f us/inference (host x86-64, NOT the MCU)\n", ANN_USE_INT8?"int8":"f32 ", us);
    return 0;
}
