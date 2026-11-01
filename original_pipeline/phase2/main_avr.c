/* main_avr.c -- minimal firmware translation unit for size measurement on the
   ATmega328P. It performs one inference per loop on a volatile input buffer so
   that the optimiser cannot fold the network away, and writes the four flags to
   PORTB. It is not the Proteus sketch; it exists so that avr-size reports the
   flash and SRAM actually occupied by the generated network. */
#include <avr/io.h>
#include "ann_predict.h"

volatile float g_features[ANN_N_IN];   /* volatile: the compiler must keep it */
volatile unsigned char g_flags[ANN_N_OUT];

int main(void)
{
    DDRB = 0x0F;
    for (;;) {
        float x[ANN_N_IN];
        boolean_T f[ANN_N_OUT];
        for (unsigned char i = 0; i < ANN_N_IN; ++i) x[i] = g_features[i];
        ann_predict(x, f);
        unsigned char out = 0;
        for (unsigned char j = 0; j < ANN_N_OUT; ++j) {
            g_flags[j] = f[j];
            if (f[j]) out |= (unsigned char)(1u << j);
        }
        PORTB = out;
    }
}
