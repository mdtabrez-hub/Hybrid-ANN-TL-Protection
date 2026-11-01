/* test_host.c -- run the generated C over the frozen test set and compare it
   against the Python predictions. Host build only (reads CSV from disk).

   usage: ./test_host X_test.csv y_pred_python.csv                            */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ann_predict.h"

#define MAXN 20000

static int read_csv(const char *path, float *buf, int ncol, int maxrow)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    char line[16384];
    if (!fgets(line, sizeof line, f)) { fclose(f); return 0; }   /* header */
    int n = 0;
    while (fgets(line, sizeof line, f) && n < maxrow) {
        char *p = line;
        for (int c = 0; c < ncol; ++c) {
            buf[n * ncol + c] = strtof(p, &p);
            if (*p == ',') ++p;
        }
        ++n;
    }
    fclose(f);
    return n;
}

int main(int argc, char **argv)
{
    static float X[MAXN * ANN_N_IN];
    static float Y[MAXN * ANN_N_OUT];
    const char *xf = argc > 1 ? argv[1] : "X_test.csv";
    const char *yf = argc > 2 ? argv[2] : "y_pred_python.csv";

    int n  = read_csv(xf, X, ANN_N_IN,  MAXN);
    int ny = read_csv(yf, Y, ANN_N_OUT, MAXN);
    if (n != ny) { fprintf(stderr, "row mismatch %d vs %d\n", n, ny); return 1; }

    long disagree = 0, exact_match = 0;
    for (int i = 0; i < n; ++i) {
        boolean_T f[ANN_N_OUT];
        ann_predict(&X[i * ANN_N_IN], f);
        int all_ok = 1;
        for (int j = 0; j < ANN_N_OUT; ++j) {
            int want = (int)(Y[i * ANN_N_OUT + j] + 0.5f);
            if ((int)f[j] != want) { ++disagree; all_ok = 0; }
        }
        exact_match += all_ok;
    }
    long total = (long)n * ANN_N_OUT;
    printf("build            : %s\n", ANN_USE_INT8 ? "int8 weights" : "float32 weights");
    printf("samples          : %d\n", n);
    printf("compared outputs : %ld\n", total);
    printf("disagreements    : %ld\n", disagree);
    printf("discrepancy D    : %.6f %%\n", 100.0 * disagree / total);
    printf("rows matching py : %ld  (%.4f %%)\n", exact_match, 100.0 * exact_match / n);
    return 0;
}
