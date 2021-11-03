#include <stdio.h>
#include "stats.h"

#define SIZE 10

void print_array(double *d, int len)
{
    printf("[");
    for (int i = 0; i < len; i++)
    {
        printf("%lf", d[i]);
        if (i < len-1)
            printf(", ");
    }
    printf("]\n");
    return;
}

int main(void)
{
    
    // Allocate data
    char* input_file = "data.txt";
    double d1[SIZE];
    double d2[SIZE];
    double d3[SIZE];

    FILE *fp = fopen(input_file, "r");
    if (fp == NULL)
    {
        printf("Could not open file %s\n", input_file);
        return 1;
    }
    
    for (int i = 0; i < SIZE; i++)
    {
        fscanf(fp, "%lf,%lf,%lf", &d1[i], &d2[i], &d3[i]);
    }

    double p = ttest_equalvar(d1, SIZE, d2, SIZE);
    printf("Group 1 vs Group 2: p = %lf\n", p);
    p = ttest_unequalvar(d1, SIZE, d2, SIZE);
    printf("Group 1 vs Group 2: p = %lf\n", p);
    p = ttest_paired(d1, d2, SIZE);
    printf("Group 1 vs Group 2: p = %lf\n", p);
}

