#include <stdio.h>
#include <math.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>
#include "stats.h"

double ttest_equalvar(const double *data1, int n1, const double *data2, int n2)
{
    double mean1, mean2, var1, var2, num, sp, denom, t, dof, p;
    mean1 = gsl_stats_mean(data1, 1, n1);
    mean2 = gsl_stats_mean(data2, 1, n2);
    var1 = gsl_stats_variance(data1, 1, n1);
    var2 = gsl_stats_variance(data2, 1, n2);
    num = mean1 - mean2;
    sp = sqrt((((n1 - 1) * var1) + ((n2 - 1) * var2)) / (n1 + n2 - 2));
    denom = sp * sqrt((1 / (double) n1) + (1 / (double) n2));
    t = fabs(num / denom);
    dof = (double) n1 + n2 - 2;
    p = (1 - gsl_cdf_tdist_P(t, dof)) * 2;
    return p;
}

double ttest_unequalvar(const double *data1, int n1, const double *data2, int n2)
{
    double mean1, mean2, var1, var2, r1, r2, t, dof, p;
    mean1 = gsl_stats_mean(data1, 1, n1);
    mean2 = gsl_stats_mean(data2, 1, n2);
    var1 = gsl_stats_variance(data1, 1, n1);
    var2 = gsl_stats_variance(data2, 1, n2);
    r1 = var1 / n1;
    r2 = var2 / n2;
    // Compute t-statistic
    t = fabs((mean1 - mean2) / sqrt(r1 + r2));
    dof = pow(r1 + r2, 2) / ((pow(r1, 2) / (n1 - 1)) + (pow(r2, 2) / (n2 - 1)));
    p = (1 - gsl_cdf_tdist_P(t, dof)) * 2;
    return p;
}

double ttest_paired(const double *data1, const double *data2, int n)
{
    double data[n];
    double mean, sd, t, dof, p;
    for (int i = 0; i < n; i++)
    {
        data[i] = data1[i] - data2[i];
    }
    p = ttest(data, n); 
    return p;
}

double ttest(const double *data, int n)
{
    double mean, sd, t, dof, p;
    mean = gsl_stats_mean(data, 1, n);
    sd = gsl_stats_sd(data, 1, n);
    t = fabs((mean - 0) / (sd / sqrt((double) n)));
    dof = (double) n - 1;
    p = (1 - gsl_cdf_tdist_P(t, dof)) * 2;
    return p;
}
