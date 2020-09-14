#include <math.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>

double t_test(const double *data1, int n1, const double *data2, int n2)
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
