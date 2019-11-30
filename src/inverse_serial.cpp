#include <cstdlib>
#include <stdio.h>
#include <cmath>
#include <sys/time.h>
#include <time.h>

const int N = 5000; // N x N matrix

void displayMatrix(double matrix[][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%.2f ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

bool invertMatrix(double matrix[][N])
{
    double inverse[N][N];

    /* fill inverse with identity */

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            inverse[i][j] = (i == j) ? 1 : 0;
        }
    }

    /* row reduce at pivot m */

    for (int m = 0; m < N; m++)
    {
        /* find nonzero pivot in column m */

        int piv = m;

        while ((piv < N) && (matrix[piv][m] == 0))
        {
            piv++;
        }

        if (piv >= N)
        {
            return false; // no nonzero pivot
        }

        /* swap pivot into place by exchanging rows */

        if (piv != m)
        {
            for (int j = 0; j < N; j++)
            {
                double temp = matrix[piv][j];
                matrix[piv][j] = matrix[m][j];
                matrix[m][j] = temp;

                temp = inverse[piv][j];
                inverse[piv][j] = inverse[m][j];
                inverse[m][j] = temp;
            }
        }

        /* row reduce column m */

        for (int i = 0; i < N; i++)
        {
            if (i != m)
            {
                double temp = matrix[i][m] / matrix[m][m];
                for (int j = 0; j < N; j++)
                {
                    matrix[i][j] -= matrix[m][j] * temp;
                    inverse[i][j] -= inverse[m][j] * temp;
                }
            }
        }
    }

    /* divide row by pivot to form identity */

    for (int i = 0; i < N; i++)
    {
        double temp = matrix[i][i];
        for (int j = 0; j < N; j++)
        {
            matrix[i][j] = inverse[i][j] / temp;
        }
    }

    return true;
}

int main(const int argc, char const *argv[])
{
    double matrix[N][N];

    srand(time(NULL));

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix[i][j] = rand() % N - N / 2;
        }
    }

    //displayMatrix(matrix);

    struct timeval start;
    struct timeval end;
	gettimeofday(&start, NULL);

    bool invertible = invertMatrix(matrix);

    gettimeofday(&end, NULL);
    double time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;

    if (invertible)
    {
        //displayMatrix(matrix);
    }
    else
    {
        printf("cannot be inverted\n\n");
    }

    printf("%.2f seconds\n\n", time);

    return 0;
}
