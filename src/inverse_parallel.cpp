#include <fstream>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>

const int T = 16;

bool invertMatrix(int N, float matrix[])
{
    float inverse[N * N];

    /* fill inverse with identity */

    #pragma omp parallel for num_threads(T)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            inverse[i * N + j] = (i == j) ? 1 : 0;
        }
    }

    /* row reduce at pivot m */

    for (int m = 0; m < N; m++)
    {
        /* find nonzero pivot in column m */

        int p = m;

        while ((p < N) && (fabs(matrix[p * N + m]) < 1e-5))
        {
            p++;
        }

        if (p >= N)
        {
            return false; // no nonzero pivot
        }

        /* swap pivot into place by exchanging rows */

        if (p != m)
        {
            #pragma omp parallel for num_threads(T)
            for (int j = 0; j < N; j++)
            {
                float temp = matrix[p * N + j];
                matrix[p * N + j] = matrix[m * N + j];
                matrix[m * N + j] = temp;

                temp = inverse[p * N + j];
                inverse[p * N + j] = inverse[m * N + j];
                inverse[m * N + j] = temp;
            }
        }

        /* row reduce column m */

        #pragma omp parallel for num_threads(T)
        for (int i = 0; i < N; i++)
        {
            if (i != m)
            {
                float temp = matrix[i * N + m] / matrix[m * N + m];
                for (int j = 0; j < N; j++)
                {
                    matrix[i * N + j] -= matrix[m * N + j] * temp;
                    inverse[i * N + j] -= inverse[m * N + j] * temp;
                }
            }
        }
    }

    /* divide row by pivot to form identity */
    /* transfer inverse to matrix */

    #pragma omp parallel for num_threads(T)
    for (int i = 0; i < N; i++)
    {
        float temp = matrix[i * N + i];
        for (int j = 0; j < N; j++)
        {
            matrix[i * N + j] = inverse[i * N + j] / temp;
        }
    }

    return true;
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("need input and output file\n");
        return -1;
    }

    std::ifstream input(argv[1]);

    if (!input.good())
    {
        printf("input file does not exist\n");
        return -1;
    }

    int N;
    input >> N;
    float matrix[N * N];

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float temp;
            input >> temp;
            matrix[i * N + j] = temp;
        }
    }

    input.close();

    struct timeval start;
    struct timeval end;
    gettimeofday(&start, NULL);

    bool invertible = invertMatrix(N, matrix);

    gettimeofday(&end, NULL);
    float time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;

    if (!invertible)
    {
        printf("matrix cannot be inverted\n");
        return -1;
    }

    printf("%f seconds\n", time);

    std::ofstream output(argv[2]);
    output << N << std::endl;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            output << matrix[i * N + j] << " ";
        }
        output << std::endl;
    }

    output.close();

    return 0;
}
