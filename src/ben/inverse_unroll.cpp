#include <fstream>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <immintrin.h>

bool invertMatrix(int N, float matrix[])
{
    float inverse[N * N];

    /* fill inverse with identity */

    int index = (N * N) % 8;

    for (int m = 0; m < index; m += 8)
    {
        inverse[m] = 0;
    }

    for (int m = index; m < N * N; m += 8)
    {
        inverse[m + 0] = 0;
        inverse[m + 1] = 0;
        inverse[m + 2] = 0;
        inverse[m + 3] = 0;
        inverse[m + 4] = 0;
        inverse[m + 5] = 0;
        inverse[m + 6] = 0;
        inverse[m + 7] = 0;
    }

    index = N % 8;

    for (int m = 0; m < index; m++)
    {
        inverse[m * N + m] = 1;
    }

    for (int m = index; m < N; m += 8)
    {
        inverse[(m + 0) * N + m + 0] = 1;
        inverse[(m + 1) * N + m + 1] = 1;
        inverse[(m + 2) * N + m + 2] = 1;
        inverse[(m + 3) * N + m + 3] = 1;
        inverse[(m + 4) * N + m + 4] = 1;
        inverse[(m + 5) * N + m + 5] = 1;
        inverse[(m + 6) * N + m + 6] = 1;
        inverse[(m + 7) * N + m + 7] = 1;
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
            for (int j = 0; j < index; j++)
            {
                float temp = matrix[p * N + j];
                matrix[p * N + j] = matrix[m * N + j];
                matrix[m * N + j] = temp;

                temp = inverse[p * N + j];
                inverse[p * N + j] = inverse[m * N + j];
                inverse[m * N + j] = temp;
            }

            for (int j = index; j < N; j += 8)
            {
                float temp = matrix[p * N + j + 0];
                matrix[p * N + j + 0] = matrix[m * N + j + 0];
                matrix[m * N + j + 0] = temp;

                temp = inverse[p * N + j + 0];
                inverse[p * N + j + 0] = inverse[m * N + j + 0];
                inverse[m * N + j + 0] = temp;

                temp = matrix[p * N + j + 1];
                matrix[p * N + j + 1] = matrix[m * N + j + 1];
                matrix[m * N + j + 1] = temp;

                temp = inverse[p * N + j + 1];
                inverse[p * N + j + 1] = inverse[m * N + j + 1];
                inverse[m * N + j + 1] = temp;

                temp = matrix[p * N + j + 2];
                matrix[p * N + j + 2] = matrix[m * N + j + 2];
                matrix[m * N + j + 2] = temp;

                temp = inverse[p * N + j + 2];
                inverse[p * N + j + 2] = inverse[m * N + j + 2];
                inverse[m * N + j + 2] = temp;

                temp = matrix[p * N + j + 3];
                matrix[p * N + j + 3] = matrix[m * N + j + 3];
                matrix[m * N + j + 3] = temp;

                temp = inverse[p * N + j + 3];
                inverse[p * N + j + 3] = inverse[m * N + j + 3];
                inverse[m * N + j + 3] = temp;

                temp = matrix[p * N + j + 4];
                matrix[p * N + j + 4] = matrix[m * N + j + 4];
                matrix[m * N + j + 4] = temp;

                temp = inverse[p * N + j + 4];
                inverse[p * N + j + 4] = inverse[m * N + j + 4];
                inverse[m * N + j + 4] = temp;

                temp = matrix[p * N + j + 5];
                matrix[p * N + j + 5] = matrix[m * N + j + 5];
                matrix[m * N + j + 5] = temp;

                temp = inverse[p * N + j + 5];
                inverse[p * N + j + 5] = inverse[m * N + j + 5];
                inverse[m * N + j + 5] = temp;

                temp = matrix[p * N + j + 6];
                matrix[p * N + j + 6] = matrix[m * N + j + 6];
                matrix[m * N + j + 6] = temp;

                temp = inverse[p * N + j + 6];
                inverse[p * N + j + 6] = inverse[m * N + j + 6];
                inverse[m * N + j + 6] = temp;

                temp = matrix[p * N + j + 7];
                matrix[p * N + j + 7] = matrix[m * N + j + 7];
                matrix[m * N + j + 7] = temp;

                temp = inverse[p * N + j + 7];
                inverse[p * N + j + 7] = inverse[m * N + j + 7];
                inverse[m * N + j + 7] = temp;
            }
        }

        /* row reduce column m */

        for (int i = 0; i < N; i++)
        {
            if (i != m)
            {
                float temp = matrix[i * N + m] / matrix[m * N + m];
                for (int j = 0; j < index; j++)
                {
                    matrix[i * N + j] -= matrix[m * N + j] * temp;
                    inverse[i * N + j] -= inverse[m * N + j] * temp;
                }
                for (int j = index; j < N; j += 8)
                {
                    matrix[i * N + j + 0] -= matrix[m * N + j + 0] * temp;
                    inverse[i * N + j + 0] -= inverse[m * N + j + 0] * temp;

                    matrix[i * N + j + 1] -= matrix[m * N + j + 1] * temp;
                    inverse[i * N + j + 1] -= inverse[m * N + j + 1] * temp;

                    matrix[i * N + j + 2] -= matrix[m * N + j + 2] * temp;
                    inverse[i * N + j + 2] -= inverse[m * N + j + 2] * temp;

                    matrix[i * N + j + 3] -= matrix[m * N + j + 3] * temp;
                    inverse[i * N + j + 3] -= inverse[m * N + j + 3] * temp;

                    matrix[i * N + j + 4] -= matrix[m * N + j + 4] * temp;
                    inverse[i * N + j + 4] -= inverse[m * N + j + 4] * temp;

                    matrix[i * N + j + 5] -= matrix[m * N + j + 5] * temp;
                    inverse[i * N + j + 5] -= inverse[m * N + j + 5] * temp;

                    matrix[i * N + j + 6] -= matrix[m * N + j + 6] * temp;
                    inverse[i * N + j + 6] -= inverse[m * N + j + 6] * temp;

                    matrix[i * N + j + 7] -= matrix[m * N + j + 7] * temp;
                    inverse[i * N + j + 7] -= inverse[m * N + j + 7] * temp;
                }
            }
        }
    }

    /* divide row by pivot to form identity */
    /* transfer inverse to matrix */

    for (int i = 0; i < N; i++)
    {
        float temp = matrix[i * N + i];
        for (int j = 0; j < index; j++)
        {
            matrix[i * N + j] = inverse[i * N + j] / temp;
        }
        for (int j = index; j < N; j += 8)
        {
            matrix[i * N + j + 0] = inverse[i * N + j + 0] / temp;
            matrix[i * N + j + 1] = inverse[i * N + j + 1] / temp;
            matrix[i * N + j + 2] = inverse[i * N + j + 2] / temp;
            matrix[i * N + j + 3] = inverse[i * N + j + 3] / temp;
            matrix[i * N + j + 4] = inverse[i * N + j + 4] / temp;
            matrix[i * N + j + 5] = inverse[i * N + j + 5] / temp;
            matrix[i * N + j + 6] = inverse[i * N + j + 6] / temp;
            matrix[i * N + j + 7] = inverse[i * N + j + 7] / temp;
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
