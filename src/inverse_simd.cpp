#include <fstream>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <immintrin.h>

bool invertMatrix(int N, float matrix[])
{
    alignas(64) float inverse[N * N];

    /* fill inverse with identity */

    int index = (N * N) % 16;

    for (int m = 0; m < index; m++)
    {
        inverse[m] = 0;
    }

    for (int m = index; m < N * N; m += 16)
    {
        __m512 _zero = _mm512_set1_ps(0);
        _mm512_store_ps(inverse + m, _zero);
    }

    index = N % 16;

    for (int m = 0; m < index; m++)
    {
        inverse[m * N + m] = 1;
    }

    for (int m = index; m < N; m += 16)
    {
        __m512 _one = _mm512_set1_ps(1);
        __m512i _index = _mm512_set_epi32(0 * (N + 1), 1 * (N + 1), 2 * (N + 1), 3 * (N + 1), 4 * (N + 1), 5 * (N + 1), 6 * (N + 1), 7 * (N + 1), 8 * (N + 1), 9 * (N + 1), 10 * (N + 1), 11 * (N + 1), 12 * (N + 1), 13 * (N + 1), 14 * (N + 1), 15 * (N + 1));
        _mm512_i32scatter_ps(inverse + (m * N + m), _index, _one, 4);
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

            for (int j = index; j < N; j += 16)
            {
                __m512 _temp1 = _mm512_load_ps(matrix + (p * N + j));
                __m512 _temp2 = _mm512_load_ps(matrix + (m * N + j));
                _mm512_store_ps(matrix + (p * N + j), _temp2);
                _mm512_store_ps(matrix + (m * N + j), _temp1);

                _temp1 = _mm512_load_ps(inverse + (p * N + j));
                _temp2 = _mm512_load_ps(inverse + (m * N + j));
                _mm512_store_ps(inverse + (p * N + j), _temp2);
                _mm512_store_ps(inverse + (m * N + j), _temp1);
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

                __m512 _temp = _mm512_set1_ps(temp);
                for (int j = index; j < N; j += 16)
                {
                    __m512 _minuend = _mm512_load_ps(matrix + (i * N + j));
                    __m512 _subtrahend = _mm512_load_ps(matrix + (m * N + j));
                    _subtrahend = _mm512_mul_ps(_subtrahend, _temp);
                    _minuend = _mm512_sub_ps(_minuend, _subtrahend);
                    _mm512_store_ps(matrix + (i * N + j), _minuend);

                    _minuend = _mm512_load_ps(inverse + (i * N + j));
                    _subtrahend = _mm512_load_ps(inverse + (m * N + j));
                    _subtrahend = _mm512_mul_ps(_subtrahend, _temp);
                    _minuend = _mm512_sub_ps(_minuend, _subtrahend);
                    _mm512_store_ps(inverse + (i * N + j), _minuend);
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

        __m512 _temp = _mm512_set1_ps(temp);
        for (int j = index; j < N; j += 16)
        {
            __m512 _inverse = _mm512_load_ps(inverse + (i * N + j));
            _inverse = _mm512_div_ps(_inverse, _temp);
            _mm512_store_ps(matrix + (i * N + j), _inverse);
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
    alignas(64) float matrix[N * N];

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
