#include <fstream>
#include <stdio.h>
#include <sys/time.h>

bool invertMatrix(int N, int T, double matrix[])
{
    double inverse[N*N];

    /* fill inverse with identity */

    #pragma omp parallel for num_threads(T)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            inverse[i*N+j] = (i == j) ? 1 : 0;
        }
    }

    /* row reduce at pivot m */

    for (int m = 0; m < N; m++)
    {
        /* find nonzero pivot in column m */

        int piv = m;

        while ((piv < N) && (matrix[piv*N+m] == 0))
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
            #pragma omp parallel for num_threads(T)
            for (int j = 0; j < N; j++)
            {
                double temp = matrix[piv*N+j];
                matrix[piv*N+j] = matrix[m*N+j];
                matrix[m*N+j] = temp;

                temp = inverse[piv*N+j];
                inverse[piv*N+j] = inverse[m*N+j];
                inverse[m*N+j] = temp;
            }
        }

        /* row reduce column m */

        #pragma omp parallel for num_threads(T)
        for (int i = 0; i < N; i++)
        {
            if (i != m)
            {
                double temp = matrix[i*N+m] / matrix[m*N+m];
                for (int j = 0; j < N; j++)
                {
                    matrix[i*N+j] -= matrix[m*N+j] * temp;
                    inverse[i*N+j] -= inverse[m*N+j] * temp;
                }
            }
        }
    }

    /* divide row by pivot to form identity */
    /* transfer inverse to matrix */

    #pragma omp parallel for num_threads(T)
    for (int i = 0; i < N; i++)
    {
        double temp = matrix[i*N+i];
        for (int j = 0; j < N; j++)
        {
            matrix[i*N+j] = inverse[i*N+j] / temp;
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
    int T = 16;
    double matrix[N*N];

	for (int i = 0; i < N; i++)
	{
        for (int j = 0; j < N; j++)
        {
            double temp;
            input >> temp;
            matrix[i*N+j] = temp;
        }
	}

    input.close();

    struct timeval start;
    struct timeval end;
	gettimeofday(&start, NULL);

    bool invertible = invertMatrix(N, T, matrix);

    gettimeofday(&end, NULL);
    double time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;

    if (!invertible)
    {
        printf("matrix cannot be inverted\n");
        return -1;
    }
    
    printf("%.2f seconds\n", time);

    std::ofstream output(argv[2]);
    output << N << std::endl;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            output << matrix[i*N+j] << " ";
        }
        output << std::endl;
    }

    output.close();

    return 0;
}
