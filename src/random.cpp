#include <fstream>
#include <cstdlib>
#include <time.h>

int main(int argc, char const *argv[])
{
    if (argc != 3)
    {
        printf("need N and ouput file\n");
        return -1;
    }

    int N = atoi(argv[2]);

    if (N <= 0)
    {
        printf("need positive N\n");
        return -1;
    }

    float matrix[N * N];

    srand(time(NULL));

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix[i * N + j] = rand() % 20 + 1; // dense
            matrix[i * N + j] += (i == j) ? 1 : 0;

            //matrix[i * N + j] = rand() % 2; // sparse
            //matrix[i * N + j] += (i == j) ? 1 : 0;
        }
    }

    std::ofstream output(argv[1]);

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
