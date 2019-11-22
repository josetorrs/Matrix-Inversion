#include<stdio.h>
#include <stdlib.h>
#include <math.h>
int main(int argc, char *argv[])
{
    int i,j,k,n,m,p;
    float c;
    float **A;
    FILE * f;

// Checks to see valid number of inputs given 
    if (argc > 4 || argc < 4){
        printf("Error: expected 3 input (A .mtx file, Ainv .mtx file, n), given %d\n", argc-1);
        exit(-1);
    }

// Checks to see if a valid .mtx file was given
    f = fopen(argv[1], "rb");

    if (f == NULL) {
         printf("Error opening file %s\n", argv[1]);
         exit(-1);
    }

    n = strtol(argv[3],NULL,10);
    A = (float **)malloc(n * sizeof(float *)); 
    for (i=0; i<n; i++) 
         A[i] = (float *)malloc(2*n * sizeof(float)); 

    for (i=0; i<n; i++){
        for (j=0; j<n; j++){
            fscanf(f, "%f", &A[i][j]);
            if (i==j){
                A[i][j+n] = 1;
            }
            else{
                A[i][j+n] = 0;
            }
        }
    }

    for(i=0; i<n; i++)
    {
        if (A[i][i]*A[i][i]<.00000000001){      // checks for invertability
                for (m=1; m+i<n; i++){             // loops through lower rows for nonzero in pivot
                    if (A[i+m][i]>.00000000001){   // checks if nonzero pivot
                        for (p=i;p<n;p++){
                            A[i][p] += A[i+m][p];
                        }
                        goto INNER;                // exits if pivot found
                    }
                    else if(m==n-1){
                        printf("Error matrix not invertible \n"); // if no pivot found, not inverable
                        exit(-1);
                    }
                }
                printf("Error matrix not invertible \n");         // if at the last pivot and zero, not invertable
                exit(-1);
        }
        INNER: for(j=i+1; j<n; j++){        // eliminates lower values
            c=A[j][i]/A[i][i];
            for (k=i;k<2*n;k++){
                A[j][k]-=c*A[i][k];
            }
        }
    }

    for(i=n-1; i>=0; i--){
        c=A[i][i];
        for(j=i; j<2*n; j++){   
            A[i][j] = A[i][j]/c;    // scales to have 1 on diagonal
        }
        for(j=i-1; j>=0; j--){   
            c = A[j][i];
            for(k=i;k<2*n;k++){
                A[j][k] -= c*A[i][k]; //performs back substitution
            }
        }
    }
    // Writes output to file (for testing) blocking
    FILE *of2 = fopen(argv[2], "wb");
    for (i=0; i<n; i++){
        for(j=n; j<2*n; j++){
            fprintf(of2, "%f ", A[i][j]);
        }
        fprintf(of2, "\n");
    }
    fclose(f);
    fclose(of2);
    return(0);
}
