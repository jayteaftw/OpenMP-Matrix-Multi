#include <iostream>
#include <omp.h>

using namespace std;
#define mat_idx(i,j,col) (i*col + j)

bool PRINT_ON = false;

void printMat(double *mat, size_t nrows, size_t ncols){

    if (!PRINT_ON){
        return ;
    }
    
    printf("[");
    for(size_t i = 0; i < nrows; i++){
        printf("[");
        for (size_t j = 0; j < ncols; j++){
            printf("%f", mat[(i * ncols) + j]);
    	    if (j < ncols-1){ printf(", ");}
        }
    
		if (i < nrows-1){ 
			printf("],\n");
			}
		else{
			printf("]");
		}
  }
  printf("]\n\n");
}

size_t T_pos_eq(size_t x, size_t nrows, size_t ncols){

    if (x != ncols*nrows-1)
        return (ncols * x) % (nrows * ncols - 1);
    else
        return x;

}

static int create_mat(size_t const nrows, size_t const ncols, double ** const matp)
{
    double * mat=NULL;
    if (!(mat = (double*) malloc(nrows*ncols*sizeof(*mat)))) {
        goto cleanup;
    }

    /** Initialize matrix with random values **/
    for(size_t i = 0; i < nrows; i++){
        for (size_t j = 0; j < ncols; j++){
            mat[(i * ncols) + j] = (double)(rand() % 1000) / 353.0;
        }
    }
    /** End random initialization **/
    printMat(mat, nrows, ncols);
    *matp = mat;

    return 0;

    cleanup:
    free(mat);
    return -1;
}

double * trans_mat(size_t const nrows, size_t const ncols, double * const mat){
    
    double * new_mat=NULL;
    
    if (!(new_mat = (double*) malloc(nrows*ncols*sizeof(*new_mat)))) {
        goto cleanup;
    }
    
    for(size_t i=0; i < nrows; i++){
        for(size_t j=0; j < ncols; j++){
            new_mat[mat_idx(j,i,nrows)] = mat[mat_idx(i, j, ncols)];
        }
    }
    
    printMat(new_mat, ncols, nrows );
    return new_mat;

    cleanup:
    free(new_mat);
    return 0;

}

bool check_mat(size_t arr_size, double *A, double *B){

    for(size_t idx = 0; idx < arr_size; idx++){
        if (A[idx] != B[idx])
         return false;
    }
    return true;

}

int mult_mat(size_t const n, size_t const m, size_t const p,
             double const * const A, double const * const B,
             double ** const Cp){
    
    size_t i, j, k;
    double sum, start, end;
    double *C = NULL;
    C = (double*) malloc(n*p*sizeof(*C));

    start = omp_get_wtime();
    #pragma omp parallel for collapse(2)
    for (i=0; i<n; ++i) {
        for (j=0; j<p; ++j) {
            for (k=0, sum=0.0; k<m; ++k) {
                sum += A[i*m+k] * B[k*p+j];
            }
            C[i*p+j] = sum;
        }
    }
    end = omp_get_wtime();
    printf("%f seconds\n", end - start);
    *Cp = C;
	printMat(C, n,p);
    return 0;        
}


int mult_mat_trans(size_t const n, size_t const m, size_t const p,
             double const * const A, double const * const B,
             double ** const Cp){
    
    size_t i, j, k;
    double sum, start, end;
    double *C = NULL;
    C = (double*) malloc(n*p*sizeof(*C));

    size_t idx_B;

    start = omp_get_wtime();
    #pragma omp parallel for collapse(2)
    for (i=0; i<n; ++i) {
        for (j=0; j<p; ++j) {
            for (k=0, sum=0.0; k<m; ++k) {
                mat_idx(k,j,p);
                mat_idx(i,k,m);
                sum += A[mat_idx(i,k,m)] * B[mat_idx(j,k,m)];
            }
            C[i*p+j] = sum;
        }
    }
    end = omp_get_wtime();
    printf("%f seconds\n", end - start);
    *Cp = C;
	printMat(C, n,p);
    return 0;        
}



static int mult_mat_seq(size_t const n, size_t const m, size_t const p,
                    double const * const A, double const * const B,
                    double ** const Cp)
{
  size_t i, j, k;
  double sum;
  double * C = NULL;

  if (!(C = (double*) malloc(n*p*sizeof(*C)))) {
    goto cleanup;
  }

	double start, end;
	start = omp_get_wtime();
  for (i=0; i<n; ++i) {
    for (j=0; j<p; ++j) {
      for (k=0, sum=0.0; k<m; ++k) {
        sum += A[i*m+k] * B[k*p+j];
      }
      C[i*p+j] = sum;
    }
  }
	end = omp_get_wtime();
	printf("%f seconds\n", end - start);
  *Cp = C;
	//printMat(C, n,p);
  return 0;

  cleanup:
  free(C);

  /*failure:*/
  return -1;
}


int main(int argc, char * argv[]){
    size_t nrows, ncols, ncols2;
    double *A=NULL, *B = NULL, *B_trans = NULL, *C=NULL, *C_seq=NULL;

    if (argc != 4) {
        fprintf(stderr, "usage: matmult_omp nrows ncols ncols2\n");
        return EXIT_FAILURE;
    }

    nrows = atoi(argv[1]);
    ncols = atoi(argv[2]);
    ncols2 = atoi(argv[3]);

    create_mat(nrows, ncols, &A);
    create_mat(ncols, ncols2, &B);

    B_trans = trans_mat(ncols, ncols2, B);
    //printMat(B_trans, ncols2, ncols );

    cout<<"Parallel"<<endl;
    mult_mat(nrows, ncols, ncols2, A, B, &C);
    cout<<"Parallel w/ trans"<<endl;
    mult_mat_trans(nrows, ncols, ncols2, A, B_trans, &C);
    cout<<"Parallel Seq"<<endl;
    mult_mat_seq(nrows, ncols, ncols2, A, B, &C_seq);
    

    cout<<"Matrix check: "<<check_mat(nrows*ncols2, C, C_seq)<<endl;

    return 0;


}