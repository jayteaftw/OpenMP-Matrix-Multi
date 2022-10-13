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


int mult_mat_transposed(size_t const n, size_t const m, size_t const p,
             double const * const A, double const * const B,
             double ** const Cp){
    
    size_t i, j, k;
    double sum, start, end;
    double *C = NULL;
    C = (double*) malloc(n*p*sizeof(*C));
    start = omp_get_wtime();
    //#pragma omp parallel for collapse(2)
    for (i=0; i<n; ++i) {
        for (j=0; j<p; ++j) {
            for (k=0, sum=0.0; k<m; ++k) {
                cout<<mat_idx(i,k,m)<<" "<<mat_idx(j,k,m)<<endl;
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


void matrix_matrix_mult_tile (
    double* dst, double* src1, double* src2,
    int nr, int nc, int nq,
    int rstart, int rend, int cstart, int cend,
    int qstart, int qend)
    { /* matrix_matrix_mult_tile */
        int r, c, q;

        #pragma omp parallel for collapse(2)
        for (r = rstart; r <= rend; r++) {
            for (c = cstart; c <= cend; c++) {
                if (qstart == 0) 
                    dst[mat_idx(r,c,nc)] = 0.0;
                    //dst[r][c] = 0.0;
                for (q = qstart; q <= qend; q++) {
                    dst[mat_idx(r,c,nc)] = dst[mat_idx(r,c,nc)] + src1[mat_idx(r,q,nq)] * src2[mat_idx(q,c,nc)];
                    //dst[r][c] = dst[r][c] + src1[r][q] * src2[q][c];
                } /* for q */
            } /* for c */
        } /* for r */
    } /* matrix_matrix_mult_tile */


void matrix_matrix_mult_by_tiling (
    double** dst, double* src1, double* src2,
    int nr, int nc, int nq, //rxq qxc
    int rtilesize, int ctilesize, int qtilesize)
    { /* matrix_matrix_mult_by_tiling */
        double *C = NULL;
        C = (double*) malloc(nr*nc*sizeof(*C));
        int rstart, rend, cstart, cend, qstart, qend;
        double start, end;
        start = omp_get_wtime();
        for (rstart = 0; rstart < nr; rstart += rtilesize) {
            rend = rstart + rtilesize - 1;
            if (rend >= nr) 
                rend = nr - 1;
            for (cstart = 0; cstart < nc; cstart += ctilesize) {
                cend = cstart + ctilesize - 1;
                if (cend >= nc) 
                    cend = nc - 1;
                for (qstart = 0; qstart < nq; qstart += qtilesize) {
                    qend = qstart + qtilesize - 1;
                    if (qend >= nq) 
                        qend = nq - 1;
                        //cout<<rstart<<" "<<qstart<<" "<<cstart<<" "<<endl;
                    matrix_matrix_mult_tile(C, src1, src2, nr, nc, nq, rstart, rend, cstart, cend, qstart, qend);
                } /* for qstart */
            } /* for cstart */
        } /* for rstart */
        end = omp_get_wtime();
        printf("%f seconds\n", end - start);
        printMat(C, nr,nc);
        *dst = C;

    } /* matrix_matrix_mult_by_tiling */ 


void matrix_tranposed_matrix_mult_tile (
    double* dst, double* src1, double* src2_trans,
    int nr, int nc, int nq,
    int rstart, int rend, int cstart, int cend,
    int qstart, int qend)
    { /* matrix_matrix_mult_tile */
        int r, c, q;

        #pragma omp parallel for collapse(2)
        for (r = rstart; r <= rend; r++) {
            for (c = cstart; c <= cend; c++) {
                if (qstart == 0) 
                    dst[mat_idx(r,c,nc)] = 0.0;
                    //dst[r][c] = 0.0;
                for (q = qstart; q <= qend; q++) {
                    dst[mat_idx(r,c,nc)] = dst[mat_idx(r,c,nc)] + src1[mat_idx(r,q,nq)] * src2_trans[mat_idx(c,q,nq)];
                    //dst[r][c] = dst[r][c] + src1[r][q] * src2[q][c];
                } /* for q */
            } /* for c */
        } /* for r */
    } /* matrix_matrix_mult_tile */



void matrix_transposed_matrix_mult_by_tiling (
    double** dst, double* src1, double* src2_trans,
    int nr, int nc, int nq, //rxq qxc
    int rtilesize, int ctilesize, int qtilesize)
    { /* matrix_matrix_mult_by_tiling */
        double *C = NULL;
        C = (double*) malloc(nr*nc*sizeof(*C));
        int rstart, rend, cstart, cend, qstart, qend;
        double start, end;
        start = omp_get_wtime();
        for (rstart = 0; rstart < nr; rstart += rtilesize) {
            rend = rstart + rtilesize - 1;
            if (rend >= nr) 
                rend = nr - 1;
            for (cstart = 0; cstart < nc; cstart += ctilesize) {
                cend = cstart + ctilesize - 1;
                if (cend >= nc) 
                    cend = nc - 1;
                for (qstart = 0; qstart < nq; qstart += qtilesize) {
                    qend = qstart + qtilesize - 1;
                    if (qend >= nq) 
                        qend = nq - 1;
                        //cout<<rstart<<" "<<qstart<<" "<<cstart<<" "<<endl;
                    matrix_matrix_mult_tile(C, src1, src2_trans, nr, nc, nq, rstart, rend, cstart, cend, qstart, qend);
                } /* for qstart */
            } /* for cstart */
        } /* for rstart */
        end = omp_get_wtime();
        printf("%f seconds\n", end - start);
        printMat(C, nr,nc);
        *dst = C;
        
    } /* matrix_matrix_mult_by_tiling */ 



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
	printMat(C, n,p);
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

    /* cout<<"Parallel"<<endl;
    mult_mat(nrows, ncols, ncols2, A, B, &C);
    cout<<"Parallel w/ trans"<<endl;
    mult_mat_transposed(nrows, ncols, ncols2, A, B_trans, &C); */

    cout<<"Parallel Tiling"<<endl;
    matrix_matrix_mult_by_tiling ( &C, A, B, nrows, ncols2, ncols, 300, 300, 300);

    cout<<"Parallel Tiling Transposed"<<endl;
    matrix_matrix_mult_by_tiling ( &C, A, B_trans, nrows, ncols2, ncols, 300, 300, 300);

    cout<<"Sequential"<<endl;
    mult_mat_seq(nrows, ncols, ncols2, A, B, &C_seq);
    
    
    

    cout<<"Matrix check: "<<check_mat(nrows*ncols2, C, C_seq)<<endl;

    return 0;


}