/*
MIT License

Copyright (c) 2018 Grant Van Horn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/*
This file contains functions for computing the loglikelihood of leaf nodes in
a taxonomy.

On macOSX you can compile this file for python usage with:
$ gcc -DNDEBUG -shared -Wl,-install_name,annoprobs -o annoprobs.so -fPIC -Ofast annoprobs.c

On ubuntu you can compile this file for python usage with:
$ gcc -DNDEBUG -shared -Wl,-soname,annoprobs -o annoprobs.so -fPIC -Ofast annoprobs.c

If your gcc compiler supports OpenMP directives, then you can add the `-fopenmp` flag to the above commands, e.g.
$ gcc -DNDEBUG -shared -Wl,-soname,annoprobs -o annoprobs.so -fPIC -Ofast -fopenmp annoprobs.c

Don't forget to set the number of available threads with:
export OMP_NUM_THREADS=X

gcc argument notes:
-DNDEBUG : disable asserts
-shared : make a shared library
-Wl,option : pass the option to the linker
-fPIC : generate "position independent code" (i.e. code that can be included in a library)

*/

#include <stdlib.h>
#include <string.h> // needed for memcpy
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <float.h>

void inner_node_annotation_probs(int, int, float *, float *, float *, int *, int *, int *, int *, int *, int*, float *, int);
float compute_class_log_likelihood(int, int, int, int, int, float *, float *, float *, int *, int *, int *, int *, float *, int *, int);
void compute_log_likelihoods(int, int, int, int, float*, float *, float *, int *, int *, int *, int *, int *, int *, int*, int*, float *);

void compute_probability_of_prior_responses(int, int, int, float *, float *, float *, int *, int *, int *, int *, int *, int *, int *, int, int, float *, float *);

void inner_node_annotation_probs(int n, int l, float *M, float *N, float *P, int *R, int *S, int *parents, int *inner_nodes, int *parents_offset, int *block_starts, float *anno_probs, int mult_by_P){
    /* Compute a tensor [num inner nodes, num nodes] that contains the probability of each annotation given each inner node is correct.
    This is for a single worker. This is computing [\prod_{l=1}^{level(z)} p(z_j^l | y^l, w_j)] * p(H^{t-1} | z_j, w_j), where y is an inner node,
    which is part of the numerator for p(z_j | y_i, H^{t-1}, w_j)
    Args:
        n : number of nodes (total nodes -1 to exclude the root)
        l : number of leaf nodes in the taxonomy
        M : [scs] skill vectors for each worker (raveled block diagonal sparse matrix), this has a length equal to the sum of the children squared.
        N : [n] skill vectors for each worker (when parents don't match)
        P : [n] probability of prior annotations for each worker
        R : [n] the offset into M for each node (tells us where to index into M)
        S : [n] the number of siblings for each node (tells us how many entries to read from M)
        parents: [n] parent index for each node
        inner_nodes: [n-l] the indices of the inner nodes
        parents_offset: [n] the parent index of each node, but offset by removing all leaf nodes higher up in the taxonomy.
                            This will be needed to index into `inner_anno_probs` correctly for each leaf node.
        block_starts : [n] the column index of where the corresponding node's block start (in the block diagonal matrix)
        anno_probs: [n-l, n] Where the annotation probs will be stored.
    */

    int num_inner_nodes = n - l;
    if(num_inner_nodes == 0){
        assert(0);
        return;
    }

    // Get the number of inner node siblings at the first level
    int num_inner_nodes_in_first_block = 0;
    while(num_inner_nodes_in_first_block < num_inner_nodes && parents[inner_nodes[num_inner_nodes_in_first_block]] == -1){
        num_inner_nodes_in_first_block++;
    }

    // The number of inner nodes at level 1 should be less than or equal to all of the siblings at level 1
    assert(num_inner_nodes_in_first_block <= S[0]);

    int r,c;
    int row_node, row_parent, row_num_siblings;
    int col_parent;
    float prob;     // The value that will be stored in anno_probs
    float *row_skills;

    // Base case, fill in to the right of the first diagonal block group
    for(r = 0; r < num_inner_nodes_in_first_block; r++){

        // The user's skill matrix M is stored in a single vector.
        // The first block of skills is a matrix of size [total_num_nodes_in_first_block x total_num_nodes_in_first_block]
        row_skills = M + R[r];
        assert(R[r] == S[r] * r);
        row_num_siblings = S[r];

        // Fill in the anno probs for all nodes.
        for(c = 0; c < n; c++){

            // We are filling in sibling entries. This is the only time we should be accessing M
            if( c < row_num_siblings){
                // Copy the value over.
                assert(!isnan(*(row_skills + c)));
                *((anno_probs + r * n) + c) = *(row_skills + c);

            }
            // We are filling in entries corresponding to nodes deeper in the taxonomy.
            // So we will be multplying their parent probability (computed earlier in the loop iteration) by N[c]
            else{

                // Get the parent node to c
                col_parent = parents[c];
                assert(col_parent < c);

                // Access the prob stored at anno_probs[r][col_parent]
                prob = *((anno_probs + r * n) + col_parent);

                assert(prob < 1.0);
                assert(prob > 0);
                assert(isnan(*(N + c)) == 0);

                // Multiply the parent prob by N[c] and store in anno_probs[r][c]
                *((anno_probs + r * n) + c) = prob * ( *(N + c) );

                // if( r == 0 && c == 253){
                //     printf("p(253 | 0, w)= %0.5f\n", prob * ( *(N + c) ));
                // }

            }
        }

        // // Multiply each entry by the prior probability
        // for(c = 0; c < n; c++){
        //     *((anno_probs + r * n) + c) =  *((anno_probs + r * n) + c) * ( *(P + c) );

        // }

    }

    for(r=0; r < num_inner_nodes_in_first_block; r++){
        for(c=0; c < n; c++){
            assert(isnan(*(anno_probs + r * n + c)) == 0);
        }
    }

    int row_offset_parent, row_skills_col;//, num_to_copy;
    float * parent_row_ptr, * this_row_ptr;
    float parent_row_correct_prob;
    int blk_start;

    // Fill in the subsequent rows, using 2 memcpy operations and looks up for the blk diagonal parts
    for(r = num_inner_nodes_in_first_block; r < num_inner_nodes; r++){

        // The raw value of r could correspond to a leaf node (high up in the taxonomy). So we need to index
        // into inner_nodes to determine which row of M we should be accessing.
        row_node = inner_nodes[r];
        assert(r <= row_node);

        blk_start = block_starts[row_node];
        assert(blk_start <= row_node);

        // Get access to the parent row in anno_probs, we'll be copying most entries from the parent row to this row.
        row_parent = parents[row_node];
        row_offset_parent = parents_offset[row_node]; // the location of the parent row in anno_probs
        assert(row_offset_parent <= row_parent);
        parent_row_ptr = anno_probs + row_offset_parent * n;

        // Get access to the skills for this row
        row_skills = M + R[row_node];
        row_num_siblings = S[row_node];

        // This is where we will be storing the annotation probabilities for this row.
        this_row_ptr = anno_probs + r * n;

        // memcpy up to the blk_start position
        memcpy(this_row_ptr, parent_row_ptr, sizeof(float) * blk_start);

        // For the block diagonal part, we are going to multiply the value M[row_node][c] by the parent prob correct value anno_prob[row_offset_parent][row_parent]
        parent_row_correct_prob = *(parent_row_ptr + row_parent); // should be on the diagonal.

        // The parent_row_correct_prob value has already been multplied by the prior probability of the annotations P[row_parent]
        // We want to divide by that value to get the original value of M[row_parent][row_parent]
        //parent_row_correct_prob = parent_row_correct_prob / ( *(P + row_parent) );

        // Fill in the entry for each sibling.
        c = blk_start;
        row_skills_col = 0;
        for(row_skills_col=0; row_skills_col < row_num_siblings; row_skills_col++){
            *(this_row_ptr + c) = *(row_skills + row_skills_col) * parent_row_correct_prob; // * ( *(P + c) );
            c++;
        }

        ///////////////////////
        // GVH: OLD COMPUTATIONS for columns after the block
        // memcpy from c to the end of the array
        //num_to_copy = n - c;
        //assert(num_to_copy > 0); // there should definitely be leaf nodes remaining.
        //memcpy(this_row_ptr + c, parent_row_ptr + c, sizeof(float) * num_to_copy);
        //////////////////////

        ///////////////////////////
        // GVH: NEW COMPUTATIONS for columns after the block
        for (; c < n; c++){
            // Get the parent node to c
            col_parent = parents[c];
            assert(col_parent < c);

            // Access the prob stored at anno_probs[r][col_parent]
            prob = *((anno_probs + r * n) + col_parent);

            assert(prob < 1.0);
            assert(prob > 0);
            assert(isnan(*(N + c)) == 0);

            // Multiply the parent prob by N[c] and store in anno_probs[r][c]
            *((anno_probs + r * n) + c) = prob * ( *(N + c) );
        }
        /////////////////////////

    }

    if (mult_by_P > 0){
        // Multiply each entry by the prior probability
        for (r = 0; r < num_inner_nodes; r++){
            for(c = 0; c < n; c++){
                *((anno_probs + r * n) + c) =  *((anno_probs + r * n) + c) * ( *(P + c) );
            }
        }
    }
}

float compute_class_log_likelihood(int y, int w, int n, int l, int scs, float *M, float *N, float *P, int *R, int *S, int *parents, int *parents_offset, float *inner_anno_probs, int *Z, int normalize){
    /* Compute a value representing the log likelihood of a specific class label given worker annotations and skills.
    We are computing the numerator for the value log( p(y | Z) )
    Args:
        y : class index
        w : number of workers
        n : number of nodes (total nodes -1 to exclude the root)
        l : number of leaf nodes in the taxonomy
        scs : sum of the children squared
        M : [w, scs] skill vectors for each worker (raveled block diagonal sparse matrix)
        N : [w, n] skill vectors for each worker (when parents don't match)
        P : [w, n] probability of prior annotations for each worker
        R : [n] the offset into M for each node (tells us where to index into M)
        S : [n] the number of siblings for each node (tells us how many entries to read from M)
        parents: [n] parent index for each node
        parents_offset: [n] the parent index of each node, but offset by removing all leaf nodes higher up in the taxonomy.
                            This will be needed to index into `inner_anno_probs` correctly for each leaf node.
        inner_anno_probs: [w, n-l, n] Annotation probabilities of inner nodes.
        Z: [w] Worker annotations.
    */

    int num_inner_nodes = n - l;

    float num = 0; // holds the numerator, this will be a sum of logs
    float denom = 0; // holds the denominator, this will be a sum of logs (of a sum)
    float worker_denom; // holds the rolling sum for the denominator of each worker.

    float * anno_probs;
    float * worker_N, *worker_P;
    float * row_skills;
    int row_skills_col;
    float prob;
    float * parent_row_ptr;
    float parent_row_correct_prob;

    // Get the parent node to the class label
    int row_offset_parent = parents_offset[y];
    int row_parent = parents[y];
    int row_num_siblings = S[y];

    int col_parent;

    // When computing the likelihood of this class, we'll copy our parent's likelihood
    // and we'll have 3 groups:
    // 1) A block before the sibling block that will be just the values copied from the parent
    // 2) The sibling block, which will be the parent's values multiplied by M[y][num_siblings]
    // 3) A block after the sibling block that will be just the values copied from the parent

    // If the class node is a child of the root node, then there is no entry in the inner_anno_probs matrix
    // and we'll need to compute the values here...
    // float *temp_class_anno_prob;
    // if(row_parent == -1){
    //     temp_class_anno_prob = malloc(n * sizeof(temp_class_anno_prob));
    // }

    float *temp_class_anno_prob = malloc(n * sizeof(temp_class_anno_prob));

    int widx, c;
    for(widx=0; widx < w; widx++){

        //printf("HERE 4\n"); fflush(stdout);

        // Index into M to get the class skill for this worker
        row_skills = M + widx * scs + R[y];

        // Index into N (probability of a worker choosing a node)
        worker_N = N + widx * n;

        // Index into the probability of prior responses
        worker_P = P + widx * n;
        worker_denom = 0;

        // There is no entry in the inner_anno_probs matrix
        // This class is a child of the root node
        if(row_parent == -1){

            //printf("HERE 5\n"); fflush(stdout);


            for(c = 0; c < n; c++){

                // We are filling in sibling entries. This is the only time we should be accessing M
                if( c < row_num_siblings){
                    // copy the value over
                    *(temp_class_anno_prob + c) = *(row_skills + c);
                }
                // We are filling in entries corresponding to nodes deeper in the taxonomy.
                // So we will be multiplying their parent probability (computed earlier in the loop iteration) by N[c]
                else{

                    // Get the parent node to c
                    col_parent = parents[c];
                    assert(col_parent < c);

                    // Access the prob stored at anno_probs[r][col_parent]
                    prob = *(temp_class_anno_prob + col_parent);

                    // Multiply the parent prob by N[c] and store in anno_probs[r][c]
                    *(temp_class_anno_prob + c) = prob * ( *(worker_N + c) );

                }
            }



            // // Multiply the probability of prior annotations and update the numerator and denominator
            // for(c = 0; c < n; c++){

            //     prob =  *(temp_class_anno_prob + c) * ( *(worker_P + c) );
            //     if(c == Z[widx]){
            //         num = num + log(prob); // this is the worker's annotation, so add it to the numerator
            //     }
            //     worker_denom = worker_denom + prob;
            // }

        }
        else{

            //printf("HERE 6\n"); fflush(stdout);
            //printf("HERE 6.1: %d, %d\n", row_offset_parent, row_parent); fflush(stdout);

            // Get the worker's annotation probability matrix
            anno_probs = inner_anno_probs + widx * num_inner_nodes * n;

            // Get the anno prob row corresponding to the class label's parent
            parent_row_ptr = anno_probs + row_offset_parent * n;



            // For the block diagonal part, we are going to multiply the value M[row_node][c] by the parent prob correct value anno_prob[row_offset_parent][row_parent]
            parent_row_correct_prob = *(parent_row_ptr + row_parent); // should be on the diagonal.

            // The parent_row_correct_prob value has already been multplied by the prior probability of the annotations P[row_parent]
            // We want to divide by that value to get the original value of M[row_parent][row_parent]
            //parent_row_correct_prob = parent_row_correct_prob / ( *(worker_P + row_parent) );
            //printf("HERE 6.1\n"); fflush(stdout);
            row_skills_col = 0;
            for(c=0; c < n; c++){
                // siblings, multiply the skill by the parent's probability
                //printf("HERE 6.2\n"); fflush(stdout);
                if(parents[c] == row_parent){
                    //printf("HERE 6.3\n"); fflush(stdout);
                    *(temp_class_anno_prob + c) = ( *(row_skills + row_skills_col) ) * parent_row_correct_prob; // * ( *(worker_P + c) );
                    row_skills_col++;
                }
                // non-siblings
                else{
                    //printf("HERE 6.4\n"); fflush(stdout);
                    // We are still before the block portion of this row, so just take the parent value
                    if (row_skills_col == 0){
                        *(temp_class_anno_prob + c) = *(parent_row_ptr + c);
                    }
                    else{

                        // We are after the block portion of this row, so grab the parent value of this column and multiply by N[c]
                        assert(row_skills_col == row_num_siblings);


                        // Get the parent node to c
                        col_parent = parents[c];
                        assert(col_parent < c);

                        // Access the prob stored at anno_probs[r][col_parent]
                        prob = *(temp_class_anno_prob + col_parent);

                        // Multiply the parent prob by N[c] and store in anno_probs[r][c]
                        *(temp_class_anno_prob + c) = prob * ( *(worker_N + c) );

                    }

                }
                //printf("HERE 6.5\n"); fflush(stdout);

                // if( c == Z[widx] ){
                //     num = num + log(prob); // this is the worker's annotation, so add it to the numerator
                // }
                // worker_denom = worker_denom + prob;

                // if(y == 253 && c == 253){
                //     printf("GT prob = %0.5f\n", prob);
                // }

                // //if(y == 253 && c == 35){
                // //    printf("prob = %0.5f\n", prob);
                // //}

                // if(y == 35 && c == 253){
                //     printf("Comp prob = %0.5f\n", prob);
                //     printf("Comp parent = %0.5f\n", parent_row_correct_prob);
                // }

            }
            assert(row_skills_col == row_num_siblings);
            //printf("HERE 6.9\n"); fflush(stdout);

        }

        //printf("HERE 7\n"); fflush(stdout);

        // Multiply the probability of prior annotations and update the numerator and denominator
        for(c = 0; c < n; c++){

            prob =  *(temp_class_anno_prob + c) * ( *(worker_P + c) );
            if(c == Z[widx]){
                num = num + log(prob); // this is the worker's annotation, so add it to the numerator
            }
            worker_denom = worker_denom + prob;

            // if(y == 253 && c == 253){
            //     printf("GT prob = %0.5f\n", prob);
            // }

            // //if(y == 253 && c == 35){
            // //    printf("prob = %0.5f\n", prob);
            // //}

            // if(y == 35 && c == 253){
            //     printf("Comp prob = %0.5f\n", prob);
            //     //printf("Comp parent = %0.5f\n", parent_row_correct_prob);
            // }

        }
        //printf("HERE 8\n"); fflush(stdout);

        denom = denom + log(worker_denom);

    }

    // If this node is a child of root, then we malloced a temp array
    // if(row_parent == -1){
    //     free(temp_class_anno_prob);
    // }
    free(temp_class_anno_prob);

    float lly = num;
    if (normalize > 0){
        lly -= denom; // working in logs, so we can subtract the two.
    }

    return lly;
}

void compute_log_likelihoods(int w, int n, int l, int scs, float *M, float *N, float *P, int *R, int *S, int *parents, int *inner_nodes, int *leaf_nodes, int *parents_offset, int *block_starts, int*Z, float *lls){
    /* Compute the log likelihoods (not including the class prior) for each leaf node. Returns a vector of
    log likelihood values for each leaf node.
    Args:
        w : number of workers
        n : number of nodes (total nodes -1 to exclude the root)
        l : number of leaf nodes in the taxonomy
        scs : sum of the children squared, this is the number of columns in M
        M : [w, scs] skill vectors for each worker (raveled block diagonal sparse matrix)
        N : [w, n] skill vectors for each worker (when parents don't match)
        P : [w, n] probability of prior annotations for each worker
        R : [n] the offset into M for each node (tells us where to index into M)
        S : [n] the number of siblings for each node (tells us how many entries to read from M)
        parents: [n] parent index for each node
        inner_nodes: [n-l] the indices of the inner nodes
        leaf_nodes: [l] the leaf node indices
        parents_offset: [n] the parent index of each node, but offset by removing all leaf nodes higher up in the taxonomy.
                            This will be needed to index into `inner_anno_probs` correctly for each leaf node.
        block_starts: [n] the column index of where the corresponding node's block start (in the block diagonal matrix)
        Z : [w] worker annotations
        lls : [l] output vector that will store the log likelihood of each leaf node.
    */

    // Make a tensor of shape [w, n - l, n] that will holds the probability of the annotations for the innner nodes.
    int num_inner_nodes = n - l;
    float *inner_anno_probs;
    if(num_inner_nodes > 0){
        inner_anno_probs = malloc(w * num_inner_nodes * n * sizeof(*inner_anno_probs));

        // This can be done in parallel for each worker
        int widx;
        #pragma omp parallel for
        for(widx = 0; widx < w; widx++){
            float *worker_inner_anno_probs = inner_anno_probs + widx * num_inner_nodes * n;
            int mult_by_P = 0;
            inner_node_annotation_probs(n, l, M + widx * scs, N + widx * n, P + widx * n, R, S, parents, inner_nodes, parents_offset, block_starts, worker_inner_anno_probs, mult_by_P);
        }
    }
    else{
        // This is a "flat taxonomy" where the root node is connected to all leaf nodes.
        // We'll assign `inner_anno_probs` to M, but it is not actually used.
        inner_anno_probs = M;
    }


    // Ensure that no NANs were computed for the inner nodes.
    int widx;
    int r, c;
    for(widx=0; widx < w; widx++){
        for(r=0; r < num_inner_nodes; r++){
            for(c=0; c < n; c++){
                float v = *(inner_anno_probs + widx * num_inner_nodes * n + r * n + c);
                assert(isnan(v) == 0);
            }
        }
    }

    // Compute the likelihood of each class. This can be done in parallel for each class.
    int y;
    #pragma omp parallel for
    for(y=0; y < l; y++){
        int normalize = 1;
        *(lls + y) = compute_class_log_likelihood(leaf_nodes[y], w, n, l, scs, M, N, P, R, S, parents, parents_offset, inner_anno_probs, Z, normalize);
    }

    // Free up the memory that we malloced.
    if(num_inner_nodes > 0){
        free(inner_anno_probs);
    }
}


void compute_probability_of_prior_responses(int n, int l, int scs, float *M, float *N, float *P, int *R, int *S, int *parents, int *inner_nodes, int *leaf_nodes, int *parents_offset, int *block_starts, int z_prev, int normalize, float *prp_inner, float *prp_leaf){
    /* Compute the probability of prior responses p(H_i^{t-1} | z, w_j) for each possible response z. Returns a vector of probabilities.
    Args:
        n : number of nodes (total nodes -1 to exclude the root)
        l : number of leaf nodes in the taxonomy
        scs : sum of the children squared, this is the number of columns in M
        M : [scs] skill vectors for a worker (raveled block diagonal sparse matrix)
        N : [n] skill vectors for a worker (when parents don't match)
        P : [n] p(H_i^{t-2} | z, w_j^{t-1}) probability of prior responses for the previous worker
        R : [n] the offset into M for each node (tells us where to index into M)
        S : [n] the number of siblings for each node (tells us how many entries to read from M)
        parents: [n] parent index for each node
        inner_nodes: [n-l] the indices of the inner nodes
        leaf_nodes: [l] the leaf node indices
        parents_offset: [n] the parent index of each node, but offset by removing all leaf nodes higher up in the taxonomy.
                            This will be needed to index into `inner_anno_probs` correctly for each leaf node.
        block_starts: [n] the column index of where the corresponding node's block start (in the block diagonal matrix)

        z_prev : previous worker's annotation
        prp_inner : [n - l] output vector that will store p(H_i^{t-1} | z, w_j) for each possible inner node response z.
        prp_leaf : [l] output vector that will store p(H_i^{t-1} | z, w_j) for each possible leaf node response z.
    */

    // Make a tensor of shape [n - l, n] that will holds the probability of the annotations for the innner nodes.
    int num_inner_nodes = n - l;
    float *inner_anno_probs;
    if(num_inner_nodes > 0){
        inner_anno_probs = malloc(num_inner_nodes * n * sizeof(*inner_anno_probs));

        //printf("HERE 1\n"); fflush(stdout);

        int mult_by_P = 1;
        inner_node_annotation_probs(n, l, M, N, P, R, S, parents, inner_nodes, parents_offset, block_starts, inner_anno_probs, mult_by_P);

        //printf("HERE 2\n"); fflush(stdout);

        // We now have a matrix of shape [num inner nodes, n] storing the probability of p(z_j | z, w_j) * p(H^{t-1} | z_j, w_j)
        // To compute the probability of the prior response for each inner node z:
        // The numerator is inner_anno_probs[z][z_prev]
        // The denominator is the sum of inner_anno_probs[z]

        // Go through each inner node
        int i;
        for(i=0; i < num_inner_nodes; i++){

            float num = *(inner_anno_probs + i * n + z_prev);
            if (normalize > 0){
                float denom = 0;
                int c;
                for(c = 0; c < n; c++){
                    denom += *(inner_anno_probs + i * n + c);
                }

                // Store the probability of the previous responses given this inner node response
                prp_inner[i] = num / denom;
            }
            else{
                prp_inner[i] = num;
            }

        }

        //printf("HERE 3\n"); fflush(stdout);

    }
    else{
        // This is a "flat taxonomy" where the root node is connected to all leaf nodes.
        // We'll assign `inner_anno_probs` to M, but it is not actually used.
        inner_anno_probs = M;
    }

    int y;
    #pragma omp parallel for
    for(y=0; y < l; y++){
        int Z[1] = {z_prev};
        *(prp_leaf + y) = exp(compute_class_log_likelihood(leaf_nodes[y], 1, n, l, scs, M, N, P, R, S, parents, parents_offset, inner_anno_probs, Z, normalize));
    }

    // Free up the memory that we malloced.
    if(num_inner_nodes > 0){
        free(inner_anno_probs);
    }

}


int main(int argc, char **argv){
    /* Basic test case to make sure things are working.
    */


    int w = 2; // number of workers
    int n = 4; // number of nodes (not including the root)
    int l = 3; // number of leaf nodes
    int scs = 8; // "sum of the children squared" -- number of columns in the M matrix

    // The M matrix is a raveled block diagonal matrix, each block corresponds to a skill matrix for the children of a node
    float M[2][8] = {
        { 0.93333334, 0.022,
          0.044 , 0.93333334,
                             0.60000002, 0.13200001,
                             0.13200001, 0.60000002},
        { 0.60000002, 0.13200001,
          0.26400003, 0.60000002,
                             0.89999998, 0.033,
                             0.033     , 0.89999998}
    };

    // The N matrix contains a row for each worker that contains the likelihood of the worker selecting a particular node.
    float N[2][4] = {
        {0.044, 0.022, 0.132, 0.132},
        {0.264, 0.132, 0.033, 0.033}
    };

    // The P matix contains the prior probabilites of selecting a node given what previous annotators have selected.
    float P[2][4] = {
        {1.0, 1.0, 1.0, 1.0},
        {0.1, 0.05, 0.65, 0.2}
    };

    // R contains the offset into M for a node, the offset corresponds to where that node's "row" starts in M
    int R[4] = {0, 2, 4, 6};

    // S contains the number of siblings for each node (including the node, so its simply the number of children under the parent of that node)
    int S[4] = {2, 2, 2, 2};

    // parents contains the index of the parent node in the previous data stuctures
    // -1 signifies that the parent is the root node (which is not represented in the data structures)
    int parents[4] = {-1, -1, 0, 0};

    // A list of node indices corresponding to inner nodes
    int inner_nodes[1] = {0};

    // A list of node indices corresponding to leaf nodes
    int leaf_nodes[3] = {1, 2, 3};

    // parent indices, but this time when we don't include leaf nodes. This is used to index
    // into an intermediate data structure that contains only inner node probabilities.
    int parents_offset[4] = {-1, -1, 0, 0};

    // block starts contains the column index of where this node's block starts in the block diagonal matrix
    int block_starts[4] = {0, 0, 2, 2};

    // These are the worker labels, note that 0 corresponds to the root node's first child, NOT to the root node itself.
    int Z[2] = {2, 2};

    // This list will be filled in with the log likelihood of each leaf node
    float lls[3];

    // Compute the log likelihoods of the leaf nodes
    compute_log_likelihoods(w, n, l, scs, (float *)M, (float *)N, (float *)P, (int *)R, (int *)S, (int *)parents, (int *)inner_nodes, (int *)leaf_nodes, (int *)parents_offset, (int *)block_starts, (int*)Z, (float *)lls);

    // Print out the log likelihoods
    int c=0;
    for(c=0; c < l; c++){
        printf("Leaf node %d: %f\n", c, lls[c]);
    }

    // This should produce:
    // 0: -5.137407
    // 1: -1.073620
    // 2: -2.587748

    float expected_lls[3] = {-7.559322, -1.256796, -5.266468};
    for(c=0; c < l; c++){
       float diff = fabsf(expected_lls[c] - lls[c]);
       if ((diff / FLT_MAX) > 0.00000001){
           printf("ERROR: computed & expected log likelihood values differ %f != %f", lls[c], expected_lls[c]);
       }
    }

    return 0;
}