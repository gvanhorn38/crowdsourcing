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

//////////////////////////////
Compilation Notes

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

//////////////////////////////

//////////////////////////////
Implementation Notes

See drawing notes here: https://docs.google.com/drawings/d/1grFfTIL7XqrPuyedWOY4VkMcD19EU7yCJWJsVH4h6qQ/edit?usp=sharing

These functions assume we are working with a taxonomy composed of n + 1 nodes (1 extra for the root node).
The taxonomy has been arranged such that the children of each node are sorted by the maximum depth of their descendants,
deepest first. Further, the nodes are numbered in breadth first order (left to right) such that the first child
of the root node (which has the deepest descendants) is labeled as 0.

Several data structures need to be precomputed to help speed up the computations:

A parent index of -1 signifies that the parent for this node is the root node.

Notes on variables:

n : the number of nodes (not including the root node)
l : the number of leaf nodes
w : the number of workers
scs: "sum children squared" -- for each inner node we square the number of children and add

//////////////////////////////

*/

#include <stdlib.h>
#include <string.h> // needed for memcpy
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <float.h>


void prob_of_anno_given_inner_node(int, int, float *, float *, int *, int *, int *, int *, int *, int *, float *);
void prob_of_anno_given_leaf_node(int, int, int, float *, float *, int *, int *, int *, int *, int *, float *, float *);
void compute_log_likelihoods(int, int, int, int, float *, float *, float *, int *, int *, int *, int *, int *, int *, int *, int*, float *);
float compute_class_log_likelihood(int, int, int, int, int, float *, float *, float *, int *, int *, int *, int *, int *, float *, int *);
void compute_probability_of_prior_responses(int n, int l, int scs, float *, float *, float *, int *, int *, int *, int *, int *, int *, int *, int, int, float *, float *);


void prob_of_anno_given_inner_node(int n, int l, float *M, float *N, int *R, int *S, int *parents, int *inner_nodes, int *parents_offset, int *block_starts, float *anno_probs){
    /* Compute a tensor of shape [n - l, n] that contains the probability of each annotation given each inner node is correct.
    This is for a single worker. This is computing p(z | y, w) = \prod_{l=1}^{level(z)} p(z_j^l | y^l, w_j)], where y is an inner node.

    Args:
        n : number of nodes (total nodes -1 to exclude the root)
        l : number of leaf nodes in the taxonomy
        M : [scs] A raveled block diagonal sparse matrix, containing the skills of the worker across the taxonomy. This has a length equal to the sum of the children squared.
        N : [n] A vector containing the probability this worker will select a specific node.
        R : [n] A vector containing the offset into M for each node. This tells us where to index into M to get the skills for a particular node.
        S : [n] A vector containing the number of siblings for each node. This tells us how many entries to read from M when accessing the skills for a node.
        parents: [n] A vector containing the parent index for each node.
        inner_nodes: [n-l] A vector containing the indices that correspond to the inner nodes
        parents_offset: [n] A vector containing the parent index of each node, but offset by removing all leaf nodes higher up in the taxonomy.
                            This is used to index into `inner_anno_probs` correctly for each leaf node.
        block_starts : [n] A vector containing the column index of where a node's "block" would start in the big block diagonal matrix of shape [n, n].
                            Since we are not creating the big [n, n] matrix, but rather a [n-l, n] matrix, we need this data structure to index properly
                            into the anno_probs matrix.
        anno_probs: [n-l, n] Where the annotation probs will be stored.
    Returns:
        void : the computations are stored in anno_probs, which should be malloced by the caller
    */

    int num_inner_nodes = n - l;

    // If there are no inner nodes, then there is nothing to compute.
    if(num_inner_nodes == 0){
        return;
    }

    // Get the number of inner node siblings at the first level (excluding leaf node siblings)
    int num_inner_nodes_in_first_block = 0;
    // Go through the inner nodes and keep incrementing while the parent of the inner node is -1 (i.e. the root node)
    while(num_inner_nodes_in_first_block < num_inner_nodes && parents[inner_nodes[num_inner_nodes_in_first_block]] == -1){
        num_inner_nodes_in_first_block++;
    }

    // The number of inner nodes at level 1 should be less than or equal to all of the siblings at level 1 (including leaf node siblings)
    assert(num_inner_nodes_in_first_block <= S[0]);

    int r,c;             // loop variables for row and column
    int row_num_siblings;// Keep track of the number of siblings in the current row
    int col_parent;      // Parent index of the current row
    float prob;          // The value that will be stored in anno_probs
    float *row_skills;   // Pointer into the M vector

    // Base case: Fill in the first "block" along with all columns to the right of the first "block"
    for(r = 0; r < num_inner_nodes_in_first_block; r++){

        // The user's skill matrix M is stored in a single vector.
        // The first block of skills is a matrix of size [total_num_nodes_in_first_block x total_num_nodes_in_first_block]
        row_skills = M + R[r];
        assert(R[r] == S[r] * r); // Sanity check: the row offset for this fist block is simply r * num_siblings_in_the_first_block
        row_num_siblings = S[r];

        // Fill in the anno probs for all nodes.
        for(c = 0; c < n; c++){

            // We are filling in sibling entries. This is the only time we should be accessing M
            if( c < row_num_siblings){
                // This is the first "level" so simply copy the value over from M.
                assert(!isnan(*(row_skills + c))); // Sanity check the skill value
                *((anno_probs + r * n) + c) = *(row_skills + c);
            }
            // We are filling in entries corresponding to nodes deeper in the taxonomy.
            // So we will be multplying their parent probability (computed earlier in the loop iteration) by N[c]
            else{

                // Get the parent node to c
                col_parent = parents[c];
                assert(col_parent < c); // the parent index should always be less than

                // Access the prob stored at anno_probs[r][col_parent]
                prob = *((anno_probs + r * n) + col_parent);

                // Sanity check the probability
                assert(prob < 1.0);
                assert(prob > 0);
                assert(isnan(*(N + c)) == 0); // Sanity check the probability of the worker choosing this node.

                // Multiply the parent prob by N[c] and store in anno_probs[r][c]
                *((anno_probs + r * n) + c) = prob * ( *(N + c) );

            }
        }

    }

    int row_offset_parent, row_skills_col;
    int row_node, row_parent;
    float * parent_row_ptr, * this_row_ptr;
    float parent_row_correct_prob;
    int blk_start;

    // Fill in the subsequent rows using:
    //   A memcpy operation for the columns to the left of the "block"
    //   Parent look up in the parent row for the block
    //   Parent look up in the current row for the columns to the right of the "block"
    for(r = num_inner_nodes_in_first_block; r < num_inner_nodes; r++){

        // The actual value of r could correspond to a leaf node (high up in the taxonomy). So we need to index
        // into inner_nodes to determine which row of M we should be accessing.
        row_node = inner_nodes[r];
        assert(r <= row_node);

        // This lets us know how many columns of the parent row we can memcpy
        blk_start = block_starts[row_node];
        assert(blk_start <= row_node); // the block start should never be bigger than the row node index (which would correspond to the diagonal)

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

        // Fill in the entry for each sibling.
        c = blk_start;
        row_skills_col = 0;
        for(row_skills_col=0; row_skills_col < row_num_siblings; row_skills_col++){
            *(this_row_ptr + c) = *(row_skills + row_skills_col) * parent_row_correct_prob;
            c++;
        }

        // Fill in the columns after the block
        for (; c < n; c++){
            // Get the parent node to c
            col_parent = parents[c];
            assert(col_parent < c);

            // Access the parent prob stored at anno_probs[r][col_parent]
            prob = *(this_row_ptr + col_parent);

            assert(prob < 1.0);
            assert(prob > 0);
            assert(isnan(*(N + c)) == 0);

            // Multiply the parent prob by N[c] and store in anno_probs[r][c]
            *(this_row_ptr + c) = prob * ( *(N + c) );
        }

    }

    // Sanity check that all computed values are safe
    for(r=0; r < num_inner_nodes; r++){
        for(c=0; c < n; c++){
            assert(isnan(*(anno_probs + r * n + c)) == 0);
        }
    }

}


void prob_of_anno_given_leaf_node(int y, int n, int l, float *M, float *N, int *R, int *S, int *parents, int *parents_offset, int *block_starts, float *inner_anno_probs, float *leaf_anno_prob){
    /* Compute p(z | y, w) for each possible z. Stores these probabilities in a vector of size [n]. This is for a single worker
    Args:
        y : class index
        n : number of nodes (total nodes -1 to exclude the root)
        l : number of leaf nodes in the taxonomy
        M : [scs] A raveled block diagonal sparse matrix, containing the skills of the worker across the taxonomy.
        N : [n] A vector containing the probability the worker will select a specific node.
        R : [n] A vector containing the offset into M for each node. This tells us where to index into M to get the skills for a particular node.
        S : [n] A vector containing the number of siblings for each node. This tells us how many entries to read from M when accessing the skills for a node.
        parents: [n] A vector containing the parent index for each node.
        parents_offset: [n] A vector containing the parent index of each node, but offset by removing all leaf nodes higher up in the taxonomy.
                            This is used to index into `inner_anno_probs` correctly for each leaf node.
        block_starts : [n] A vector containing the column index of where a node's "block" would start in the big block diagonal matrix of shape [n, n].
                            Since we are not creating the big [n, n] matrix, but rather a [n-l, n] matrix, we need this data structure to index properly
                            into the anno_probs matrix.
        inner_anno_probs: [w, n-l, n] For each worker, provides the probability of an annotation response given an inner node.
        leaf_anno_prob : [n] A vector containing the p(z | y, w) for each z

    Returns:
        void: the computations are stored in leaf_anno_prob, which should be malloced by the caller
    */


    int c; // loop variable for the column
    int row_parent = parents[y];
    int row_num_siblings = S[y];
    int col_parent;

    // Index into M to get the class skill for this worker
    float *row_skills = M + R[y];
    float prob;

    // There is no entry in the inner_anno_probs matrix
    // This class is a child of the root node
    if(row_parent == -1){

        for(c = 0; c < n; c++){

            // We are filling in sibling entries. This is the only time we should be accessing M
            if( c < row_num_siblings){
                // copy the value over
                *(leaf_anno_prob + c) = *(row_skills + c);
            }
            // We are filling in entries corresponding to nodes deeper in the taxonomy.
            // So we will be multiplying their parent probability (computed earlier in the loop iteration) by N[c]
            else{

                // Get the parent node to c
                col_parent = parents[c];
                assert(col_parent < c);

                // Access the prob stored at anno_probs[r][col_parent]
                prob = *(leaf_anno_prob + col_parent);

                // Multiply the parent prob by N[c] and store in anno_probs[r][c]
                *(leaf_anno_prob + c) = prob * ( *(N + c) );

            }
        }

    }
    else{

        // The computation of p(z |y, w) follows from it's parent's values in inner_anno_probs

        // The row index of y's parent in the inner_anno_probs matrix
        int row_offset_parent = parents_offset[y];
        assert(row_offset_parent <= row_parent);

        // Get access to the parent row in inner_anno_probs, we'll be copying most entries from the parent row to this row.
        float * parent_row_ptr = inner_anno_probs + row_offset_parent * n;

        // This lets us know how many columns of the parent row we can memcpy
        int blk_start = block_starts[y];
        assert(blk_start <= y); // the block start should never be bigger than the row node index (which would correspond to the diagonal)

        // memcpy up to the blk_start position
        memcpy(leaf_anno_prob, parent_row_ptr, sizeof(float) * blk_start);

        // For the block diagonal part, we are going to multiply the value M[row_node][c] by the parent prob correct value anno_prob[row_offset_parent][row_parent]
        float parent_row_correct_prob = *(parent_row_ptr + row_parent); // should be on the diagonal.

        // Fill in the entry for each sibling.
        c = blk_start;
        int row_skills_col;
        for(row_skills_col=0; row_skills_col < row_num_siblings; row_skills_col++){
            *(leaf_anno_prob + c) = *(row_skills + row_skills_col) * parent_row_correct_prob;
            c++;
        }

        // Fill in the columns after the block
        for (; c < n; c++){
            // Get the parent node to c
            col_parent = parents[c];
            assert(col_parent < c);

            // Access the parent prob stored at anno_probs[r][col_parent]
            prob = *(leaf_anno_prob + col_parent);

            assert(prob < 1.0);
            assert(prob > 0);
            assert(isnan(*(N + c)) == 0);

            // Multiply the parent prob by N[c] and store in anno_probs[r][c]
            *(leaf_anno_prob + c) = prob * ( *(N + c) );
        }

    }

}


float compute_class_log_likelihood(int y, int w, int n, int l, int scs, float *M, float *N, float *P, int *R, int *S, int *parents, int *parents_offset, int *block_starts, float *inner_anno_probs, int *Z){
    /* Compute a value representing the log likelihood of a specific class label given worker annotations and skills.
    This assumes we are in the "argmax" setting so we don't care about the denominator for p(y | Z).
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

    // malloc a tempory vector to store the probability of annotations for a worker
    // we will reuse this for each worker
    float * worker_anno_probs = malloc(n * sizeof(worker_anno_probs));

    float lly = 0; // log likelihood of y (except we don't have the class prior)

    int c;
    int widx;
    for( widx=0; widx < w; widx++){

        float * worker_M = M + widx * scs;
        float * worker_N = N + widx * n;
        float * worker_P = P + widx * n;
        float * worker_inner_anno_probs = inner_anno_probs + widx * num_inner_nodes * n;

        // worker_anno_probs will store p(z | y, w) for all z. It has shape [n]
        prob_of_anno_given_leaf_node(y, n, l, worker_M, worker_N, R, S, parents, parents_offset, block_starts, worker_inner_anno_probs, worker_anno_probs);

        // compute p(z | y, w) * p(H^{t-1} | z, w)
        float denom_sum = 0;
        for(c = 0; c < n; c++){
              float prob = *(worker_anno_probs + c) * ( *(worker_P + c) );
              denom_sum += prob;
              if(c == Z[widx]){
                  lly += log(prob);
              }
        }
        // If this is the "first" worker, then we simply keep p(z | y, w)  [assume p(H^{t-1} | z, w) = 1]
        // otherwise we subtract off the numerator
        if (widx > 0){
            lly -= log(denom_sum);
        }
    }

    free(worker_anno_probs);

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

            float * worker_M = M + widx * scs;
            float * worker_N = N + widx * n;
            float *worker_inner_anno_probs = inner_anno_probs + widx * num_inner_nodes * n;

            prob_of_anno_given_inner_node(n, l, worker_M, worker_N, R, S, parents, inner_nodes, parents_offset, block_starts, worker_inner_anno_probs);
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
        *(lls + y) = compute_class_log_likelihood(leaf_nodes[y], w, n, l, scs, M, N, P, R, S, parents, parents_offset, block_starts, inner_anno_probs, Z);
    }

    // Free up the memory that we malloced.
    if(num_inner_nodes > 0){
        free(inner_anno_probs);
    }
}

void compute_probability_of_prior_responses(int n, int l, int scs, float *M, float *N, float *P, int *R, int *S, int *parents, int *inner_nodes, int *leaf_nodes, int *parents_offset, int *block_starts, int z_prev, int normalize, float *prp_inner, float *prp_leaf){
    /* Compute the probability of prior responses p(H_i^{t-1} | z, w_j) for each possible response z. Returns a vector of probabilities. This is for a single worker.
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
        normalize : whether to divide or not (the base case has no division, and the prior response is assumed to be 1 for all z)
        prp_inner : [n - l] output vector that will store p(H_i^{t-1} | z, w_j) for each possible inner node response z.
        prp_leaf : [l] output vector that will store p(H_i^{t-1} | z, w_j) for each possible leaf node response z.
    */

    // Make a tensor of shape [n - l, n] that will holds the probability of the annotations for the innner nodes.
    int num_inner_nodes = n - l;
    float *inner_anno_probs;
    if(num_inner_nodes > 0){

        inner_anno_probs = malloc(num_inner_nodes * n * sizeof(*inner_anno_probs));

        // p(z_j | z, w_j) where z is inner node. It has shape [n]
        prob_of_anno_given_inner_node(n, l, M, N, R, S, parents, inner_nodes, parents_offset, block_starts, inner_anno_probs);

    }
    else{
        // This is a "flat taxonomy" where the root node is connected to all leaf nodes.
        // We'll assign `inner_anno_probs` to M, but it is not actually used.
        inner_anno_probs = M;
    }

    int y;
    #pragma omp parallel for
    for(y=0; y < l; y++){

        float * leaf_prob = malloc(n * sizeof(*leaf_prob));

        // leaf_prob will store p(z | y, w) for all z. It has shape [n]
        prob_of_anno_given_leaf_node(leaf_nodes[y], n, l, M, N, R, S, parents, parents_offset, block_starts, inner_anno_probs, leaf_prob);

        // scale each column by the probability of the prior response so that we have p(z | y, w_j) * p(H^{t-1} | z, w_j^{t-1})
        int c;
        for(c = 0; c < n; c++){

            // if (y == 49 && c == 104){
            //     printf("p(z=104 | z_j=49, w_j) * p(H^t-2 | z=104, w_j) = %0.5f * %0.5f\n", *(leaf_prob + c), * (P + c));
            // }
            // if (y == 49 && c == 105){
            //     printf("p(z=105 | z_j=49, w_j) * p(H^t-2 | z=105, w_j) = %0.5f * %0.5f\n", *(leaf_prob + c), * (P + c));
            // }
            // if (y == 49 && c == 106){
            //     printf("p(z=106 | z_j=49, w_j) * p(H^t-2 | z=106, w_j) = %0.5f * %0.5f\n", *(leaf_prob + c), * (P + c));
            // }
            // if (y == 49 && c == 107){
            //     printf("p(z=107 | z_j=49, w_j) * p(H^t-2 | z=107, w_j) = %0.5f * %0.5f\n", *(leaf_prob + c), * (P + c));
            // }
            *(leaf_prob + c) = ( *(leaf_prob + c) ) * ( * (P + c) );

        }

        float prob_prior_response = *( leaf_prob + z_prev);
        if( normalize > 0){
            float denom = 0;
            for(c = 0; c < n; c++){
                denom += *(leaf_prob + c);
            }
            prob_prior_response /= denom;
        }

        *(prp_leaf + y) = prob_prior_response;

        free(leaf_prob);

    }

    // Now that we have finished computing the probability of the leaf nodes, we can do the same for the inner nodes.
    // We do this second so that we can scale the inner_anno_probs by the probability of prior responses (which we don't want to do before computing p(z | y, w)

    // multiply each column by the probability of the prior response
    int r, c;
    for(r = 0; r < num_inner_nodes; r++){
        for(c = 0; c < n; c++){
            *(inner_anno_probs + r * n + c) = ( *(inner_anno_probs + r * n + c) ) * ( * (P + c) );
        }
    }

    // We now have a matrix of shape [num inner nodes, n] storing the probability of p(z_j | z, w_j) * p(H^{t-1} | z_j, w_j)
    // To compute the probability of the prior response for each inner node z:
    // The numerator is inner_anno_probs[z][z_prev]
    // The denominator is the sum of inner_anno_probs[z]

    // Go through each inner node
    for(r = 0; r < num_inner_nodes; r++){

        float num = *(inner_anno_probs + r * n + z_prev);
        if (normalize > 0){
            float denom = 0;
            for(c = 0; c < n; c++){
                denom += *(inner_anno_probs + r * n + c);
            }

            // Store the probability of the previous responses given this inner node response
            num /= denom;
        }
        prp_inner[r] = num;

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