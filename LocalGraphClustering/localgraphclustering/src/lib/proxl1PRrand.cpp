/**
 * Randomized proximal coordinate descent for l1 regularized pagerand vector
 * INPUT:
 *     alpha     - teleportation parameter between 0 and 1
 *     rho       - l1-reg. parameter
 *     v         - Seed node
 *     ai,aj,a   - Compressed sparse row representation of A
 *     d         - vector of node strengths
 *     epsilon   - accuracy for termination criterion
 *     n         - size of A
 *     ds        - the square root of d
 *     dsinv     - 1/ds
 *     offset    - offset for zero based arrays (matlab) or one based arrays (julia)
 *
 * OUTPUT:
 *     p              - PageRank vector as a row vector
 *     not_converged  - flag indicating that maxiter has been reached
 *     grad           - last gradient
 *
 */


// include write data to files
#include <iostream>
#include <fstream>
// end
#include <vector>
#include <cmath>
#include <ctime>
#include <unordered_set>
#include "include/routines.hpp"
#include "include/proxl1PRrand_c_interface.h"

using namespace std;

namespace proxl1PRrand 
{
    bool fileExist(const string& fname) {
        ifstream file(fname);
        return file.good();
    }


    template<typename vtype>
    double compute_l2_norm(double* vec, vtype n){
        double l2_norm = 0;
        for(vtype i = 0; i < n; i ++){
            l2_norm += (vec[i])*(vec[i]);
        }
        l2_norm = sqrt(l2_norm);

        return l2_norm;
    }

    template<typename vtype, typename itype>
    void writeQdiag(vtype num_nodes, const string& fname, itype* ai, vtype* aj, double* d, double alpha) {
        if (fileExist(fname)) {
            return;
        }
        vector<bool> Adiag(num_nodes, false);

        for (vtype node = 0; node < num_nodes; ++node) {
            itype idx = ai[node];
            while (idx < ai[node + 1] && aj[idx] < node) ++idx;
            Adiag[node] = idx < ai[node + 1] && aj[idx] == node;
        }

        double c = (1 - alpha) / 2;
        ofstream file(fname);
        if (file.is_open()) {
            for (vtype node = 0; node < num_nodes; ++node) {
                file << (1.0 - c - c * Adiag[node] / d[node]);
                if (node < num_nodes - 1) file << ',';
            }
            file.close();
        }
    }


    template<typename vtype, typename itype>
    void writeGraph(vtype num_nodes, const string& fname, itype* ai, vtype* aj) {
        if (fileExist(fname)) {
            return;
        }
        ofstream file(fname);
        if (file.is_open()) {
            itype idx = 0;
            for (vtype node = 0; node < num_nodes; ++node) {
                for (vtype neighbor = 0; neighbor < num_nodes; ++neighbor) {
                    if (idx < ai[node + 1] && aj[idx] == neighbor) {
                        ++idx;
                        file << 1;
                    } else {
                        file << 0;
                    }
                    if (neighbor < num_nodes - 1) file << ',';
                }
                file << '\n';
            }
            file.close();
        }
    }

    void writeTime(clock_t& timeStamp, const string& fname) {
        clock_t currTime = clock();
        ofstream file(fname, fstream::app);
        if (file.is_open()){
            file << double(currTime - timeStamp) / CLOCKS_PER_SEC << '\n';
            file.close();
        }
        timeStamp = currTime;
    }

    static long long g_seed = 1;

    inline
    long long getRand() {
        g_seed = (214013*g_seed+2531011); 
        return (g_seed>>16)&0x7FFF; 
    }

    template<typename vtype, typename itype>
    void updateGrad(vtype& node, double& stepSize, double& c, double& ra, double* q, double* grad, double* ds, double* dsinv, itype* ai, vtype* aj, double* a, bool* visited, vtype* candidates, vtype& candidates_size) {
        double dqs = -grad[node]-ds[node]*ra;
        double dq = dqs*stepSize;
        double cdq = c*dq;
        double cdqdsinv = cdq*dsinv[node];
        q[node] += dq;
        grad[node] += dqs;

        vtype neighbor;
        for (itype j = ai[node]; j < ai[node + 1]; ++j) {
            neighbor = aj[j];
            grad[neighbor] -= cdqdsinv*dsinv[neighbor]*a[j]; //(1 + alpha)
            
            if (!visited[neighbor] && q[neighbor] - stepSize*grad[neighbor] >= stepSize*ds[neighbor]*ra) {
                visited[neighbor] = true;
                candidates[candidates_size++] = neighbor;
            }
            
        }
    }


    template<typename vtype, typename itype>
    void updateGrad_unnormalized(vtype& node, double& stepSize_constant, double& c, double& ra, double* q, double* grad, double* d, itype* ai, vtype* aj, double* a, bool* visited, vtype* candidates, vtype& candidates_size) {
        double dqs = -grad[node]/d[node]-ra;
        double dq = dqs*stepSize_constant;
        double cdq = c*dq;
        q[node] += dq;
        grad[node] += dqs*d[node];

        vtype neighbor;
        for (itype j = ai[node]; j < ai[node + 1]; ++j) {
            neighbor = aj[j];
            grad[neighbor] -= cdq*a[j]; //(1 + alpha)
            if (!visited[neighbor] && q[neighbor] - stepSize_constant*(grad[neighbor]/d[neighbor]) >= stepSize_constant*ra) {
                visited[neighbor] = true;
                candidates[candidates_size++] = neighbor;
            }
        }
    }
    
}


namespace utilities {

    template<typename vtype, typename itype>
    class NeighborIterator {
        public:
            NeighborIterator(itype* ai, vtype* aj, double* a) {
                m_ai = ai;
                m_aj = aj;
                m_a = a;
            }

            // set the node of the iterator, iterator will point to neighbor of the node
            void set_node(vtype node) {
                m_node = node;
                m_ptr = m_ai[m_node];
                m_end = m_ai[m_node + 1];
            }

            // move forward iterator, should always be used after has_next
            void next() {
                ++m_ptr;
            }

            // get the index of the current neighbor
            vtype get_idx() const {
                return m_aj[m_ptr];
            }

            // get the adjacency matrix's value of the current neighbor
            double get_val() const {
                return m_a[m_ptr];
            }

            // check if the iterator has next value
            bool has_next() const {
                return m_ptr < m_end;
            }


        private:
            vtype m_node;
            itype m_ptr;
            itype m_end;
            // graph data
            vtype* m_aj;
            itype* m_ai;
            double* m_a;
    };

    template<typename vtype, typename itype>
    NeighborIterator<vtype, itype> get_neighbor_iterator(itype* ai, vtype* aj, double* a) {
        NeighborIterator<vtype, itype> iter(ai, aj, a);
        return iter;
    }

    template<typename vtype, typename itype>
    void add_neighbours(const vtype node, NeighborIterator<vtype, itype>& iter, unordered_set<vtype>& node_set) {
        iter.set_node(node);
        while (iter.has_next()) {
            node_set.insert(iter.get_idx());
            iter.next();
        }
    }
    
    /*--- test iterator efficiency ---*/
    static double time_iter;
    static double time_loop;
    /*--- remove in production vertion---*/
    template <typename vtype, typename itype>
    void test_neighbor_iter(const vtype& node_i, NeighborIterator<vtype, itype>& iter, itype* ai, vtype* aj, double* a) {
        clock_t begin;
        clock_t end;
        // measure iterator
        begin = clock();
        iter.set_node(node_i);
        while (iter.has_next()) {
            iter.get_idx();
            iter.get_val();
            iter.next();
        }
        end = clock();
        time_iter += double(end - begin);
        
        // compute baseline
        begin = clock();
        vtype idx;
        double val;
        for (itype j = ai[node_i]; j < ai[node_i + 1]; ++j) {
            idx = aj[j];
            val = a[j];
        }
        end = clock();
        time_loop += double(end - begin);
    }
    /*--- remove in production version ---*/
    
    // compute the gradient of f(q) at coordinate node_i
    template<typename vtype, typename itype>
    double compute_grad_i(
        const vtype& node_i, const double& theta, double* u, double* z,
        const double& alpha, double* dsinv, 
        vtype* candidates, vtype& candidates_size, bool* visited, bool* is_seed, double seed_val, 
        NeighborIterator<vtype, itype>& neighbor_iter) 
    {
        double theta_sqr = theta * theta;
        double sum = 0;
        vtype neighbor;
        neighbor_iter.set_node(node_i);
        while (neighbor_iter.has_next()) {
            neighbor = neighbor_iter.get_idx();
            sum += (theta_sqr * u[neighbor] + z[neighbor]) * neighbor_iter.get_val() * dsinv[neighbor];
            neighbor_iter.next();
        }

        sum *= (1 - alpha) / 2 * dsinv[node_i];
        double grad_i = (1 + alpha) / 2 * (theta_sqr * u[node_i] + z[node_i]) - sum;
        if (is_seed[node_i]) {
            grad_i -= alpha * dsinv[node_i] * seed_val;
        }
        return grad_i;
    }

    template<typename vtype, typename itype>
    void update_neighbour_grad_term2(
        const vtype& node_i, double delta_q_i, double* u, double* z, 
        const double& alpha, double* dsinv, double* grad_term2, double* a,
        NeighborIterator<vtype, itype>& neighbor_iter) 
    {
        vtype neighbor;
        neighbor_iter.set_node(node_i);
        
        // grad[node_i] = grad[node_i] + (1.0 + alpha) / 2.0 * delta_q_i;
        double delta_neighbor = (1.0 - alpha) * 0.5 * dsinv[node_i] * delta_q_i * a[node_i];
        while (neighbor_iter.has_next()) {
            neighbor = neighbor_iter.get_idx();
            grad_term2[neighbor] -= delta_neighbor * dsinv[neighbor];
            neighbor_iter.next();
        }
    }
    
    template<typename vtype>
    double compute_Qij(vtype i, vtype j, double alpha, double a_ij, double* dsinv) {
        double Qij = -a_ij * dsinv[i] * dsinv[j];
        if (i == j) {
            Qij += (1 + alpha) / 2;
        }
        return Qij;
    }

    template<typename vtype, typename itype>
    double compute_func_val(vtype num_nodes, double alpha, double rho, double* q, double* dsinv, itype* ai, vtype* aj, double* a, bool* is_seed) {
        auto neighbor_iter = get_neighbor_iterator(ai, aj, a);
        double func_val = 0;
        for (vtype i = 0; i < num_nodes; ++i) {
            neighbor_iter.set_node(i);
            while  (neighbor_iter.has_next()) {
                vtype j = neighbor_iter.get_idx();
                double a_ij = neighbor_iter.get_val();
                neighbor_iter.next();
                double Qij = compute_Qij(i, j, alpha, a_ij, dsinv);
                func_val += 0.5 * q[i] * Qij * q[j] / dsinv[i] / dsinv[j];
            }

            if (is_seed[i]) {
                func_val -= alpha * q[i] * dsinv[i] / dsinv[i];
            }
            func_val += abs(q[i] / dsinv[i]) * rho * alpha / dsinv[i];
        }
        return func_val;
    }

    template<typename vtype, typename itype>
    void update_candidates_nzeros(
        vtype num_nodes, double prev_theta,
        double* u, double* z,
        itype* ai, vtype* aj, double* a,
        vtype* candidates, vtype& candidates_size, 
        bool* visited, bool* is_seed
    )
    {
        candidates_size = 0;
        for (vtype i = 0; i < num_nodes; ++i) visited[i] = 0;

        auto neighbor_iter = get_neighbor_iterator(ai, aj, a);
        for (vtype i = 0; i < num_nodes; ++i) {
            if (is_seed[i] || abs(prev_theta * prev_theta * u[i] + z[i]) > 1e-7) {
                vector<vtype> indices = {i};
                neighbor_iter.set_node(i);
                while (neighbor_iter.has_next()) {
                    indices.push_back(neighbor_iter.get_idx());
                    neighbor_iter.next();
                }
                for (vtype idx : indices) {
                    if (!visited[idx]) {
                        visited[idx] = true;
                        candidates[candidates_size++] = idx;
                    }
                }
            }
        }
    }

    template<typename vtype>
    void update_candidates_prox(
        vtype num_nodes, double prev_theta, double theta, double rho, double alpha, double lipschtz,
        double* u, double* z, double* grad, double* ds,
        vtype* candidates, vtype& candidates_size,
        bool* visited, bool* is_seed
    ) 
    {
        candidates_size = 0;
        // for (vtype i = 0; i < num_nodes; ++i) visited[i] = true;
        
        for (vtype i = 0; i < num_nodes; ++i) {
            if (abs(prev_theta * prev_theta * z[i] + u[i]) < 1e-7 && abs(grad[i]) < 1e-7) {
                continue;
            }
            if (z[i] > (grad[i] + rho * alpha * ds[i]) / (num_nodes * theta * lipschtz) || z[i] < (grad[i] - rho * alpha * ds[i]) / (num_nodes * theta * lipschtz)) {
                candidates[candidates_size++] = i;
            }
        }
    }

    template<typename vtype>
    void update_candidates_loop(
        vtype num_nodes, vtype prev_node,
        vtype* candidates, vtype& candidates_size
    ) 
    {
        candidates_size = 1;
        vtype next_idx = (prev_node + 1) % num_nodes;
        candidates[0] = next_idx;
    }

    template<typename vtype, typename itype>
    void update_candidates_max_grad(
        vtype num_nodes, double* grad,
        itype* ai, vtype* aj, double* a,
        vtype* candidates, vtype& candidates_size
    )
    {
        vtype node = 0;
        auto neighbor_iter = get_neighbor_iterator(ai, aj, a);
        for (vtype i = 1; i < num_nodes; ++i) {
            if (abs(grad[node]) < abs(grad[i])) {
                node = i;
            }
        }

        candidates[0] = node;
        candidates_size = 1;
        neighbor_iter.set_node(node);
        while (neighbor_iter.has_next()) {
            vtype neighbor = neighbor_iter.get_idx();
            candidates[candidates_size++] = neighbor;
            neighbor_iter.next();
        }
    }

    template<typename size_type, typename array_type>
    void write_array(array_type& array, size_type size, const string& fname) {
        ofstream file(fname, fstream::app);
        if (file.is_open()) {
            for (size_type i = 0; i < size; ++i) {
                file << array[i];
                if (i < size - 1) file << ',';
            }
            file << '\n';
            file.close();
        }
    }
}


template<typename vtype, typename itype>
vtype graph<vtype,itype>::proxl1PRrand(vtype num_nodes, vtype* seed, vtype num_seeds, double epsilon, double alpha, double rho, double* q, double* y, double* d, double* ds, double* dsinv, double* grad, vtype maxiter)
{
	vtype not_converged = 0;
    vtype* candidates = new vtype[num_nodes];
    bool* visited = new bool[num_nodes];
    bool* is_seed = new bool[num_nodes];  // for testing
    vector<double> fun_values;
    for (vtype i = 0; i < num_nodes; ++i) visited[i] = false;

    // initialize seed nodes as candidates
    double maxNorm = 0;
    vtype candidates_size = num_seeds;
    for (vtype i = 0; i < num_seeds; ++i) {
        // set gradient and update max norm
        grad[seed[i]] = -alpha*dsinv[seed[i]]/num_seeds;
        maxNorm = max(maxNorm, abs(grad[seed[i]]*dsinv[seed[i]]));
        // set as candidate nodes
        candidates[i] = seed[i];
        is_seed[seed[i]] = visited[seed[i]] = true;
    }

    double c = (1-alpha)/2;
    double ra = rho*alpha;
    double stepSize = 2.0/(1+alpha);
    
    for(vtype i = 0; i < num_seeds; i ++){
        grad[seed[i]] = -alpha*dsinv[seed[i]]/num_seeds;
    }

    //Find nonzero indices in y and dsinv
    unordered_map<vtype,vtype> indices;
    unordered_set<vtype> nz_ids;
    for (vtype i = 0; i < num_nodes; i ++) {
        if (y[i] != 0 && dsinv[i] != 0) {
            indices[i] = 0;
            q[i] = y[i];
        }
        if (y[i] != 0 || grad[i] != 0) {
            nz_ids.insert(i);
        }
    }
    
    
    for(auto it = nz_ids.begin(); it != nz_ids.end(); ++it){
        vtype i = *it;
        grad[i] += y[i]/stepSize;
    }
    vtype temp;

    for(auto it = indices.begin() ; it != indices.end(); ++it){
        vtype i = it->first;
        for(itype j = ai[i]; j < ai[i+1]; j ++){
            temp = aj[j];
            grad[temp] -= y[i] * a[j] * dsinv[i] * dsinv[temp] * c;
            
            if (!visited[temp] && y[temp] - stepSize*grad[temp] >= stepSize*ds[temp]*ra) {
                visited[temp] = true;
                candidates[candidates_size++] = temp;
            }
        }
    }
    // force the method sample from a fixed set of nodes
    /*
    candidates_size = 0;
    vector<int> temp_candidates = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12, 13, 515, 1117, 2105, 2163, 2181, 2662, 3678, 4386, 4895, 4910};
    
    for (auto x : temp_candidates) {
        candidates[candidates_size++] = x;
    }
    */
    /*
    unordered_set<int> true_set;
    vector<int> true_candidates = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 416, 1123, 2209, 2403, 2744, 3335, 3460, 3357, 3618, 4097, 4584, 5013};
    for (int x : true_candidates) true_set.insert(x);
    */
    // auto neighbor_iter = utilities::get_neighbor_iterator(ai, aj, a);
    double threshold = (1+epsilon)*rho*alpha;
    vtype numiter = 0;
    // some constant
    maxiter *= 100;
    double total_time = 0;
    while (maxNorm > threshold) {
        numiter++;
        auto _time = clock();
        vtype r = proxl1PRrand::getRand() % candidates_size;
        /*
        if (!true_set.count(candidates[r])) {
            candidates[r] = (candidates[r] + 1) % num_nodes;
            double fun_value = utilities::compute_func_val(num_nodes, alpha, rho, q, dsinv, ai, aj, a, is_seed);
            fun_values.push_back(fun_value);
            continue;
        }
        */
        proxl1PRrand::updateGrad(candidates[r], stepSize, c, ra, q, grad, ds, dsinv, ai, aj, a, visited, candidates, candidates_size);
        // process coordinate sequentially
        // utilities::update_candidates_loop(num_nodes, candidates[r], candidates, candidates_size);
        // write function value for plot
        // double fun_value = utilities::compute_func_val(num_nodes, alpha, rho, q, dsinv, ai, aj, a, is_seed);
        // fun_values.push_back(fun_value);
        if (1) {
            maxNorm = 0;
            for (vtype i = 0; i < num_nodes; ++i) {
                // double grad_i = utilities::compute_grad_i(i, 0, q, q, alpha, dsinv, candidates, candidates_size, visited, is_seed, 1.0 / num_seeds, neighbor_iter);
                maxNorm = max(maxNorm, abs(grad[i]*dsinv[i]));
            }
            if (numiter > maxiter) {
                not_converged = 1;
                break;
            }
        }
        total_time += clock() - _time;
    }
    cout << "method: randomized coordinate descent" << endl;
    cout << "number of nodes: " << num_nodes << endl;
    cout << "number of candidates: " << candidates_size << endl;
    cout << "number of iterations: " << numiter << endl;
    cout << "average time per iteration: " << total_time / numiter << endl;
    //proxl1PRrand::writeTime(timeStamp, "/home/c55hu/Documents/research/experiment/output/time-rand.txt");
    // write fun_values
    cout << "function values length: " << fun_values.size() << endl;
    utilities::write_array(fun_values, fun_values.size(), "/home/c55hu/Documents/research/experiment/output/fun_values_rand.txt");
    // update y and q
    
    for (vtype i = 0; i < num_nodes; ++i) y[i] = q[i];
    
    for (vtype i = 0; i < num_nodes; ++i) q[i] *= ds[i];
    
    delete [] candidates;
    delete [] visited;
    return not_converged;
}

template<typename vtype, typename itype>
vtype graph<vtype,itype>::proxl1PRrand_unnormalized(vtype num_nodes, vtype* seed, vtype num_seeds, double epsilon, double alpha, double rho, double* q, double* y, double* d, double* ds, double* dsinv, double* grad, vtype maxiter)
{
    clock_t timeStamp = clock();
    
	vtype not_converged = 0;
    vtype* candidates = new vtype[num_nodes];
    bool* visited = new bool[num_nodes];
    for (vtype i = 0; i < num_nodes; ++i) visited[i] = false;
    
    
    // initialize seed nodes as candidates
    double maxNorm = 0;
    vtype candidates_size = num_seeds;
    for (vtype i = 0; i < num_seeds; ++i) {
        // set gradient and update max norm
        grad[seed[i]] = -alpha/num_seeds;
        maxNorm = max(maxNorm, abs(grad[seed[i]]*d[seed[i]]));
        // set as candidate nodes
        candidates[i] = seed[i];
        visited[seed[i]] = true;
    }
    
    double c = (1-alpha)/2;
    double ra = rho*alpha;
    double stepSize_const = 2.0/(1+alpha);

    for(vtype i = 0; i < num_seeds; i ++){
        grad[seed[i]] = -alpha/num_seeds;
    }

    //Find nonzero indices in y and dsinv
    unordered_map<vtype,vtype> indices;
    unordered_set<vtype> nz_ids;
    for (vtype i = 0; i < num_nodes; i ++) {
        if (y[i] != 0 && dsinv[i] != 0) {
            indices[i] = 0;
            q[i] = y[i];
        }
        if (y[i] != 0 || grad[i] != 0) {
            nz_ids.insert(i);
        }
    }
    
    
    for(auto it = nz_ids.begin() ; it != nz_ids.end(); ++it) {
        vtype i = *it;
        grad[i] += y[i]*d[i]/stepSize_const;
    }
    vtype temp;

    for(auto it = indices.begin() ; it != indices.end(); ++it) {
        vtype i = it->first;
        for(itype j = ai[i]; j < ai[i+1]; j ++){
            temp = aj[j];
            grad[temp] -= y[i] * a[j] * c;
            
            if (!visited[temp] && y[temp] - stepSize_const*(grad[temp]/d[temp]) >= stepSize_const*ra) {
                visited[temp] = true;
                candidates[candidates_size++] = temp;
            }
        }
    }

    // exp start write graph
    // proxl1PRrand::writeGraph(num_nodes, "/home/c55hu/Documents/research/experiment/output/graph.txt", ai, aj);
    // proxl1PRrand::writeQdiag(num_nodes, "/home/c55hu/Documents/research/experiment/output/Qdiag.txt", ai, aj, d, alpha);
    // exp end
    double threshold = (1+epsilon)*rho*alpha;
    vtype numiter = 0;
    // some constant
    maxiter *= 100;
    //for (vtype i = 0; i < num_nodes; ++i) ds[i] *= ra;
    while (maxNorm > threshold) {
        
        vtype r =  proxl1PRrand::getRand() % candidates_size;
        proxl1PRrand::updateGrad_unnormalized(candidates[r], stepSize_const, c, ra, q, grad, d, ai, aj, a, visited, candidates, candidates_size);
        
        if (numiter % num_nodes == 0) {
            maxNorm = 0;
            for (vtype i = 0; i < num_nodes; ++i) {
                maxNorm = max(maxNorm, abs(grad[i]/d[i]));
            }
        }
        
        if (numiter++ > maxiter) {
            not_converged = 1;
            break;
        }
   
        
    }
    //proxl1PRrand::writeTime(timeStamp, "/home/c55hu/Documents/research/experiment/output/time-rand.txt");
    //proxl1PRrand::writeLog(num_nodes, "/home/c55hu/Documents/research/experiment/output/q-rand.txt", q);

    for (vtype i = 0; i < num_nodes; ++i) y[i] = q[i];
    
    delete [] candidates;
    delete [] visited;
    return not_converged;
}

// method for accelerated proximal coordinate descent
template<typename vtype, typename itype>
vtype graph<vtype,itype>::proxl1PRaccel_rand(
    vtype num_nodes, vtype* seed, vtype num_seeds, 
    double epsilon, double alpha, double rho, double* q, double* y, 
    double* d, double* ds, double* dsinv, double* grad, vtype maxiter) 
{

    vtype not_converged = 0;
    vtype* candidates = new vtype[num_nodes];
    bool* visited = new bool[num_nodes];
    bool* is_seed = new bool[num_nodes];
    double* grad_term2 = new double[num_nodes];
    vector<double> fun_values;
    vector<int> candidates_records;
    for (vtype i = 0; i < num_nodes; ++i) grad[i] = grad_term2[i] = is_seed[i] = visited[i] = false;

    // initialize gradient, seed nodes and candidate pool
    double maxNorm = 0;
    vtype candidates_size = 0;
    for (vtype i = 0; i < num_seeds; ++i) {
        grad[seed[i]] = grad_term2[seed[i]] = -alpha * dsinv[seed[i]] / num_seeds;
        candidates[candidates_size++] = seed[i];
        is_seed[seed[i]] = visited[seed[i]] = true;
        maxNorm = max(maxNorm, abs(grad[seed[i]] * dsinv[seed[i]]));
    }
    // test
    /*
    candidates_size = 0;
    vector<int> temp = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12, 13, 515, 1117, 2105, 2163, 2181, 2662, 3678, 4386, 4895, 4910};
    for (auto x : temp) {
        candidates[candidates_size++] = x;
    }
    */
    // additional sequences for accelerated method
    double* u = new double[num_nodes];
    double* z = new double[num_nodes];
    for (vtype i = 0; i < num_nodes; ++i) {
        u[i] = 0;
        z[i] = q[i];
    }

    // some constants
    double tau = 1.0;
    double lipschtz = (1.0 + alpha) / 2.0;
    double num_nodes_lipschtz_inv = 1.0 / (num_nodes * lipschtz);
    double rho_alpha = rho * alpha;
    double seed_val = 1.0 / num_seeds;
    double threshold = (1+epsilon)*rho_alpha;
    
    // variables
    vtype node_i;
    vtype numiter = 1;
    maxiter *= 100;  // hard coded for test, coordinate descent uses more iterations
    
    double prev_theta = 1.0 / num_nodes;
    double theta = tau / num_nodes;  // choose tau = 1
    double rho_alpha_ds_i, frac, upper, lower, t, theta_sqr;
    
    auto neighbor_iter = utilities::get_neighbor_iterator(ai, aj, a);
    /*--- test iterator efficiency ---*/
    utilities::time_iter = 0;
    utilities::time_loop = 0;
    /*--- remove in the production version ---*/
    double total_time = 0;

    double grad_i;
    double q_i;
    double delta_q_i;
    double avg_candidates_size = 0;

    vtype max_cand_size = 0;

    while (maxNorm > threshold) {
        auto _time = clock();
        avg_candidates_size += candidates_size;
        node_i = candidates[proxl1PRrand::getRand() % candidates_size];
        grad_i = utilities::compute_grad_i(node_i, theta, u, z, alpha, dsinv, candidates, candidates_size, visited, is_seed, seed_val, neighbor_iter);
        // proxl1PRrand::updateGrad_accel(node_i, step_size, c, rho_alpha, q, theta, prev_theta, u, z, grad, ds, dsinv, ai, aj, a, visited, candidates, candidates_size);
        /*--- test iterator efficiency ---*/
        // utilities::test_neighbor_iter(node_i, neighbor_iter, ai, aj, a);
        /*--- remove in the production version ---*/

        // compute t
        rho_alpha_ds_i = rho_alpha * ds[node_i];
        frac = num_nodes_lipschtz_inv / theta * tau;
        upper = (grad_i + rho_alpha_ds_i) * frac;
        lower = (grad_i - rho_alpha_ds_i) * frac;
        theta_sqr = theta * theta;

        // compute previous delta q i
        // q_i = prev_theta * prev_theta * u[node_i] + z[node_i];

        // l1 cases, proximal operator
        if (z[node_i] <= upper && z[node_i] >= lower) {
            u[node_i] += (1 - num_nodes * theta / tau) / theta_sqr * z[node_i];
            z[node_i] = 0;
        } 
        else {
            if (z[node_i] > upper) {
                t = -upper;
                //cout << "case 1\n";
            } else {
                t = -lower;
            }
            z[node_i] += t;
            u[node_i] -= (1 - num_nodes * theta) / theta_sqr * t;
        }        
        // compute next q i
        // delta_q_i = theta * theta * u[node_i] + z[node_i] - q_i;
        // TODO: debug
        // utilities::update_neighbour_grad_term2(node_i, delta_q_i, u, z, alpha, dsinv, grad, a, neighbor_iter);
        prev_theta = theta;
        theta = (sqrt(theta_sqr * theta_sqr + 4 * theta_sqr) - theta_sqr) / 2;
        // update iteration status
        ++numiter; 
        if (1) {
            if (numiter > maxiter) {
                not_converged = 1;
                break;
            } else {
                
                double theta_inv = 1.0 / theta;
                double prev_theta_sqr = prev_theta * prev_theta;

                maxNorm = 0;
                candidates_size = 0;
                for (vtype i = 0; i < num_nodes; ++i) {
                    grad[i] = utilities::compute_grad_i(i, prev_theta, u, z, alpha, dsinv, candidates, candidates_size, visited, is_seed, seed_val, neighbor_iter);
                    grad_i = utilities::compute_grad_i(i, theta, u, z, alpha, dsinv, candidates, candidates_size, visited, is_seed, seed_val, neighbor_iter);
                    if (z[i] > (grad_i + rho_alpha * ds[i]) * num_nodes_lipschtz_inv * theta_inv || z[i] < (grad_i - rho_alpha * ds[i]) * num_nodes_lipschtz_inv * theta_inv) {
                        candidates[candidates_size++] = i;
                    } else if (abs(prev_theta_sqr * u[i] + z[i]) > 0 || abs(grad[i]) > 0.00001) {
                        candidates[candidates_size++] = i;
                    }
                    maxNorm = max(maxNorm, abs(grad[i]*dsinv[i]));
                }

                // utilities::update_candidates_nzeros(num_nodes, prev_theta, u, z, ai, aj, a, candidates, candidates_size, visited, is_seed);
                // utilities::update_candidates_prox(num_nodes, prev_theta, theta, rho, alpha, lipschtz, u, z, grad, ds, candidates, candidates_size, visited, is_seed);
                max_cand_size = max(max_cand_size, candidates_size);
                // utilities::update_candidates_loop(num_nodes, node_i, candidates, candidates_size);
                // utilities::update_candidates_max_grad(num_nodes, grad, ai, aj, a, candidates, candidates_size);
                // double fun_value = utilities::compute_func_val(num_nodes, alpha, rho, q, dsinv, ai, aj, a, is_seed);
                // fun_values.push_back(fun_value);
                // candidates_records.push_back(candidates_size);
            }
        }
        
        total_time += double(clock() - _time);
    }
    // compute q
    double prev_theta_sqr = prev_theta * prev_theta;
    for (vtype i = 0; i < num_nodes; ++i) {
        q[i] = (prev_theta_sqr * u[i] + z[i]) * ds[i];
        // grad[i] = utilities::compute_grad_i(i, prev_theta, u, z, grad, alpha, dsinv, candidates, candidates_size, visited, is_seed, seed_val, neighbor_iter);
    }

    cout << "number of candidates: " << candidates_size << endl;
    cout << "number of iterations: " << numiter << endl;
    cout << "max candidates size: " << max_cand_size << endl;
    cout << "average time per iteration: " << total_time / numiter << "ms" << endl;
    cout << "average candidates size per iteration: " << avg_candidates_size / numiter << endl;
    unordered_set<vtype> node_set;
    for (int i = 0; i < candidates_size; ++i) {
        vtype node = candidates[i];
        utilities::add_neighbours(node, neighbor_iter, node_set);
    }
    cout << "number of non zeros and neighbours: " << node_set.size() << endl;
    cout << "average time of using iterator: " << utilities::time_iter / numiter << endl;
    cout << "average time of using loop: " << utilities::time_loop / numiter << endl;
    utilities::write_array(fun_values, fun_values.size(), "/home/c55hu/Documents/research/experiment/output/fun_values_rand_accel.txt");
    utilities::write_array(candidates_records, candidates_records.size(), "/home/c55hu/Documents/research/experiment/output/candidates_records.txt");

    delete [] candidates;
    delete [] visited;
    delete [] is_seed;
    delete [] u;
    delete [] z;

    /*--- test iterator efficiency ---*/
    // cout << "time used by iterator: " << utilities::time_iter << "ms" << endl;
    // cout << "time used by norm loop: " << utilities::time_loop << "ms" << endl; 
    /*--- remove in the production version ---*/

    return not_converged;
}

// randomized method
uint32_t proxl1PRrand32(uint32_t n, uint32_t* ai, uint32_t* aj, double* a, double alpha,
                         double rho, uint32_t* v, uint32_t v_nums, double* d, double* ds,
                         double* dsinv, double epsilon, double* grad, double* p, double* y,
                         uint32_t maxiter, uint32_t offset, double max_time,bool normalized_objective)
{
    graph<uint32_t,uint32_t> g(ai[n],n,ai,aj,a,offset,NULL);
    if (normalized_objective){
        return g.proxl1PRrand(n, v, v_nums, epsilon, alpha, rho, p, y, d, ds, dsinv, grad, maxiter);
    }
    else{
        return g.proxl1PRrand_unnormalized(n, v, v_nums, epsilon, alpha, rho, p, y, d, ds, dsinv, grad, maxiter);
    }
}

int64_t proxl1PRrand64(int64_t n, int64_t* ai, int64_t* aj, double* a, double alpha,
                        double rho, int64_t* v, int64_t v_nums, double* d, double* ds,
                        double* dsinv,double epsilon, double* grad, double* p, double* y,
                        int64_t maxiter, int64_t offset, double max_time,bool normalized_objective)
{
    graph<int64_t,int64_t> g(ai[n],n,ai,aj,a,offset,NULL);
    if (normalized_objective){
        return g.proxl1PRrand(n, v, v_nums, epsilon, alpha, rho, p, y, d, ds, dsinv, grad, maxiter);
    }
    else{
        return g.proxl1PRrand_unnormalized(n, v, v_nums, epsilon, alpha, rho, p, y, d, ds, dsinv, grad, maxiter);
    }
}

uint32_t proxl1PRrand32_64(uint32_t n, int64_t* ai, uint32_t* aj, double* a, double alpha,
                            double rho, uint32_t* v, uint32_t v_nums, double* d, double* ds,
                            double* dsinv, double epsilon, double* grad, double* p, double* y,
                            uint32_t maxiter, uint32_t offset, double max_time,bool normalized_objective)
{
    graph<uint32_t,int64_t> g(ai[n],n,ai,aj,a,offset,NULL);
    if (normalized_objective){
        return g.proxl1PRrand(n, v, v_nums, epsilon, alpha, rho, p, y, d, ds, dsinv, grad, maxiter);
    }
    else{
        return g.proxl1PRrand_unnormalized(n, v, v_nums, epsilon, alpha, rho, p, y, d, ds, dsinv, grad, maxiter);
    }
}

// accelerated method
uint32_t proxl1PRaccel_rand32(uint32_t n, uint32_t* ai, uint32_t* aj, double* a, double alpha,
                         double rho, uint32_t* v, uint32_t v_nums, double* d, double* ds,
                         double* dsinv, double epsilon, double* grad, double* p, double* y,
                         uint32_t maxiter, uint32_t offset, double max_time,bool normalized_objective)
{
    graph<uint32_t,uint32_t> g(ai[n],n,ai,aj,a,offset,NULL);
    if (normalized_objective){
        return g.proxl1PRaccel_rand(n, v, v_nums, epsilon, alpha, rho, p, y, d, ds, dsinv, grad, maxiter);
    }
    else{
        return g.proxl1PRrand_unnormalized(n, v, v_nums, epsilon, alpha, rho, p, y, d, ds, dsinv, grad, maxiter);
    }
}

int64_t proxl1PRaccel_rand64(int64_t n, int64_t* ai, int64_t* aj, double* a, double alpha,
                        double rho, int64_t* v, int64_t v_nums, double* d, double* ds,
                        double* dsinv,double epsilon, double* grad, double* p, double* y,
                        int64_t maxiter, int64_t offset, double max_time,bool normalized_objective)
{
    graph<int64_t,int64_t> g(ai[n],n,ai,aj,a,offset,NULL);
    if (normalized_objective){
        return g.proxl1PRaccel_rand(n, v, v_nums, epsilon, alpha, rho, p, y, d, ds, dsinv, grad, maxiter);
    }
    else{
        return g.proxl1PRrand_unnormalized(n, v, v_nums, epsilon, alpha, rho, p, y, d, ds, dsinv, grad, maxiter);
    }
}

uint32_t proxl1PRaccel_rand32_64(uint32_t n, int64_t* ai, uint32_t* aj, double* a, double alpha,
                            double rho, uint32_t* v, uint32_t v_nums, double* d, double* ds,
                            double* dsinv, double epsilon, double* grad, double* p, double* y,
                            uint32_t maxiter, uint32_t offset, double max_time,bool normalized_objective)
{
    graph<uint32_t,int64_t> g(ai[n],n,ai,aj,a,offset,NULL);
    if (normalized_objective){
        return g.proxl1PRaccel_rand(n, v, v_nums, epsilon, alpha, rho, p, y, d, ds, dsinv, grad, maxiter);
    }
    else{
        return g.proxl1PRrand_unnormalized(n, v, v_nums, epsilon, alpha, rho, p, y, d, ds, dsinv, grad, maxiter);
    }
}