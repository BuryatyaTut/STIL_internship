/* dynamic_prog.cpp  --- dynamic programming function fill_dp_matrix and
 *   various versions of backtrack.
 *
 * Copyright (C) 2010-2016 Mingzhou Song and Haizhou Wang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
//
// Joe Song
// Created: August 20, 2016. Extracted from Ckmeans.1d.dp.cpp

//#define DEBUG


#include "Ckmeans.1d.dp.h"
#include "EWL2_within_cluster.h"

//#define DEBUG


void backtrack(const std::vector<double> & x,
               const std::vector< std::vector< size_t > > & J,
               int* cluster, double* centers)
{
  const size_t K = J.size();
  const size_t N = J[0].size();
  size_t cluster_right = N-1;
  size_t cluster_left;

    //std::cout<<"K: "<<K<<std::endl;
  // Backtrack the clusters from the dynamic programming matrix
  for(int q = ((int)K)-1; q >= 0; --q) {
    
    cluster_left = J[q][cluster_right];

    for(size_t i = cluster_left; i <= cluster_right; ++i)
      cluster[i] = q;

    double sum = 0.0;

    for(size_t i = cluster_left; i <= cluster_right; ++i)
      sum += x[i];

    centers[q] = sum / (cluster_right-cluster_left+1);

    //for(size_t i = cluster_left; i <= cluster_right; ++i)
    //  withinss[q] += (x[i] - centers[q]) * (x[i] - centers[q]);

    /*count[q] = (int) (cluster_right - cluster_left + 1);*/

    if(q > 0) {
      cluster_right = cluster_left - 1;
    }
  }
}



