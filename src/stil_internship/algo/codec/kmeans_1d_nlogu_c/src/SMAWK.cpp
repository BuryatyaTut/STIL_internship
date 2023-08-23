/* EWL2_fill_SMAWK.cpp --- a divide-and-conquer algorithm to compute a
 *   row in the dynamic programming matrix in O(n) time for equally
 *   weighted L2 univariate k-means
 *
 * Copyright (C) 2016-2020 Mingzhou Song
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
// Created: January 3, 2020
#include <algorithm>
#include <vector>

#include "metrics.h"
void reduce_in_place(long long imin, long long imax, long long istep, 
					 const std::vector<long long> & js,
					 std::vector<long long> & js_red,
					 double* sum_x,
					 double* sum_x_sq,
					 double tau,
					 double* F)
{

  long long N = (imax - imin) / istep + 1;

  js_red = js;

  if(N >= js.size()) {
	return;
  }

  // Two positions to move candidate j's back and forth
  long long left = -1; // points to last favorable position / column
  long long right = 0; // points to current position / column

  size_t m = js_red.size();

  while(m > N) { // js_reduced has more than N positions / columns

	long long p = left + 1;

	long long i = imin + p * istep;
	long long j = (js_red[right]);
	double Sl = (F[j] +
	  w(j, i, sum_x, sum_x_sq, tau, N));
	// ssq(j, i, sum_x, sum_x_sq, sum_w));

	long long jplus1 = (js_red[right+1]);
	double Slplus1 = (F[jplus1] +
	  w(jplus1, i, sum_x, sum_x_sq, tau, N));
	// ssq(jplus1, i, sum_x, sum_x_sq, sum_w));

	if(Sl < Slplus1 && p < N-1) {
	  js_red[ ++ left ] = j; // i += istep;
	  right ++; // move on to next position / column p+1
	} else if(Sl < Slplus1 && p == N-1) {
	  js_red[ ++ right ] = j; // delete position / column p+1
	  m --;
	} else { // (Sl >= Slplus1)
	  if(p > 0) { // i > imin
		// delete position / column p and
		//   move back to previous position / column p-1:
		js_red[right] = js_red[left --];
		// p --; // i -= istep;
	  } else {
		right ++; // delete position / column 0
	  }
	  m --;
	}
  }

  for(int r=left+1; r < m; ++r) {
	js_red[r] = js_red[right++];
  }

  js_red.resize(m);
}

inline void fill_even_positions
(long long imin, long long imax, long long istep,
 const std::vector<long long> & js,
 double* sum_x,
 double* sum_x_sq,
 double tau,
 double* F,
 long long* J,
 double* F_top,
 size_t N)
{
  // Derive j for even rows (0-based)
  long long n = js.size();
  long long istepx2 = istep << 1;
  long long jl = js[0];
  for(long long i=imin, r(0); i<=imax; i+=istepx2) {

	while(js[r] < jl) {
	  // Increase r until it points to a value of at least jmin
	  r ++;
	}

	// Initialize S[q][i] and J[q][i]
	F[i] = F_top[js[r]] +
	  w(js[r],i, sum_x, sum_x_sq, tau, N);
	// ssq(js[r], i, sum_x, sum_x_sq, sum_w);
	J[i] = js[r]; // rmin

	// Look for minimum S upto jmax within js
	long long jh =  i + istep <= imax ? J[i + istep] : js[n-1] ;

	long long jmax = std::min(jh, i);

	double sjimin(
		w( jmax,i, sum_x, sum_x_sq, tau, N)
	  // ssq(jmax, i, sum_x, sum_x_sq, sum_w)
	);

	for(++ r; r < n && js[r]<=jmax; r++) {
		//std::cout<<"r: "<<r<<std::endl;
	  const long long & jabs = js[r];

	  if(jabs > i) break;

	  if(jabs < J[i]) continue;

	  double s =
		w(jabs, i, sum_x, sum_x_sq, tau, N);
	  // (ssq(jabs, i, sum_x, sum_x_sq, sum_w));
	  double Sj = (F_top[jabs] + s);

	  if(Sj <= F[i]) {
		F[i] = Sj;
		J[i] = js[r];
	  } else if(F_top[jabs] + sjimin > F[i]) {
		break;
	  } /*else if(S[q-1][js[rmin]-1] + s > S[q][i]) {
 break;
	  } */
	}
	r --;
	jl = jh;
  }
}

inline void find_min_from_candidates
(long long imin, long long imax, long long istep,
 const std::vector<long long> & js,
 double* sum_x,
 double* sum_x_sq,
 double tau,
 double* F,
 long long* J,
 double* F_top,
 size_t N)
{
  long long rmin_prev = 0;
  for(long long i=imin; i<=imax; i+=istep) { //find_min_from candidates is only ever called with imin<=imax, why is this even needed
											 //unless istep can be negative or zero, which would be extremely weird
											 //wait no, this might actually be useful when we receive a 1xN matrix and want to populate all the J's
	
	long long rmin = (rmin_prev);

	F[i] = F_top[js[rmin]] + w( js[rmin],i, sum_x, sum_x_sq, tau, N);
	J[i] = js[rmin];
	
	for(long long r = (rmin+1); r<js.size(); ++r) {

	  const long long & j_abs = js[r];

	  if(j_abs < J[i]) continue;
	  if(j_abs > i) break;

	  double Sj = (F_top[j_abs] +
		w(j_abs,i, sum_x, sum_x_sq, tau, N));
	  // ssq(j_abs, i, sum_x, sum_x_sq, sum_w));
	  if(Sj <= F[i]) {
		F[i] = Sj;
		J[i] = js[r];
		
		rmin_prev = r;
	  }
	}
  }
}

void SMAWK
(long long imin, long long imax, long long istep,
 const std::vector<long long> & js,
 double* sum_x,
 double* sum_x_sq,
 double* F, 
 double tau,
 long long* J,
 double* F_top,
size_t N)
{
  if(imax - imin <= 0 * istep) { // base case only one element left
	
	find_min_from_candidates(
	  imin, imax, istep, js, sum_x, sum_x_sq, tau, F, J, F_top, N
	);
   

  } else {

	// REDUCE
	  
	std::vector<long long> js_odd;

	reduce_in_place(imin, imax, istep, js, js_odd,
					sum_x, sum_x_sq, tau, F);
   
	long long istepx2 = istep << 1;
	long long imin_odd = imin + istep;
	long long imax_odd = imin_odd + (imax - imin_odd) / istepx2 * istepx2;

	// Recursion on odd rows (0-based):
	SMAWK(imin_odd, imax_odd, istepx2,
		 js_odd, sum_x, sum_x_sq, F, tau, J, F_top, N);


	fill_even_positions(imin, imax, istep, js,
						sum_x, sum_x_sq, tau, F, J, F_top, N);
   
  }
}

void run_SMAWK(double* F, long long* J, double* F_top, long long x0, long long x1, long long y0, long long y1, double tau, double* sum_x, double* sum_x_sq, size_t N) 
{
	//F is where the values are put
	//F_top is where they're calculated from (needed for bottom SMAWK)

	//throw std::exception("This is broken");
  // Assumption: each cluster must have at least one point.

  std::vector<long long> js(x1 -  x0 + 1);
  long long abs = x0;
  std::generate(js.begin(), js.end(), [&] { return abs++; } );

  SMAWK(y0, y1, 1, js, sum_x, sum_x_sq, F, tau, J, F_top, N);
	//Note the swapped axes. We need to obtain the column minima here, 
	//unlike the original row minima
}

