module math;

import std.traits : ReturnType;
import std.stdio;

import mir.ndslice;
import mir.sparse;
// import ldc.attributes : fastmath;

// @fastmath:

alias CSMatrix = ReturnType!(compress!(uint, uint, Contiguous, 2, FieldIterator!(SparseField!double)));
alias CSIntMatrix = ReturnType!(compress!(uint, uint, Contiguous, 2, FieldIterator!(SparseField!int)));

auto max(T)(T a, T b) {
    return a > b ? a : b;
}

auto min(T)(T a, T b) {
    return a > b ? b : a;
}

auto norm_one(T)(T u) {
    import std.algorithm.iteration : map, sum;
    import std.math : abs;
    return u.map!abs.sum;
}

auto norm_infinity(T)(T u) {
    import std.algorithm.iteration : map, reduce;
    import std.math : abs, fmax;
    return u.map!abs.reduce!fmax;
}

auto norm_euclidean(T)(T u) {
    import std.algorithm.iteration : map, sum;
    import std.math : pow;
    return u.map!(x => x * x).sum.pow(0.5);
}

double factorial(int n) {
    auto result = 1.0;
    foreach (i; 0..n) {
        result *= i + 1;        
    }
    return result;
}

///
pragma(inline, true):
auto sub_jacobi(
        CSMatrix a, 
        Slice!(double*, 1, Contiguous) x0,
        inout Slice!(double*, 1, Contiguous) b,
        inout Slice!(double*, 1, Contiguous) d, 
        inout double eps) {
    import mir.sparse.blas.gemv : gemv;
    import std.math : fabs;

    auto temp = uninitSlice!double(x0.length);
    core_loop:
    gemv(-1.0, a, x0, 0.0, temp);
    temp[] = (temp + b) / d;
    foreach(immutable q; iota!int(cast(int) x0.length)) {
        if(fabs(x0[q] - temp[q]) > eps) {
            x0[] = temp;
            goto core_loop;
        }
    }
    x0[] = temp;
}

///
pragma(inline, true): @safe @nogc nothrow pure 
void sub_sor(
        CSMatrix a, 
        Slice!(double*, 1, Contiguous) x,
        inout Slice!(double*, 1, Contiguous) b,
        inout Slice!(double*, 1, Contiguous) d, 
        inout double omega,
        inout double eps
        // int max_iters
        ) {
    import mir.sparse.blas.dot : dot;
    import std.math : fabs;

    core_loop:
    // --max_iters;
    auto a_copy = a;
    auto done = true;
    foreach(immutable i; iota!int(cast(int) x.length)) {
        immutable prev_xi = x[i];
        x[i] = (1 - omega) * x[i] + omega * (b[i] - dot(a_copy.front, x)) / d[i];
        a_copy.popFront;
        done = done && fabs(x[i] - prev_xi) < eps;
    }
    if(!done /*&& (max_iters > 0)*/) goto core_loop;
    // writeln(iters);
}

auto jacobi(
        CSMatrix a, 
        Slice!(double*, 1, Contiguous) b,            
        Slice!(double*, 1, Contiguous) d, 
        double tol = 1e-8, int maxIters = 0) {

    import mir.sparse.blas.gemv : gemv;

    if(maxIters < 1) {
        maxIters = int.max;
    }

    immutable n = cast(int) b.length;
    auto x = slice!double([n], 1.0 / n);
    auto y = x.slice;

    auto iters = 0;
    while(iters++ < maxIters) {
        gemv(-1.0, a, y, 0.0, x);
        x[] += b; 
        x[] /= d;

        immutable norm = norm_infinity(x - y);
        writefln("iteration %s, norm = %s", iters, norm);

        if(norm <= tol)
            break;
    
        y[] = x;
    }

    // import std.stdio;
    // writeln("iterations: ", iters);
    return x;
}

auto jacobi(T, Q)(T a, Q b, double tol = 1e-8, int maxIters = 0) {
    import mir.sparse.blas.gemv : gemv;
    
    if(maxIters < 1) {
        maxIters = int.max;
    }

    immutable n = cast(int) b.length;
    auto x = slice!double([n], 0.0);
    auto y = x.slice;
    auto d = a.diagonal.slice;
    a.diagonal[] = 0.0;
    auto compressed_lu = a.compress();

    auto iters = 0;
    double norm;
    while(iters++ < maxIters) {
        gemv(-1.0, compressed_lu, y, 0.0, x);
        x[] += b;
        x[] /= d;

        norm = norm_infinity(x - y);
        if(norm <= tol) {
            break;
        }
    
        y[] = x[];
    }

    // import std.stdio;
    // if(iters > maxIters) {
    //     writefln("Jacobi: all %s iterations consumed. Norm = %s.", maxIters, norm);
    // }
    a.diagonal[] = d[];
    return x;
}

auto sor(T, Q)(T a, Q b, double omega, double tol = 1.0e-8, int maxIters = 100) {
    import lubeck : mtimes;

    // auto sw = StopWatch(AutoStart.yes);
    // double time_mtimes = 0.0;
    
    import std.algorithm.iteration : reduce;
    import std.range : iota;
    import std.math : fabs;
    import mir.sparse.blas.dot : dot;

    immutable n = cast(int) b.length;
    auto x = slice!double([n], 0.0);
    auto y = x.slice;
    int iters = 0, multiplications = 0;

    // auto sparse_a = sparse!double(n, n);
    // foreach(item; a.byKeyValue()) {
    //     sparse_a[item.key.i, item.key.j] = item.value;
    // }
    // auto compressed_sparse_a = sparse_a.compress();
    while(true) {

        // sw.reset();
        foreach(immutable i; iota!int(n)) {
            // immutable s = mtimes(a[i], x) - a[i, i] * x[i];
            // x[i] += (-x[i] + (b[i] - s) / a[i, i]) * omega;
            
            immutable s = mtimes(a[i], x);
            // auto v = sparse_a[i].compress();
            // immutable s = dot(v, v);
            x[i] += omega * (b[i] - s) / a[i, i];
            ++multiplications;
        }
        // time_mtimes += sw.peek.total!"msecs";
        
        immutable norm = reduce!((a, b) => max(fabs(a), fabs(b)))(0.0, x[] - y[]);
        if(norm < tol) {
            // writeln(iters, " iterations. ", multiplications, " multiplications.");
            return x;
        }

        y[] = x[];
        ++iters;
    }

    assert(0);
    //writeln("failed convergence in specified iterations");
    // writeln("converged after ", iters, " iterations!");
    // return x;    
}

// auto aosor(Sparse!(double, 2) a, Sparse!(double, 2) b, 
//            double tol = 1e-08, int maxIters = 0) {

//     import mir.sparse.blas.gemv : gemv;
//     import mir.sparse.blas.axpy : axpy;
//     import mir.sparse.blas.dot : dot;
    
//     import std.algorithm.iteration : reduce;
//     import std.range : iota;
//     import std.math : fabs;

//     immutable n = cast(int) b.length;
//     immutable beta = 1.0; 
//     immutable gamma = 1.0;
    
//     // compressed_vector x = new CompressedField!(double)(n);// slice!double([n], 0.0);
//     // compressed_vector y = new CompressedField!(double)(n);// slice!double([n], 0.0);
//     // compressed_vector r = new CompressedField!(double)(n);// slice!double([n], 0.0);
//     // compressed_vector u = new CompressedField!(double)(n);// slice!double([n], 0.0);
//     // compressed_vector v = new CompressedField!(double)(n);// slice!double([n], 0.0);
//     // compressed_vector w = new CompressedField!(double)(n);// slice!double([n], 0.0);
//     // compressed_vector s = new CompressedField!(double)(n);// slice!double([n], 0.0);
//     // compressed_vector t = new CompressedField!(double)(n);// slice!double([n], 0.0);

//     Sparse!(double, 2) l;
//     foreach(immutable k; iota(1, n)) {
//         l[k][0 .. k-1] = -a[k][0 .. k-1];
//     }

//     auto d = a.diagonal.slice;
//     auto compressed_a = a.compress();
//     auto compressed_b = b.compress();
//     auto compressed_l = l.compress();

//     alias compressed_vector = typeof(compressed_b);
//     auto x = slice!double([n], 0.0); /*compressed_vector*/ 
//     auto y = slice!double([n], 0.0); /*compressed_vector*/ 
//     auto r = slice!double([n], 0.0); /*compressed_vector*/ 
//     auto u = slice!double([n], 0.0); /*compressed_vector*/ 
//     auto v = slice!double([n], 0.0); /*compressed_vector*/ 
//     auto w = slice!double([n], 0.0); /*compressed_vector*/ 
//     auto s = slice!double([n], 0.0); /*compressed_vector*/ 
//     auto t = slice!double([n], 0.0); /*compressed_vector*/ 

//     int iters = 0;
//     while(iters < maxIters) {

//         // r = b - A x
//         gemv(-1.0, compressed_a, x, 0.0, r);
//         axpy(1.0, b, r.field);
//         // r[] += b;

//         // u = L r; v = A r; t = L u; s = A u; w = A t
//         gemv(1.0, compressed_l, r, 0.0, u);
//         gemv(1.0, compressed_a, r, 0.0, v);
//         gemv(1.0, compressed_l, u, 0.0, t);
//         gemv(1.0, compressed_a, u, 0.0, s);
//         gemv(1.0, compressed_a, t, 0.0, w);

//         double[5] delta = [
//             2 * beta * dot(r, s) - dot(v, v),
//             (beta * beta + 2 * gamma * gamma) * dot(r, w) - 3 * beta * dot(v, s),
//             (beta * beta + 3 * gamma * gamma) * dot(v, w) + 2 * beta * beta * dot(s, s),
//             beta * (beta * beta + 4 * gamma * gamma) * dot(s, w),
//             gamma * gamma * (beta * beta + 2 * gamma * gamma) * dot(w, w)
//         ];
//         delta[] /= dot(r, v);
        
//         {/* get omega from newton-raphson */}
//         auto omega = 1.0;
//         while(true) {
//             immutable f = 1 + delta[0] * omega + delta[1] * omega^^2 - delta[2] * omega^^3 - delta[3] * omega^^4 - delta[4] * omega^^5;
//             immutable fp = delta[0] + 2 * delta[1] * omega - 3 * delta[2] * omega^^2 - 4 * delta[3] * omega^^3 - 5 * delta[4] * omega^^4;
//             immutable corr = -f/fp;
            
//             if(fabs(corr) < tol * 10)
//                 break;
            
//             omega += corr;
//         }

//         // compute y from (D - omega L)y == r
//         y[0] = r[0] / d[0];
//         foreach(immutable k; iota(1, n)) {
//             y[k] = (r[k] - omega * dot(a[k, 1 .. k-1], y[1 .. k-1])) / d[k];
//         }

//         // x = x + omega y
//         axpy(omega, y, x);
//         // x[] += y;
        
//         immutable norm = y.map!fabs.reduce!max;
//         if(norm < tol) {
//             return x;
//         }

//         ++iters;
//     }    
// }


//auto InitializePartitions(inout int n, inout int max_m) {
auto getPartition(inout int k, inout int max_m) {
	import std.range;

	int x, q;
	int n = k;
	int m = 1;
	SortedRange!(int[], "a > b")[] partition;
	auto a = new int[](n + 1);
	
	while(true) {
		a[m] = n;
		q = m - (n == 1);

		while(true) {
			if(m <= max_m) {
				//writeln(a[1 .. m + 1]);
				partition ~= assumeSorted!"a > b"(a[1 .. m + 1].dup);
			}

			if(a[q] != 2)
				break;

			a[++m] = a[q--] = 1;
		}

		if(q == 0) {
			//writeln("  P(", k, ", ", max_m, ") = ", partition);
			return partition;
		}

		x = --a[q];
		n = m - q + 1;
		m = q + 1;

		while(n > x) {
			a[m++] = x;
			n -= x;
		}
	}

	assert(0);
}

struct Partition {
	pure nothrow:

	this(inout int n, inout int m = 0) {
		m_ = m ? (m < n ? m : n) : n;
		k_ = 1;
		a_ = new byte[](n + 1);
		a_[1] = cast(byte) n;
		empty_ = false;
		popFront();
	}

	@property auto empty() { return empty_; }
	@property auto front() { return a_[0 .. k_ + 1]; }
	void popFront() {
		if(k_ == 0) {
			empty_ = true;
			return;
		}

		immutable x = cast(byte) (a_[k_ - 1] + 1);
		auto y = cast(byte) (a_[k_] - 1);
		--k_;
		while(x <= y && k_ < m_ - 1) {
			a_[k_] = x;
			y -= x;
			++k_;
		}
		a_[k_] = cast(byte) (x + y);
	}

	private:
	bool empty_;
	int m_, k_;
	byte[] a_;
}
