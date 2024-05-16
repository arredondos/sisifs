module math;

import std.traits : ReturnType;
import std.stdio;

import mir.ndslice;
import mir.sparse;
import ldc.attributes : fastmath;

@fastmath:

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

    a.diagonal[] = d[];
    return x;
}

auto sor(T, Q)(T a, Q b, double omega, double tol = 1.0e-8, int maxIters = 100) {
    import lubeck : mtimes;

    import std.algorithm.iteration : reduce;
    import std.range : iota;
    import std.math : fabs;
    import mir.sparse.blas.dot : dot;

    immutable n = cast(int) b.length;
    auto x = slice!double([n], 0.0);
    auto y = x.slice;
    int iters = 0, multiplications = 0;

    while(true) {

        foreach(immutable i; iota!int(n)) {
            immutable s = mtimes(a[i], x);
            x[i] += omega * (b[i] - s) / a[i, i];
            ++multiplications;
        }
        
        immutable norm = reduce!((a, b) => max(fabs(a), fabs(b)))(0.0, x[] - y[]);
        if(norm < tol) {
            return x;
        }

        y[] = x[];
        ++iters;
    }

    assert(0);
}

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
				partition ~= assumeSorted!"a > b"(a[1 .. m + 1].dup);
			}

			if(a[q] != 2)
				break;

			a[++m] = a[q--] = 1;
		}

		if(q == 0) {
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
