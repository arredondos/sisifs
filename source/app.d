import std.stdio;
import std.range;
import std.format : format;
import std.datetime.stopwatch;
import std.conv : to, ConvException;

import afs;
import math;
import darg;

int main(string[] args) {
	import std.algorithm.iteration : sum, map;
	import std.parallelism : totalCPUs;
	import std.exception : enforce;
	import std.math : ceil;
	
	immutable help = helpString!Options;
    Options options;

    try {
        options = parseArgs!Options(args[1 .. $]);
    }
    catch (ArgParseError e) {
        writeln(e.msg);
        write(help);
        return 1;
    }
    catch (ArgParseHelp e) { // Help was requested
        write(help);
        return 0;
    }

	// Parsing n
	int[] n;
	try {
		n ~= options.n.to!int;
	}
	catch (ConvException e0) {
		try {
			n = options.n.to!(int[]);
		}
		catch (ConvException e1) {
			enforce(0, "Invalid parameter (n): the number of islands could not be parsed.");
		}
	}
	foreach(ni; n) {
		enforce(ni > 1, "Invalid parameter (n): the number of islands must be greater than 1.");
	}

	// Parsing k
	byte[] k;
	try {
		k ~= options.k.to!byte;
	}
	catch (ConvException e0) {
		try {
			k = options.k.to!(byte[]);
		}
		catch (ConvException e1) {
			enforce(0, "Invalid parameter (k): the number of sampled lineages could not be parsed.");
		}
	}
	foreach(ki; k) {
		enforce(ki > 1, "Invalid parameter (k): the number of sampled lineages must be greater than 1.");
	}

	// Parsing M
	double[] M;
	try {
		M = [options.M.to!double];
	}
	catch (ConvException e0) {
		try {
			M = options.M.to!(double[]);
		}
		catch (ConvException e1) {
			enforce(0, "Invalid parameter (M): the migration rate could not be parsed.");
		}
	}
	foreach(Mi; M) {
		enforce(Mi > 0, "Invalid parameter (M): the migration rate must be a positive number.");
	}

	// Parsing c
	double[] c;
	try {
		c = [options.c.to!double];
	}
	catch (ConvException e0) {
		try {
			c = options.c.to!(double[]);
		}
		catch (ConvException e1) {
			enforce(0, "Invalid parameter (c): the relative deme size could not be parsed.");
		}
	}
	foreach(ci; c) {
		enforce(ci > 0, "Invalid parameter (c): the relative deme size must be a positive number.");
	}

	// Parsing sv
	immutable sv_sum = options.sv.to!(double[]).sum;
	byte[][] sv;
	if(k.length == 1) {
		enforce(sv_sum == k[0].to!double, "Invalid parameter (sv): the sampling vector must contain as many lineages as specified by the parameter k.");
		sv = [options.sv.to!(double[]).map!(to!byte).array];
	}
	else {
		enforce(sv_sum == 1.0, "Invalid parameter (sv): when specifying multiple values for k, the sampling vector must be a probability distribution.");
		foreach(byte ki; k) {
			auto remaining = ki;
			sv ~= options.sv.to!(double[]).map!(x => () {
				auto kii = min(remaining, ceil(x * ki).to!byte);
				remaining -= kii;
				return kii;
			}()).array;
		}
	}

	// Parsing reps
	int reps;
	try {
		reps = options.reps.to!int;
	}
	catch (ConvException e0) {
		enforce(0, "Invalid parameter (reps): the number of computation repetitions could not be parsed.");
	}
	enforce(reps > 0, "Invalid parameter (reps): the number of computation repetitions must be a positive integer.");

	// Parsing tol
	double tol;
	try {
		tol = options.tol.to!double;
	}
	catch (ConvException e0) {
		enforce(0, "Invalid parameter (e): the error tolerance epsilon could not be parsed.");
	}
	enforce(tol > 0, "Invalid parameter (e): the error tolerance epsilon must be a positive value.");

	// Parsing d
	int d;
	try {
		d = options.d.to!int;
	}
	catch (ConvException e0) {
		enforce(0, "Invalid parameter (d): the number of introspection steps could not be parsed.");
	}
	enforce(d > 0, "Invalid parameter (d): the number of introspection steps must be a positive integer.");

	// Parsing omega
	double omega;
	try {
		omega = options.omega.to!double;
	}
	catch (ConvException e0) {
		enforce(0, "Invalid parameter (omega): the relaxation factor could not be parsed.");
	}
	enforce(omega > 0 && omega < 2, "Invalid parameter (omega): the relaxation factor must be in the interval (0; 2).");

	// Parsing threads
	int threads;
	if(options.threads == threads0) {
		threads = totalCPUs - 1;
	}
	else {
		try{
			threads = options.threads.to!int;
		}
		catch (ConvException e0) {
			enforce(0, "Invalid parameter (threads): the number of additional execution threads could not be parsed.");
		}
	}
	enforce(threads >= 0, "Invalid parameter (threads): the number of additional execution threads must be a non-negative integer.");
	
	// Parsing v
	int v;
	try {
		v = options.v.to!int;
	}
	catch (ConvException e0) {
		enforce(0, "Invalid parameter (v): the level of verbosity could not be parsed.");
	}
	enforce(d >= 0, "Invalid parameter (v): the level of verbosity must be 0, 1 or 2.");

	// performing requested computations
	writefln("reps = %s, e = %s, d = %s, omega = %s, threads = %s, verbosity = %s.", reps, tol, d, omega, threads, v);
	writeln();
	foreach(ci; c) {
		foreach(Mi; M) {
			foreach(ni; n) {
				foreach(i; 0..k.length) {
					singleSFSRun(k[i], ni, Mi, ci, sv[i], reps, tol, d, omega, threads, v);
				}
			}
		}
	}
	
	return 0;
}

private:

immutable k0 = "12";
immutable n0 = "10";
immutable M0 = "1.0";
immutable c0 = "1.0";
immutable sv0 = format!"[%s]"(k0);
immutable reps0 = "1";
immutable tol0 = "1e-6";
immutable d0 = "1";
immutable omega0 = "1.25";
immutable threads0 = "default";
immutable v0 = 0;

struct Options {
    @Option("help", "h")
    @Help("Prints this help.")
    OptionFlag help;

	@Option("samples", "k")
	@Help(format!"Total number of haploid lineages in the sample. May be vectorized. Default is %s."(k0))
	string k = k0;

	@Option("demes", "n")
	@Help(format!"Number of demes or islands in the model. May be vectorized. Default is %s."(n0))
	string n = n0;

	@Option("migration", "M")
	@Help(format!"Migration rate M of an n-island model. May be vectorized. Default is %s."(M0))
	string M = M0;

	@Option("size", "c")
	@Help(format!"Relative deme size c in an n-island model. May be vectorized. Default is %s."(c0))
	string c = c0;

	@Option("sv")
	@Help(format!"Sampling vector or initial distribution of sampled lineages across the demes. Default is %s."(sv0))
	string sv = sv0;

	@Option("repetitions", "r")
	@Help(format!"Number of times each SFS computation is performed. Default is %s."(reps0))
	string reps = reps0;

	@Option("epsilon", "e")
	@Help(format!"Maximum absolute error tolerance for the normalized expected SFS. Default is %s."(tol0))
	string tol = tol0;

	@Option("steps", "d")
	@Help(format!"Approximate number of error introspection steps during computation. Default is %s."(d0))
	string d = d0;

	@Option("omega")
	@Help(format!"Relaxation parameter for the SOR method. Default is %s."(omega0))
	string omega = omega0;

	@Option("threads")
	@Help("Number of additional execution threads in parallel workloads. 
		   Default is one less than the total system threads as reported by the OS.")
	string threads = threads0;

	@Option("verbosity", "v")
	@Help(format!"Controls the level of reporting detail. The three possible values are 0, 1 and 2. Default is %s"(v0))
	int v = v0;
}

enum compare_to_panmictic_sfs = false;
enum force_generate_qmat_files = false;
void singleSFSRun(int k, int n, double M, double c, byte[] sv, 
				  int reps, double tol, int d, double omega, int threads, 
				  int v, bool lowmem = false) {
	
	import mir.ndslice : Slice, Contiguous;
	import std.algorithm.sorting : sort;
	sv.sort!("a > b");
	
	auto header = format!"k = %s, n = %s, M = %s, c = %s, sv = %s."(k, n, M, c, sv);
	writeln(header);
	writeln('='.repeat(header.length).to!string);

	static if(force_generate_qmat_files) {
		auto system = generateSymbolicSystem(cast(byte) k, n, threads, v, lowmem);
		auto s_system = serialize(system);
	}
	else {
		auto s_system = loadOrGenerateSystem(cast(byte) k, n, threads, v, lowmem);
	}

	if(v > 0) {
		writeln("system states = ", s_system.offsets[$ - 1][$ - 1]);
	}

	auto sw = StopWatch(AutoStart.no);
	auto best_time = double.max;
	auto best_i = 0;
	
	Slice!(double*, 1, Contiguous) afs;
	foreach (i; 0 .. reps) {
		if(v > 0) {
			writeln();
			if(reps > 1) {
				writefln("[repetition %s]", i + 1);
			}
		}

		afs = expectedNislandAFS(s_system, sv, n, M, c, omega, tol, 100, d, threads, v, &sw);
		static if(compare_to_panmictic_sfs) {
			import afstools : expectedNeutralAFS;
			import std.math : fabs;
			
			auto p_afs = expectedNeutralAFS(k);
			auto error = 0.0;
			foreach(immutable h; iota(k - 1))
				error = max(error, fabs(afs[h] - p_afs[h]));
			writeln("distance from panmictic SFS = ", error);
		}

		auto time = sw.peek.total!"msecs" / 1e3;
		if(time < best_time) {
			best_time = time;
			best_i = i;
		}
	}

	import std.algorithm : reduce;
	if(v > 0) {
		writeln();
	}

	writeln("esfs = ", afs);
	if(reps > 1) {
		writefln("fastest repetition was #%s at %s seconds.", best_i + 1, best_time);
	}
	writeln();
}
