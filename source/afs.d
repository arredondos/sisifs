module afs;
import math;

import std.math : fabs;
import std.stdio;
import std.traits;
import std.typecons;
import std.datetime.stopwatch;
import std.container.array : Array;
import std.parallelism;
import std.concurrency;
import core.memory : GC;
import core.thread;

import mir.sparse;
import mir.ndslice;
import mir.ndslice.sorting : sort;
import mir.math.sum : sum;

import ldc.attributes : fastmath;
@fastmath:

alias Index = int[2];
alias MigrationDestination = Tuple!(State, "state", int[2], "poly");
alias CoalescenceDestination = Tuple!(State, "state", int, "count");

///
struct State {
    import std.digest.murmurhash : MurmurHash3, digest;

    ///
    @safe pure nothrow
    this(T, Q)(inout T rows, inout Q cols) if(isIntegral!T && isIntegral!Q) {
        this.rows = cast(byte) rows;
        this.cols = cast(byte) cols;
        payload = slice!byte([this.rows, this.cols], 0);
    }

    ///
    this(Slice!(byte*, 2, Contiguous) slice) @safe pure nothrow @nogc {
        payload = slice;
        rows = cast(byte) slice.length!0;
        cols = cast(byte) slice.length!1;
    }

    size_t toHash() const pure @safe nothrow @nogc {
        immutable prehash = digest!(MurmurHash3!(128, 64))(payload.field);
        return prehash[0] | (prehash[1] << 8) | (prehash[2] << 16) | (prehash[3] << 24); 
    }

    bool opEquals(ref const State q) const @safe pure nothrow @nogc {
        return payload.field == q.payload.field;
    }

    ///
    auto classRepresentative() {
        auto sorted_indexes = iota(cols).slice;
        sorted_indexes.sort!((i, j) => isHeavier(payload.transposed[i], payload.transposed[j]));
        
        auto temp_mat = uninitSlice!byte(rows, cols);
        foreach (immutable c; iota(cols))
            temp_mat.transposed[c][] = payload.transposed[sorted_indexes[c]];
        
        return State(temp_mat);
    }

    private:

    Slice!(byte*, 2, Contiguous) payload;
    byte rows, cols;

    alias payload this;
}

struct StateColumn {
    import std.digest.murmurhash : MurmurHash3, digest;

    ///
    @safe pure nothrow
    this(T)(inout T rows) if(isIntegral!T) {
        this.rows = cast(byte) rows;
        payload = slice!byte([this.rows], 0);
    }

    ///
    this(Slice!(byte*, 1, Contiguous) slice) @safe pure nothrow @nogc {
        payload = slice;
        rows = cast(byte) slice.length;
    }

    size_t toHash() const pure @safe nothrow @nogc {
        immutable prehash = digest!(MurmurHash3!(128, 64))(payload.field);
        return prehash[0] | (prehash[1] << 8) | (prehash[2] << 16) | (prehash[3] << 24); 
    }

    bool opEquals(ref const StateColumn q) const @safe pure nothrow @nogc {
        return payload.field == q.payload.field;
    }

    private:

    Slice!(byte*, 1, Contiguous) payload;
    byte rows;

    alias payload this;    
}

auto trim_vector(inout byte[] v) {
    int i = 0;
    while(i < v.length && v[i] > 0) ++i;
    return v[0..i];
}

///
struct SamplingState {
    import std.digest.murmurhash : MurmurHash3, digest;

    @safe nothrow pure
    this(byte[] sampling_vector, inout int index = -1) {
        this.field = trim_vector(sampling_vector);
        this.index = index;
    }

    @nogc @safe pure nothrow 
    size_t toHash() const {
        immutable prehash = digest!(MurmurHash3!(128, 64))(field);
        return prehash[0] | (prehash[1] << 8) | (prehash[2] << 16) | (prehash[3] << 24); 
    }

    @nogc @safe pure nothrow
    bool opEquals(ref const SamplingState beta) const {
        return field == beta.field;
    }

    private:

    byte[] field;
    int index;
}

struct MacroState {
    @nogc @safe pure nothrow
    size_t toHash() const {
        import std.digest.murmurhash : MurmurHash3, digest;
        immutable prehash = digest!(MurmurHash3!(128, 64))(contribution);
        return prehash[0] | (prehash[1] << 8) | (prehash[2] << 16) | (prehash[3] << 24); 
    }

    @nogc @safe pure nothrow
    bool opEquals(ref const MacroState q) const {
        return contribution == q.contribution;
    }

    // private:
    byte[] contribution;
    alias contribution this;
}

auto state_contribution(inout State q) {
    auto p = new byte[] (q.rows);
    foreach(immutable i; iota(q.rows)) {
        p[i] = cast(byte) sum!"fast"(q[i]);
    }
    return MacroState(p);
}

///
struct SymbolicRatesSystem {
    byte lineages;
    int islands;
    int[MacroState][] macro_maps;
    int[State][][] micro_maps;
    int[2][Index][][] mig_sub_blocks;
    int[Index][][] coal_sub_blocks;
    int[][] offsets;
}

struct SerializableSystem {
    byte lineages;
    SamplingState[] sampling_states;
    byte[][] contributions;

    uint[][][] mig_pointers;
    uint[][][] mig_indexes;
    int[][][] mig_data_0;
    int[][][] mig_data_1;

    uint[][][] coal_pointers;
    uint[][][] coal_indexes;
    int[][][] coal_data;

    int[][][] diag_0, diag_1, diag_c;
    int[][] offsets;
}

auto loadOrGenerateSystem(inout byte lineages, inout int islands, 
                          inout int threads, inout int verbosity,
                          inout bool lowmem) {

    import std.format : format;
    import std.file : exists;

    auto sw = StopWatch(AutoStart.yes);
    SerializableSystem sys;

    auto filename = format!"nipa_nn_k%s_n%s.qmat"(lineages, min(islands, lineages));
    if(exists(filename)) {
        if(verbosity > 0)
            writeln("Loading file ", filename);

        sys = loadSerializableSystem(filename);
    }
    else {
        if(verbosity > 0)
            writeln("Generating Markov system");

        auto system = generateSymbolicSystem(lineages, islands, threads, verbosity, lowmem);
        
        if(verbosity > 0)
            writeln("Generating file ", filename);

        sys = saveSerializableSystem(system, filename, verbosity);
    }

    auto time = sw.peek.total!"msecs" / 1e3;
    if(verbosity > 0)
        writeln("Done. Time to Markov system ready = ", time);

    return sys;
}

auto loadSerializableSystem(string filename) {
    import msgpack : unpack;

    auto file = File(filename, "r");
    auto buffer = new ubyte[] (file.size);
    auto data = file.rawRead(buffer);
    auto s_system = data.unpack!SerializableSystem;
    return s_system;
}

auto saveSerializableSystem(SymbolicRatesSystem system, string filename, inout int verbosity) {
    import msgpack : pack;

    auto file = File(filename, "w");
    
    if(verbosity > 1)
        writeln("  Serializing the data structures");
    auto s_system = serialize(system);
    auto bytes = pack(s_system);
    
    if(verbosity > 1)
        writeln("  Writing to file");
    file.rawWrite(bytes);
    
    return s_system;
}

auto serialize(SymbolicRatesSystem system) {
    import std.range : array;

    immutable non_negative = (inout int x) @safe @nogc pure nothrow => max(0, x);
    immutable k = system.lineages;


    //////////
    static if(export_symbolic_system) {
        auto f = File("./symbolic_system.nb", "w");
    }
    /////////

    byte[][] contributions;
    SamplingState[] sampling_states;
    auto mig_pointers = new uint[][][] (k - 1);
    auto mig_indexes = new uint[][][] (k - 1);
    auto mig_data_0 = new int[][][] (k - 1);
    auto mig_data_1 = new int[][][] (k - 1);
    auto coal_pointers = new uint[][][] (k - 1);
    auto coal_indexes = new uint[][][] (k - 1);
    auto coal_data = new int[][][] (k - 1);
    auto diag_0 = new int[][][] (k - 1);
    auto diag_1 = new int[][][] (k - 1);
    auto diag_c = new int[][][] (k - 1);
    
    foreach(r; /*parallel(*/iota(k - 1)/*, 1)*/) {

        static if(export_symbolic_system) {
            f.writef("mig%sCoefs = Most@{", r + 1);
        }

        immutable ell_H = system.offsets[r + 1][0] - system.offsets[r][0];
        
        immutable P_r = cast(int) system.macro_maps[r].length;
        mig_pointers[r] = new uint[][] (P_r);
        mig_indexes[r] = new uint[][] (P_r);
        mig_data_0[r] = new int[][] (P_r);
        mig_data_1[r] = new int[][] (P_r);
        diag_0[r] = new int[][] (P_r);
        diag_1[r] = new int[][] (P_r);
        diag_c[r] = new int[][] (P_r);

        immutable P_r1 = cast(int) system.macro_maps[r + 1].length;
        coal_pointers[r] = new uint[][] (P_r1);
        coal_indexes[r] = new uint[][] (P_r1);
        coal_data[r] = new int[][] (P_r1);

        if(r == 0) {
            foreach(const item; system.micro_maps[0][0].byKeyValue) {
                sampling_states ~= SamplingState(item.key.field, item.value);
            }
        }
        
        foreach(immutable p; iota!int(P_r)) {
            immutable ell_h = system.offsets[r][p + 1] - system.offsets[r][p];
            contributions ~= state_contribution(system.micro_maps[r][p].keys[0]).contribution;
            
            diag_0[r][p] = new int[] (ell_h);
            diag_1[r][p] = new int[] (ell_h);
            diag_c[r][p] = new int[] (ell_h);
            auto mig_mats_0 = sparse!int(ell_h, ell_h);
            auto mig_mats_1 = sparse!int(ell_h, ell_h);
            foreach(const item; system.mig_sub_blocks[r][p].byKeyValue) {
                immutable v_0 = item.value[0];
                immutable v_1 = item.value[1];
                if(v_0 == 0 && v_1 == 0)
                    continue;

                mig_mats_0[item.key[1], item.key[0]] = v_0;
                mig_mats_1[item.key[1], item.key[0]] = v_1 ? v_1 : -1;

                static if(export_symbolic_system) {
                    f.writef("{%s, %s} -> %s + (%s)*n, ", 
                        system.offsets[r][p] + item.key[0] + 1, 
                        system.offsets[r][p] + item.key[1] + 1, 
                        v_0, v_1);
                }

                diag_0[r][p][item.key[0]] += v_0;
                diag_1[r][p][item.key[0]] += v_1;
            }
            Series!(uint*, int*, 1LU, Contiguous) mig_series_0;
            Series!(uint*, int*, 1LU, Contiguous) mig_series_1;
            decompress(mig_mats_0, mig_pointers[r][p], mig_series_0);
            decompress(mig_mats_1, mig_pointers[r][p], mig_series_1);
            mig_indexes[r][p] = mig_series_0.index.field;
            mig_data_0[r][p] = mig_series_0.data.slice.field;
            mig_data_1[r][p] = mig_series_1.data.map!non_negative.slice.field;
        }

        static if(export_symbolic_system) {
            f.writeln("Null};");
            f.writef("coal%sCoefs = Most@{", r + 1);
        }

        foreach(immutable q; iota!int(P_r1)) {
            immutable ell_v = system.offsets[r + 1][q + 1] - system.offsets[r + 1][q];

            auto coal_mats = sparse!int(ell_v, ell_H);
            foreach(const item; system.coal_sub_blocks[r][q].byKeyValue) {
                auto macro_i = 0;
                auto micro_i = item.key[0];
                foreach(immutable p; iota(P_r)) {
                    immutable l = system.micro_maps[r][p].length;
                    if(micro_i >= l) {
                        micro_i -= l;
                        ++macro_i;
                    } else {
                        break;
                    }
                }
                coal_mats[item.key[1], item.key[0]] = item.value;
                diag_c[r][macro_i][micro_i] += item.value;

                static if(export_symbolic_system) {
                    f.writef("{%s, %s} -> %s, ", 
                        system.offsets[r][0] + item.key[0] + 1, 
                        system.offsets[r + 1][q] + item.key[1] + 1, 
                        item.value);
                }

            }
            Series!(uint*, int*, 1LU, Contiguous) coal_series;
            decompress(coal_mats, coal_pointers[r][q], coal_series);
            coal_indexes[r][q] = coal_series.index.field;
            coal_data[r][q] = coal_series.data.field;
        }

        static if(export_symbolic_system) {
            f.writeln("Null};");
        }

        GC.collect();
    }

    static if(export_symbolic_system) {
        import std.range : iota;
        f.writefln("migCoefs = Union[%(mig%sCoefs, %)Coefs];", iota(1, k));
        f.writefln("coalCoefs = Union[%(coal%sCoefs, %)Coefs];", iota(1, k));
        f.writefln("bigE = %s;", system.offsets[$ - 1][$ - 1]);
        f.writefln("k = %s;", k);
        f.writefln("mQm = m SparseArray[migCoefs, bigE];");
        f.writefln("mQc = c SparseArray[coalCoefs, bigE];");
        f.writefln("mQraw = mQm + mQc;");
    }

    foreach (immutable i; iota(k - 1)) {
        immutable Pi = cast(int)system.offsets[i].length - 1;
        immutable sigma = system.offsets[i][0];
        foreach(immutable p; iota!int(Pi + 1)) {
            system.offsets[i][p] -= sigma;
        }
    }

    return SerializableSystem(
        k, sampling_states, 
        contributions,
        mig_pointers, mig_indexes, mig_data_0, mig_data_1, 
        coal_pointers, coal_indexes, coal_data, 
        diag_0, diag_1, diag_c,
        system.offsets
    );
}

void decompress(V, I = uint, J = size_t, T, size_t N)
        (Slice!(FieldIterator!(SparseField!T), N) slice, 
        out J[] pointers, out Series!(I*, V*, 1LU, Contiguous) compressedData)
        if (is(T : V) && N > 1 && isUnsigned!I) {

    compressedData = slice.iterator._field._table.series!(size_t, T, I, V);
    pointers = new J[] (slice.shape[0 .. N - 1].iota.elementCount + 1);
    pointers[0] = 0;
    pointers[1] = 0;
    size_t k = 1, shift;
    immutable rowLength = slice.length!(N - 1);
    if(rowLength) foreach (ref index; compressedData.index.field) {
        while(true) {
            immutable newIndex = index - shift;
            if(newIndex >= rowLength) {
                pointers[k + 1] = pointers[k];
                shift += rowLength;
                k++;
                continue;
            }
            index = cast(I)newIndex;
            pointers[k] = cast(J) (pointers[k] + 1);
            break;
        }
    }
    pointers[k + 1 .. $] = pointers[k];
}

enum export_symbolic_system = false;
enum report_uncompressed_state_space_size = false;
auto generateSymbolicSystem(inout byte lineages, inout int islands, 
                            inout int threads, inout int verbosity, 
                            inout bool lowmem) {

    import std.algorithm.iteration : cumulativeFold, map;

    import std.range : array, iota;
    defaultPoolThreads(threads);
    
    immutable k = lineages;
    auto macro_maps = new int[MacroState][] (k);
    auto micro_maps = new int[State][][] (k);
    auto mig_sub_blocks = new int[2][Index][][] (k);
    static if(report_uncompressed_state_space_size)
        double[] uncompressed_blocks_sizes;

    immutable partitions = Partition(k).map!dup.array;
    foreach(const a; partitions) {
        immutable i = k - a.length;
        immutable rows = cast(byte) (i + 1);
        auto alpha = new byte[] (rows);
        foreach(immutable a_j; a)
            ++alpha[a_j - 1];
        
        auto psi = MacroState(alpha);
        macro_maps[i][psi] = cast(int) macro_maps[i].length;
        ++micro_maps[i].length;
        ++mig_sub_blocks[i].length;
    }

    foreach (i; parallel(iota(k), 1)) {
        immutable cols = cast(byte) min(islands, k - i);
        foreach (const item; macro_maps[i].byKeyValue()) {
            immutable psi = item.key;
            immutable ell = item.value;
            int[State] map_ip;
            int[2][Index] A_ip;

            static if(report_uncompressed_state_space_size)
                uncompressed_blocks_sizes ~= symbolic_mig_submat(psi, cols, A_ip, map_ip, islands);
            else
                symbolic_mig_submat(psi, cols, A_ip, map_ip, islands);

            micro_maps[i][ell] = map_ip;
            mig_sub_blocks[i][ell] = A_ip;

            if(verbosity > 1)
                writefln("\tA_%s,%s done", i + 1, ell + 1);
        }
        if(lowmem)
            GC.collect();
    }

    // foreach (const a; Partition(k)) {
    //     immutable i = k - a.length;
    //     immutable rows = cast(byte) (i + 1);
    //     immutable cols = cast(byte) min(islands, k - i);
    //     auto alpha = new byte[] (rows);
    //     foreach(immutable a_j; a)
    //         ++alpha[a_j - 1];
        
    //     auto psi = MacroState(alpha);
    //     int[State] map_ip;
    //     int[2][Index] A_ip;

    //     static if(report_uncompressed_state_space_size)
    //         uncompressed_blocks_sizes ~= symbolic_mig_submat(psi, cols, A_ip, map_ip, islands);
    //     else
    //         symbolic_mig_submat(psi, cols, A_ip, map_ip, islands);

    //     macro_maps[i][psi] = cast(int) macro_maps[i].length;
    //     micro_maps[i] ~= map_ip;
    //     mig_sub_blocks[i] ~= A_ip;
    //     // GC.collect();
    //     if(verbosity > 1) {
    //         writefln("\tA_%s,%s done", i + 1, mig_sub_blocks[i].length);
    //     }
    // }

    auto coal_sub_blocks = new int[Index][][] (k - 1);
    // foreach (immutable i; iota(k - 1)) {
    foreach (i; parallel(iota(k - 1), 1)) {
        coal_sub_blocks[i] = new int[Index][] (macro_maps[i + 1].length);
        immutable P_i = macro_maps[i].length;
        immutable cols = micro_maps[i + 1][0].keys[0].cols;
        immutable rows = micro_maps[i + 1][0].keys[0].rows;
        auto work_mat = uninitSlice!byte(rows, cols);
        auto show_mat = uninitSlice!byte(rows, cols);
        auto sigma = 0;
        foreach (immutable p; iota(P_i)) {
            foreach (const item; micro_maps[i][p].byKeyValue()) {
                auto alpha = item.key;
                immutable l0 = item.value;
                foreach(destination; CoalescenceDestinations(alpha, work_mat, show_mat)) {
                    auto beta = destination.state;
                    auto psi = state_contribution(beta);
                    immutable q = macro_maps[i + 1][psi];
                    immutable l1 = micro_maps[i + 1][q][beta];
                    coal_sub_blocks[i][q][[sigma + l0, l1]] += destination.count;
                }
            }
            sigma += cast(int) micro_maps[i][p].length;
        }
        if(verbosity > 1)
            writefln("\tB_%s done", i + 1);

        if(lowmem)
            GC.collect();
    }

    auto offsets = new int[][] (k);
    foreach (i; iota(k)) {
        immutable P_i = macro_maps[i].length;
        immutable prev = (i == 0) ? 0 : offsets[i - 1][$ - 1];
        offsets[i] = [prev] ~ iota(P_i).map!(p => cast(int) micro_maps[i][p].length)
            .cumulativeFold!((a, b) => a + b)(prev).array;
    }

    static if(report_uncompressed_state_space_size) {
        import std.algorithm.iteration : sum;
        writeln("Uncompressed state space size = ", uncompressed_blocks_sizes.sum);
    }

    return SymbolicRatesSystem(
        lineages, 
        islands, 
        macro_maps, 
        micro_maps, 
        mig_sub_blocks, 
        coal_sub_blocks, 
        offsets
    );
}

auto symbolic_mig_submat(
        inout MacroState psi, 
        inout byte cols, 
        ref int[2][Index] qmat, 
        ref int[State] map,
        inout int n) {

    immutable rows = psi.length;
    auto indexes = uninitSlice!byte(cols);
    auto work_mat = uninitSlice!byte(rows, cols);
    auto show_mat = uninitSlice!byte(rows, cols);

    auto proto_state = State(rows, cols);
    proto_state.transposed[0][] = psi[];
    map[proto_state] = 0;

    static if(report_uncompressed_state_space_size)
        double[] states_class_cardinals;

    auto unvisitedStates = Array!(State)(map.keys);
    while(!unvisitedStates.empty) {
        auto alpha = unvisitedStates.back;
        unvisitedStates.removeBack;
        immutable l0 = map[alpha];
        
        static if(report_uncompressed_state_space_size)
            states_class_cardinals ~= class_cardinal(alpha, n); 
        
        foreach(ref destination; MigrationDestinations(alpha, indexes, work_mat, show_mat)) {
            if(destination.poly == [0, 0])
                continue;
                
            const p_l1 = destination.state in map;
            if(p_l1) {
                immutable l1 = *p_l1;
                if(l0 == l1) 
                    continue;

                auto pIndex = [l0, l1] in qmat;
                if(pIndex)
                    (*pIndex)[] += destination.poly[];
                else
                    qmat[[l0, l1]] = destination.poly;
            }
            else {
                auto beta = State(destination.state.slice);
                immutable l1 = cast(int) map.length;
                qmat[[l0, l1]] = destination.poly;
                unvisitedStates.insert(beta);
                map[beta] = l1;
            }
        }
    }
    import std.algorithm.iteration : sum;
    static if(report_uncompressed_state_space_size)
        return states_class_cardinals.sum;
}

enum report_sub_sizes = false;
auto expectedNislandAFS(
        SerializableSystem sys, byte[] samplingVector, inout int islands,
        inout double migration_rate, inout double deme_size,
        inout double omega = 1.25, inout double tol = 1e-6, 
        int max_iterations = 20, int min_steps = 2, int threads = 4, int verbosity = 0,
        StopWatch* sw = null) {
    
    import mir.sparse.blas.gemv : gemv;
    import mir.sparse.blas.dot : dot;
    import mir.ndslice.topology : zip;
    import std.algorithm.searching : find;
    import std.range : iota, front, popFront;
    
    sw.reset();
    sw.start();
    auto cumul_time = 0.0, sub_cumul_time = 0.0;
    defaultPoolThreads(threads);
    immutable workers = 1;

    immutable k = sys.lineages;
    immutable M = migration_rate / (2 * (islands - 1));
    auto mig_mats = new CSMatrix[][] (k - 1);
    auto coal_mats = new CSMatrix[][] (k - 1);
    auto diagonals = new Slice!(double*, 1, Contiguous)[][] (k - 1);
    foreach(r; parallel(iota(k - 1), workers)) {
        immutable P_r = cast(int)sys.offsets[r].length - 1;
        mig_mats[r] = new CSMatrix[] (P_r);
        diagonals[r] = new Slice!(double*, 1, Contiguous)[] (P_r);
        foreach(immutable p; iota!int(P_r)) {
            auto mig_data = 
                zip!true(sys.mig_data_0[r][p], sys.mig_data_1[r][p])
                .map!((inout a, inout b) @safe @nogc pure nothrow => 
                    -M * a - islands * M * b).slice;
            mig_mats[r][p] = chopped(series(sys.mig_indexes[r][p], mig_data), sys.mig_pointers[r][p]);

            diagonals[r][p] = 
                zip!true(sys.diag_0[r][p], sys.diag_1[r][p], sys.diag_c[r][p])
                .map!((inout a, inout b, inout c) @safe @nogc pure nothrow => 
                    M * a + islands * M * b + c / deme_size).slice;
        }

        immutable P_r1 = (r == k - 2) ? 1 : cast(int) sys.offsets[r + 1].length - 1;
        coal_mats[r] = new CSMatrix[] (P_r1);
        foreach(immutable q; iota!int(P_r1)) {
            auto coal_data = sys.coal_data[r][q].map!((inout x) @safe @nogc pure nothrow => -x / deme_size).slice;
            coal_mats[r][q] = chopped(series(sys.coal_indexes[r][q], coal_data), sys.coal_pointers[r][q]);
        }
    }

    if(verbosity > 0) { ///////////////////////////
        auto time = sw.peek.total!"msecs" / 1e3;
        writefln("Time to Markov system instantiation = %s", time);
        cumul_time = time;
    } /////////////////////////////////////////////

    auto rhs = new Slice!(double*, 1, Contiguous)[][] (k - 1);
    auto e_t = new Slice!(double*, 1, Contiguous)[] (k - 1);
    auto e_afs_curr = uninitSlice!(double)(k - 1);
    auto e_afs_prev = slice!double([k - 1], 0.0);
    foreach (immutable r; iota(k - 1)) {
        immutable P_r = cast(int)sys.offsets[r].length - 1;
        rhs[r] = new Slice!(double*, 1, Contiguous)[] (P_r);

        immutable E_r = sys.offsets[r][$ - 1];
        e_t[r] = slice!double([E_r], 0.0);

        static if(report_sub_sizes)
            writeln("\ni = ", r);

        foreach(immutable p; iota!int(P_r)) {
            immutable E_rp = sys.offsets[r][p + 1] - sys.offsets[r][p];
            rhs[r][p] = uninitSlice!double(E_rp);

            static if(report_sub_sizes)
                writefln("A_%s,%s; %s; %s", r, p, sys.diag_0[r][p].length, sys.mig_indexes[r][p].length);
        }
    }


    auto sampling_state = SamplingState(samplingVector);
    auto sampling_index = sys.sampling_states.find(sampling_state).front.index;
    rhs[0][0][] = 0.0;
    rhs[0][0][sampling_index] = 1.0;

    static if(export_symbolic_system) {
        auto f = File("./symbolic_system.nb", "a");
        f.writefln("offsets = {%(%s, %)};", sys.offsets[0..$-1].map!(x => x[$-1]));
        f.writefln("samplingIndex = %s;", sampling_index + 1);
        f.writefln("mPsi = Table[0, %s, %s];", sys.offsets[$ - 1][$ - 1] - 1, k - 1);
    }

    if(verbosity > 0) { //////////////////////////////
        auto time = sw.peek.total!"msecs" / 1e3;
        writefln("Time to solver initialization = %s (delta = %s)", time, time - cumul_time);
        if(verbosity > 1) {
            writeln("Sampling at state ", sampling_index + 1);
            writeln();
            writeln("  [iteration] - [iteration time] - [error] - [current esfs]");
            sub_cumul_time = time;
        }
        cumul_time = time;
    } ////////////////////////////////////////////////

    auto iters = 0;
    if(max_iterations < 1)
        max_iterations = int.max;

    import std.range : chain, repeat, takeExactly;
    auto cumul_offsets = 0;

    immutable delta_tol = tol ^^ (1.0 / min_steps);
    main_loop: while(iters++ < max_iterations) {
        immutable sub_tol = max(1e-14, delta_tol ^^ iters);//tol / (100 ^^ iters);
        
        e_afs_curr[] = 0.0;
        auto macro_index = 0;
        foreach(immutable i; 0 .. k - 1) {
            immutable Pi = cast(int)sys.offsets[i].length - 1;
            foreach(p; parallel(iota!int(Pi), workers)) {
                if(i > 0) gemv(-1.0, coal_mats[i - 1][p], e_t[i - 1], 0.0, rhs[i][p]);
                sub_sor(
                    mig_mats[i][p],
                    e_t[i][sys.offsets[i][p] .. sys.offsets[i][p + 1]],
                    rhs[i][p],
                    diagonals[i][p],
                    omega, sub_tol
                    // , max_iterations
                );
            }
            foreach(immutable p; iota!int(Pi)) {
                const contribution = sys.contributions[macro_index++];

                static if(export_symbolic_system) {
                    if(iters == 1) {
                        f.writefln("mPsi[[%s ;; %s]] = Table[{%(%s, %)}, %s];", 
                            cumul_offsets + sys.offsets[i][p] + 1, 
                            cumul_offsets + sys.offsets[i][p + 1],
                            chain(contribution, repeat(0).takeExactly(k - 1 - contribution.length)),
                            sys.offsets[i][p + 1] - sys.offsets[i][p]);
                    }
                }

                immutable macro_time = sum(e_t[i][sys.offsets[i][p] .. sys.offsets[i][p + 1]]);
                foreach(immutable h; iota!byte(cast(byte) contribution.length)) {
                    e_afs_curr[h] += macro_time * contribution[h];
                }
            }
            cumul_offsets += sys.offsets[i][$ - 1];
        }
        e_afs_curr[] /= sum(e_afs_curr);

        double actual_err = 0.0;
        foreach(immutable h; iota(k - 1)) {
            actual_err = max(actual_err, fabs(e_afs_curr[h] - e_afs_prev[h]));
        }

        if(verbosity > 1) {
            auto time = sw.peek.total!"msecs" / 1e3;
            writefln("  %s - %s - %s - %s", iters, time - sub_cumul_time, actual_err, e_afs_curr);
            sub_cumul_time = time;
        }

        e_afs_prev[] = e_afs_curr;
        if(actual_err > tol) {
            continue main_loop;
        }

        break;
    }
    
    sw.stop;
    if(verbosity > 0) { /////////////////////////////////////////
        auto time = sw.peek.total!"msecs" / 1e3; 
        if(verbosity > 1) {
            writeln();
        }
        writefln("Done. Time to system solve = %s (delta = %s)", time, time - cumul_time);
        writeln("Iterations = ", iters);
    } ///////////////////////////////////////////////////////////
    
    return e_afs_curr;
}

private:

@safe pure nothrow @nogc
auto isHeavier(T)(inout T a, inout T b) {
    import std.range : iota;
    foreach (immutable i; iota!byte(cast(byte) a.length))
        if(a[i] != b[i]) return a[i] > b[i];
    
    return false;
}

auto class_cardinal(State q, int n) {
    int[StateColumn] frequencies;
    immutable empty_col_index = empty_column_index(q);

    foreach (byte i; 0 .. empty_col_index) {
        auto current_col = StateColumn(q.transposed[i].slice);
        auto pfreq = current_col in frequencies;
        if(pfreq) {
            *pfreq += 1;
        }
        else {
            frequencies[current_col] = 1;
        }
    }
    if(empty_col_index < n) {
        auto empty_col = StateColumn(q.rows);
        frequencies[empty_col] = n - empty_col_index;
    }
    
    auto denominator = 1.0;
    foreach (freq; frequencies.values) {
        denominator *= factorial(freq);
    }
    immutable cardinal = factorial(n) / denominator;
    return cardinal;
}

@nogc:

auto empty_column_index(State q) {
    auto empty_col = cast(byte) q.cols;
    cols: foreach(immutable j; iota(q.cols)) {
        foreach(immutable coef; q.transposed[j]) {
            if(coef > 0) 
                continue cols;
        }

        empty_col = cast(byte) j;
        break;
    }
    return empty_col;
}

struct MigrationDestinations {
    @nogc
    auto this(State p,
              Slice!(byte*, 1, Contiguous) indexes, 
              Slice!(byte*, 2, Contiguous) work_mat,
              Slice!(byte*, 2, Contiguous) show_mat) {

        work_mat[] = p;
        q = State(work_mat);
        q_rep = State(show_mat);
        this.indexes = indexes;

        empty_col = empty_column_index(q);
        if(empty_col == q.cols) {
            empty_islands = [0, 0];
        }
        else {
            empty_islands = [-cast(int) empty_col, 1];
        }

        _empty = false;
        i = j = 0;
        k = 1;

        visitCurrentOrFindAndVisitNext();
    }

    @property @nogc auto empty() { return _empty; }
    @property @nogc auto front() { return MigrationDestination(q_rep, poly); }
    @nogc void popFront() {
        if(k < q.cols) {
            --q[i, k];
            ++q[i, j];
        }

        ++k;
        visitCurrentOrFindAndVisitNext();
    }

    @nogc auto visitCurrentOrFindAndVisitNext() {
        if(k == j) {
            ++k;
        }

        if(k > empty_col) {
            ++j;
            k = 0;
        }

        while(i < q.rows) {
            while(j < empty_col) {
                if(q[i, j] == 0) {
                    ++j;
                    k = 0;
                    continue;
                }

                if(k < empty_col) {
                    poly = [q[i, j], 0];
                }
                else {
                    poly = [empty_islands[0] * q[i, j], empty_islands[1] * q[i, j]];
                }

                if(k == q.cols) {
                    q_rep[] = q;
                }
                else {
                    --q[i, j];
                    ++q[i, k];
                    updateClassRepresentative();
                }

                return;
            }
            ++i;
            j = 0;
            k = 1;
        }
        _empty = true;
    }

    @nogc @safe nothrow pure
    bool larger(inout byte i) inout {
        import std.range : iota;

        if(indexes[i - 1] >= empty_col)
            return true;
        
        foreach (immutable k; iota!byte(q.rows))
            if(q[k, indexes[i]] != q[k, indexes[i - 1]]) 
                return q[k, indexes[i]] > q[k, indexes[i - 1]];
        
        return false;
    }

    @nogc @safe nothrow pure
    void swap_left(inout byte s) {
        immutable temp = indexes[s];
        indexes[s] = indexes[s - 1];
        indexes[s - 1] = temp;
    }

    @nogc @safe nothrow pure
    void updateClassRepresentative() {
        indexes[] = iota!byte(q.cols);

        byte s = k;
        while(s > 0 && larger(s))
            swap_left(s--);

        s = (k > j && s <= j) ? cast(byte)(j + 2) : cast(byte)(j + 1);
        while(s < q.cols && larger(s))
            swap_left(s++);

        import std.range : iota;
        foreach (immutable j; iota!byte(q.cols))
            q_rep.transposed[j][] = q.transposed[indexes[j]];
    }

    Slice!(byte*, 1, Contiguous) indexes;
    int[2] empty_islands, poly;
    byte i, j, k, empty_col;
    State q, q_rep;
    bool _empty;
}

struct CoalescenceDestinations {
    @nogc
    auto this(State p,
              Slice!(byte*, 2, Contiguous) work_mat,
              Slice!(byte*, 2, Contiguous) show_mat) {

        immutable new_cols = work_mat.length!1;
        work_mat[0 .. $-1, 0 .. $] = p[0 .. $, 0 .. new_cols];
        work_mat[$-1, 0 .. $] = cast(byte) 0;

        q = State(work_mat);
        q_rep = State(show_mat);

        i = j = k = 0;
        _empty = false;
        visitCurrentOrFindAndVisitNext();
    }

    @property @nogc auto empty () { return _empty; }
    @property @nogc auto front() { return CoalescenceDestination(q_rep, multiplier); }
    @nogc void popFront() {
        ++q[j, i];
        ++q[k, i];
        --q[j + k + 1, i];
        ++k;
        visitCurrentOrFindAndVisitNext();
    }

    private:

    @nogc @safe nothrow pure 
    void visitCurrentOrFindAndVisitNext() {
        while(i < q.cols) {
            while(j < q.rows) {
                if(q[j, i] > 0) {
                    --q[j, i];
                    while(k < q.rows) {
                        if(q[k, i] > 0) {
                            --q[k, i];
                            ++q[j + k + 1, i];
                            multiplier = (j == k) ? 
                                (q[j, i] + 2) * (q[j, i] + 1) / 2 :
                                (q[j, i] + 1) * (q[k, i] + 1);
                            
                            updateClassRepresentative();
                            return;
                        }
                        ++k;
                    }
                    ++q[j, i];
                }
                ++j;
                k = j;
            }
            ++i;
            j = k = 0;
        }
        _empty = true;
    }

    @nogc @safe nothrow pure
    bool larger(inout byte c) inout {
        import std.range : iota;

        foreach (immutable k; iota!byte(q.rows))
            if(q[k, c] != q[k, i]) return q[k, c] > q[k, i];
        
        return false;
    }

    @nogc @safe nothrow pure
    void updateClassRepresentative() {
        byte b = cast(byte)(i + 1);
        while(b < q.cols && larger(b)) {
            ++b;
        }
        
        if(b == i + 1) {
            q_rep[] = q;    
        } else {
            q_rep.transposed[0 .. i][]   = q.transposed[0 .. i];
            q_rep.transposed[i .. b - 1][] = q.transposed[i + 1 .. b];
            q_rep.transposed[b - 1][]    = q.transposed[i];
            q_rep.transposed[b .. $][]   = q.transposed[b .. $];
        }
    }

    int multiplier;
    State q, q_rep;
    byte i, j, k;
    bool _empty;
}
