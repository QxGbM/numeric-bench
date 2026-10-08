#pragma once

#include <hyacin.h>
#include <vector>
#include <array>
#include <complex>
#include <random>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <stdexcept>
#include <type_traits>
#include <cuda_fp16.h>

using blas_int = int; // LP64 for blas
const int32_t kernel_runs = 3;
double kernel_time = 0., rep_time = 0., comm_time = 0.;

template <class T, class S> inline T conv(S x) {
  if constexpr(std::is_same_v<T, __half2> && std::is_same_v<S, std::complex<double>>) { return make_half2(__double2half(x.real()), __double2half(x.imag())); } else
  if constexpr(std::is_same_v<T, std::complex<double>> && std::is_same_v<S, __half2>) { return std::complex<double>(double(x.x), double(x.y)); } else
  { return T(x); }
}

template <class T, class S> inline void copy2d(int32_t M, int32_t N, const S* A, int32_t lda, T* B, int32_t ldb)
{ for (int32_t j = 0; j < N; ++j) { std::transform(&A[int64_t(j) * int64_t(lda)], &A[int64_t(j) * int64_t(lda) + int64_t(M)], &B[int64_t(j) * int64_t(ldb)], [](S x) { return conv<T>(x); }); } }

std::string replace_suffix(const std::string& str, const std::string& suffix) {
  std::string result = str;
  std::size_t pos = result.rfind('.');
  if (pos == std::string::npos) { result += '.' + suffix; }
  else { result.replace(pos + 1, std::string::npos, suffix); }
  return result;
}

template <class T>
void matrix_from_row_major_csv(int32_t M, int32_t N, int32_t mb, int32_t nb, T* A, int32_t lda, const std::string& file, int32_t grid_row = 0, int32_t grid_col = 0, int32_t tile_m = 1, int32_t tile_n = 1) {
  std::ifstream csv(replace_suffix(file, "csv")), idx(replace_suffix(file, "cache"));
  std::vector<int64_t> row_bytes;
  if (idx.is_open()) 
  { while (!idx.eof() && row_bytes.size() <= size_t(M)) { int64_t i; idx >> i; row_bytes.push_back(i); } idx.close(); }

  for (int32_t i = grid_row * mb, y = 0; i < M; i = grid_row * mb + tile_m * (y += mb)) {
    int32_t rows = std::min(M - i, mb);
    std::vector<T> mat(int64_t(rows) * int64_t(N));

    if (csv.is_open()) {
      int64_t start_byte = row_bytes[i], n_bytes = row_bytes[i + rows] - start_byte;
      std::vector<char> buf(n_bytes); char* str = &buf[0], *end = &buf[n_bytes];
      csv.seekg(start_byte, std::ios::beg); csv.read(str, n_bytes);
      auto cmp = [](char c){ return c == ' ' || c == ',' || c == '\n' || c == '(' || c == ')'; };
      while(cmp(*str)) ++str;

      for (int32_t y = 0; y < rows; ++y)
        for (int32_t x = 0; x < N; ++x) {
          std::string Ayx(str, std::distance(str, std::find_if(str, end, cmp)));
          int64_t i = int64_t(y) + int64_t(x) * int64_t(rows);
          if constexpr(std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, __half2>) {
            std::string::size_type l; double rl = std::stod(Ayx, &l);
            try { double im = std::stod(Ayx.substr(l)); mat[i] = conv<T>(std::complex<double>(rl, im)); }
              catch (const std::invalid_argument&) { mat[i] = conv<T>(std::complex<double>(rl, 0.)); }
          } else { mat[i] = T(std::stod(Ayx)); }
          str += Ayx.length(); while(cmp(*str)) ++str;
        }
    }

    for (int32_t j = grid_col * nb, x = 0; j < N; j = grid_col * nb + tile_n * (x += nb))
      copy2d(rows, std::min(N - j, nb), &mat[int64_t(j) * int64_t(rows)], rows, &A[int64_t(y) + int64_t(x) * int64_t(lda)], lda);
  }

  if (csv.is_open())
    csv.close();
}

template <class T>
void write_matrix_to_csv(int32_t M, int32_t N, const T* A, int32_t lda, const std::string& file) {
  std::ofstream csv(replace_suffix(file, "csv")), idx(replace_suffix(file, "cache"));
  if (csv.is_open() && idx.is_open()) {
    int64_t bytes = 0; idx << "0\n";
    for (int32_t i = 0; i < M; ++i) {
      std::string str;
      for (int32_t j = 0; j < N; ++j) {
        int64_t k = int64_t(i) + int64_t(j) * int64_t(lda);
        char s[70];
        if constexpr(std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, __half2>)
        { std::complex<double> c = conv<std::complex<double>>(A[k]); char sign = 0. <= double(c.imag()) ? '+':'-'; std::sprintf(s, " (%.18le%c%.18lej)", c.real(), sign, std::abs(c.imag())); }
          else { std::sprintf(s, "%.18le", double(A[k])); }
        str += (j == 0 ? "" : ",") + std::string(s);
      }
      str += '\n'; bytes += str.size();
      csv << str; idx << bytes << '\n';
    }
    csv.close(); idx.close();
    printf("[M=%d,N=%d] Matrix is written to %s\n", M, N, file.c_str());
  }
}

template <class T> struct matrix_generator {
  int64_t gM, gN;
  std::vector<double> bodies;
  const double w;
  matrix_generator(double w, int64_t M, int64_t N) : gM(M), gN(N), bodies(int64_t(3) * (M + N)), w(w) {
    int64_t nbodies = M + N;
    const double phi = 2.39996322972865332223; // golden angle in radians
    const double rx = 20., ry = 2., rz = 1., dn = 2. / double((nbodies - int64_t(1)) ?: int64_t(1));
    for (int64_t i = 0; i < nbodies; ++i) {
      double di = double(i);
      double x = 1. - (di * dn);  // x goes from -r to r.
      double radius = std::sqrt(1. - x * x); // radius at x
      bodies[i * 3] = x * rx;
      bodies[i * 3 + 1] = radius * std::cos(di * phi) * ry;
      bodies[i * 3 + 2] = radius * std::sin(di * phi) * rz;
    }
    /*std::mt19937_64 gen(999);
    std::uniform_real_distribution<double> uniform_dist(0., 1.);
    std::array<double, 3>* b3 = reinterpret_cast<std::array<double, 3>*>(&bodies[0]);
    std::array<double, 3>* b3_end = reinterpret_cast<std::array<double, 3>*>(&bodies[3 * nbodies]);
    std::for_each(b3, b3_end, [&](std::array<double, 3>& body) { body[0] = uniform_dist(gen); body[1] = uniform_dist(gen); body[2] = uniform_dist(gen); });*/

    auto get_bounds = [](const double* bodies, int64_t nbodies, double R[], double C[]) {
      const std::array<double, 3>* b3 = reinterpret_cast<const std::array<double, 3>*>(&bodies[0]);
      const std::array<double, 3>* b3_end = reinterpret_cast<const std::array<double, 3>*>(&bodies[nbodies * 3]);
      double Xmin[3], Xmax[3];
      for (int i = 0; i < 3; ++i) {
        auto minmax = std::minmax_element(b3, b3_end, [=](const std::array<double, 3>& x, const std::array<double, 3>& y) { return x[i] < y[i]; });
        Xmin[i] = (*minmax.first)[i]; Xmax[i] = (*minmax.second)[i];
      }
      std::transform(Xmin, &Xmin[3], Xmax, C, [](double min, double max) { return (min + max) * 0.5; });
      std::transform(Xmin, &Xmin[3], Xmax, R, [](double min, double max) { return (min == max && min == 0.) ? 0. : ((max - min) * 0.5 + 1.e-8); });
    };

    struct Cell { std::array<int64_t, 2> Body; std::array<double, 3> R; std::array<double, 3> C; };
    auto nextPowerOf2 = [](uint64_t x) { if (x <= 1llu) { return 1llu; } --x; x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16; x |= x >> 32; return x + 1llu; };
    int64_t nleaf = nextPowerOf2(uint64_t(nbodies / 512ll)); std::vector<Cell> cells(nleaf + nleaf - 1);
    cells[0].Body[0] = 0; cells[0].Body[1] = nbodies;
    get_bounds(&bodies[0], nbodies, cells[0].R.data(), cells[0].C.data());

    for (int64_t i = 0; i < nleaf - 1; ++i) {
      Cell& ci = cells[i];
      int64_t sdim = std::distance(ci.R.begin(), std::max_element(ci.R.begin(), ci.R.end()));
      int64_t i_begin = ci.Body[0];
      int64_t i_end = ci.Body[1];

      std::array<double, 3>* bodies3 = reinterpret_cast<std::array<double, 3>*>(&bodies[i_begin * 3]);
      std::array<double, 3>* bodies3_end = reinterpret_cast<std::array<double, 3>*>(&bodies[i_end * 3]);
      std::sort(bodies3, bodies3_end, 
        [=](std::array<double, 3>& i, std::array<double, 3>& j) { return i[sdim] < j[sdim]; });

      int64_t len = (i << 1) + 1;
      Cell& c0 = cells[len];
      Cell& c1 = cells[len + 1];
      int64_t loc = i_begin + (i_end - i_begin) / 2;
      c0.Body[0] = i_begin;
      c0.Body[1] = loc;
      c1.Body[0] = loc;
      c1.Body[1] = i_end;

      get_bounds(&bodies[i_begin * 3], loc - i_begin, c0.R.data(), c0.C.data());
      get_bounds(&bodies[loc * 3], i_end - loc, c1.R.data(), c1.C.data());
    }
  }

  inline double eval_real(double d) { return d == 0. ? w : std::cos(w * d) / d; }
  inline std::complex<double> eval_complex(double d) { return std::complex<double>(eval_real(d), d == 0. ? 0. : std::sin(-w * d) / d); }
  //inline double eval_real(double d) { return d == 0. ? w : std::exp((-d * d) / w); }
  //inline std::complex<double> eval_complex(double d) { return std::complex<double>(eval_real(d), 0.); }
  
  void generate_block(int32_t mb, int32_t nb, T* A, int32_t lda, int32_t grid_row = 0, int32_t grid_col = 0, int32_t tile_m = 1, int32_t tile_n = 1) {
    int64_t row_offset = int64_t(grid_row) * int64_t(mb), col_offset = int64_t(grid_col) * int64_t(nb);
    for (int64_t iA = row_offset, y = 0; iA < gM; iA = row_offset + int64_t(tile_m) * (y += int64_t(mb))) {
      int64_t rows = std::min(gM - iA, int64_t(mb));
      for (int64_t jA = col_offset, x = 0; jA < gN; jA = col_offset + int64_t(tile_n) * (x += int64_t(nb))) {
        int64_t cols = std::min(gN - jA, int64_t(nb));
        for (int64_t j = 0; j < cols; ++j) {
          int64_t j_loc = int64_t(3) * (j + jA + gM);
          double pt_j[3]{ bodies[j_loc], bodies[j_loc + int64_t(1)], bodies[j_loc + int64_t(2)]};
          for (int64_t i = 0; i < rows; ++i) {
            int64_t i_loc = int64_t(3) * (i + iA);
            double diff_x = bodies[i_loc] - pt_j[0];
            double diff_y = bodies[i_loc + int64_t(1)] - pt_j[1];
            double diff_z = bodies[i_loc + int64_t(2)] - pt_j[2];
            double d = std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
            int64_t k = (i + y) + (j + x) * int64_t(lda);
            if constexpr(std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, __half2>)
            { A[k] = conv<T>(eval_complex(d)); } else { A[k] = T(eval_real(d)); }
          }
        }
      }
    }
  }
};

void handle_param_overwrite(hyacinHandle_t* handle, char algo, int32_t batchK, int32_t u_corr, int32_t g_corr, int32_t jacobi_sweeps, int32_t oversampling) {
  handle->GramMatrixAlgorithm = algo; handle->BatchK = batchK; handle->QuantizeBitCorrection = u_corr;
  handle->GramBitCorrection = g_corr; handle->JacobiSVDSweeps = jacobi_sweeps; handle->RankOversampling = oversampling; 
}

extern "C" void dgemm_(const char*, const char*, const blas_int*, const blas_int*, const blas_int*, const double*, const double*, const blas_int*, const double*, const blas_int*, const double*, double*, const blas_int*);
extern "C" void zgemm_(const char*, const char*, const blas_int*, const blas_int*, const blas_int*, const void*, const void*, const blas_int*, const void*, const blas_int*, const void*, void*, const blas_int*);
inline void nngemm(blas_int M, blas_int N, blas_int K, const double* A, blas_int lda, const double* B, blas_int ldb, double* C, blas_int ldc)
{ char transa = 'N', transb = 'T'; double one = 1., minus_one = -1.; dgemm_(&transa, &transb, &M, &N, &K, &minus_one, A, &lda, B, &ldb, &one, C, &ldc); }
inline void nngemm(blas_int M, blas_int N, blas_int K, const std::complex<double>* A, blas_int lda, const std::complex<double>* B, blas_int ldb, std::complex<double>* C, blas_int ldc)
{ char transa = 'N', transb = 'C'; std::complex<double> one(1., 0.), minus_one(-1., 0.); zgemm_(&transa, &transb, &M, &N, &K, &minus_one, A, &lda, B, &ldb, &one, C, &ldc); }

template <class T>
double check_answer_svd(int32_t M, int32_t N, int32_t rank, const T* U, int32_t ldu, const T* V, int32_t ldv, const T* B, int32_t ldb) {
  if (rank <= 0 || M <= 0 || N <= 0) { return 0.; }
  constexpr int32_t Complex = std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, __half2>;
  using type = typename std::conditional<Complex, std::complex<double>, double>::type;
  double err = 0.; constexpr int32_t block = 512;
  std::vector<type> matB(block * block), matU(block * block), matV(block * block); 
  for (int32_t i = 0; i < M; i += block) {
    int32_t rows = std::min(M - i, block);
    for (int32_t j = 0; j < N; j += block) {
      int32_t cols = std::min(N - j, block);
      copy2d(rows, cols, &B[int64_t(i) + int64_t(j) * int64_t(ldb)], ldb, &matB[0], rows);
      for (int32_t k = 0; k < rank; k += block) {
        int32_t reduc = std::min(rank - k, block);
        copy2d(rows, reduc, &U[int64_t(i) + int64_t(k) * int64_t(ldu)], ldu, &matU[0], block); copy2d(cols, reduc, &V[int64_t(j) + int64_t(k) * int64_t(ldu)], ldv, &matV[0], block);
        nngemm(rows, cols, reduc, &matU[0], block, &matV[0], block, &matB[0], rows);
      }
      err = std::transform_reduce(matB.begin(), matB.begin() + int64_t(rows) * int64_t(cols), err, std::plus<double>(), [](auto i) { return std::norm(i); });
    }
  }
  return err;
}

template <class T>
double fnorm(int32_t M, int32_t N, const T* A, int32_t lda) {
  if (M <= 0 || N <= 0) { return 0.; }
  constexpr int32_t Complex = std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, __half2>;
  using type = typename std::conditional<Complex, std::complex<double>, double>::type;
  double nrm = 0.; constexpr int32_t block = 512;
  std::vector<type> matA(block * block); 
  for (int32_t i = 0; i < M; i += block) {
    int32_t rows = std::min(M - i, block);
    for (int32_t j = 0; j < N; j += block) {
      int32_t cols = std::min(N - j, block);
      copy2d(rows, cols, &A[int64_t(i) + int64_t(j) * int64_t(lda)], lda, &matA[0], block);
      nrm = std::transform_reduce(matA.begin(), matA.begin() + int64_t(rows) * int64_t(cols), nrm, std::plus<double>(), [](auto i) { return std::norm(i); });
    }
  }
  return nrm;
}

template <class T> inline hyacinPrecision_t __precA();
template <> inline hyacinPrecision_t __precA<double>() { return HYACIN_F64; };
template <> inline hyacinPrecision_t __precA<float>() { return HYACIN_F32; };
template <> inline hyacinPrecision_t __precA<__half>() { return HYACIN_F16; };
template <> inline hyacinPrecision_t __precA<std::complex<double>>() { return HYACIN_F64_COMPLEX; };
template <> inline hyacinPrecision_t __precA<std::complex<float>>() { return HYACIN_F32_COMPLEX; };
template <> inline hyacinPrecision_t __precA<__half2>() { return HYACIN_F16_COMPLEX; };

template <class T, class R>
int32_t svd_fit_transform(hyacinHandle_t handle, double epi, int32_t batchIter, int32_t M, int32_t gM, int32_t N, int32_t K, const T* A, int32_t lda, T* U, int32_t ldu, R* S, T* V, int32_t ldv, int32_t Mv, int32_t Nv = 0, int32_t lcol_offset = 0) {
  hyacinPrecision_t Atype = __precA<T>(), Gtype;
  int32_t* vexp = nullptr, u, cPanels, lPanels, gElemBytes; uint64_t strideC, Bbytes;
  cudaMallocFromPoolAsync((void**)&vexp, uint64_t(N) * sizeof(int32_t), handle.mempool, handle.cudaStream);
  hyacinXGautoType(&handle, epi, gM, N, Atype, &u, &cPanels, &lPanels, &strideC, &Gtype, &gElemBytes);
  hyacinXquantizeScale(&handle, M, N, Atype, A, lda, u, 0, vexp);
  hyacinAllReduceVExp(&handle, uint64_t(N), vexp);

  uint64_t* C = nullptr; cudaMallocFromPoolAsync((void**)&C, uint64_t(cPanels) * uint64_t(lPanels) * strideC * sizeof(uint64_t), handle.mempool, handle.cudaStream);
  if (handle.BatchK <= 0) { hyacinXherk(&handle, M, N, Atype, A, lda, u, vexp, 0, lPanels, C); } else {
    int8_t* Bdata = nullptr; hyacinXherkBatchInit(&handle, epi, N, Atype, &Bbytes);
    cudaMallocFromPoolAsync((void**)&Bdata, Bbytes, handle.mempool, handle.cudaStream);

    int32_t beta = 0, iter = std::min(batchIter, handle.BatchK);
    int64_t strideA = int64_t(lda) * int64_t(sizeof(T)), strideB = int64_t(handle.BatchK) * int64_t(sizeof(T));
    void* Arena = hyacinXherkBatch(&handle, 0, iter, N, Atype, vexp, &beta, lPanels, C, Bdata);
    for (int32_t i = 0; i < M; i += iter) {
      int32_t rows = std::min(M - i, iter);
      cudaMemcpy2DAsync(Arena, strideB, &A[i], strideA, int64_t(rows) * int64_t(sizeof(T)), N, cudaMemcpyDeviceToDevice, handle.cudaStream);
      Arena = hyacinXherkBatch(&handle, rows, iter, N, Atype, vexp, &beta, lPanels, C, Bdata);
    }
    hyacinXherkBatchFlush(&handle, N, Atype, vexp, beta, lPanels, C, Bdata);
    cudaFreeAsync(Bdata, handle.cudaStream);
  }
  hyacinAllReduce1Drow(&handle, cPanels, lPanels, strideC, C);

  void* G = nullptr; cudaMallocFromPoolAsync((void**)&G, uint64_t(N) * uint64_t(N) * uint64_t(gElemBytes), handle.mempool, handle.cudaStream);
  hyacinXdequantize(&handle, N, lPanels, C, vexp, Gtype, G, N);
  cudaFreeAsync(vexp, handle.cudaStream); cudaFreeAsync(C, handle.cudaStream);

  T* X = nullptr; cudaMallocFromPoolAsync((void**)&X, uint64_t(N) * uint64_t(K) * sizeof(T), handle.mempool, handle.cudaStream);
  int32_t rank = hyacinXGevd(&handle, 'A', epi, N, K, Atype, X, N, S, Gtype, G, N);
  cudaFreeAsync(G, handle.cudaStream);
  hyacinXtransform(&handle, M, N, rank, Atype, A, lda, U, ldu, 'F', X, N);
  hyacinXtransform(&handle, Mv, Nv, rank, Atype, V, ldv, V, ldv, 'F', &X[lcol_offset], N);
  cudaFreeAsync(X, handle.cudaStream);

  double eventMs[3]{ 'D', 'R', 'C' }; hyacinSync_TimerSegments(&handle, eventMs, 3);
  kernel_time += eventMs[0] + eventMs[1] + eventMs[2]; rep_time += eventMs[1]; comm_time += eventMs[2];
  return rank;
}

#ifndef NO_NCCL
#ifndef BOOTSTRAP_NO_POSIX
#include <unistd.h>

void bootstrap_posix_fork(int32_t& world_rank, int32_t& local_rank, int32_t& world_size, ncclUniqueId& id) {
  const char* devices = std::getenv("CUDA_VISIBLE_DEVICES");
  if (devices == nullptr) { throw std::runtime_error("CUDA_VISIBLE_DEVICES is unset"); }
  if (devices[0] == '\0') { throw std::runtime_error("No CUDA device visible"); }

  std::string dev_str(devices);
  world_size = 1 + std::count(dev_str.begin(), dev_str.end(), ',');
  int32_t rank = 0;
  ncclGetUniqueId(&id);
  for (int32_t i = 1; i < world_size; ++i) if (rank == 0) {
    int32_t pid = ::fork();
    if (pid == -1) { throw std::runtime_error("POSIX Fork Failure"); }
    rank = pid ? 0 : i;
  }
  world_rank = local_rank = rank;
}

#endif
#ifndef BOOTSTRAP_NO_MPI
#include <mpi.h>

void bootstrap_mpi(int32_t& world_rank, int32_t& local_rank, int32_t& world_size, ncclUniqueId& id) {
  MPI_Init(nullptr, nullptr);
  MPI_Comm shmcomm; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_rank(shmcomm, &local_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_free(&shmcomm);

  if (world_rank == 0) ncclGetUniqueId(&id);
  MPI_Bcast(&id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Finalize();
}

#endif
#endif
