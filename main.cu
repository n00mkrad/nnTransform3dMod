#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <memory>
#include <filesystem>
#include <limits>
#include <cctype>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#endif


#include <cufft.h>
#include <cuda_runtime.h>
#include <onnxruntime_cxx_api.h>

#include "y4m_ntsc.h"


// This is a kernel function running on the GPU   (这是一个运行在 GPU 上的核函数)
__global__ void applyMaskKernel(cufftDoubleComplex* d_out_batch, const float* d_mask, int total_elements) {
    // Get the global unique ID of the current GPU thread   (获取当前 GPU 线程的全局唯一 ID)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Defensive boundary check   (防御性边界检查)
    if (idx < total_elements) {
        float gain = d_mask[idx];
        d_out_batch[idx].x *= gain;
        d_out_batch[idx].y *= gain;
    }
}

// Kernel function running on the GPU: ultra-fast calculation of Magnitude and symmetric features (RefMag)   (运行在 GPU 上的核函数：极速计算 Magnitude 和对称特征 (RefMag))
__global__ void calcMagnitudeKernel(const cufftDoubleComplex* d_out_batch, float* d_trt_input, int num_blocks, int block_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks * block_size) return;

    int b = idx / block_size;  // Current block index   (当前处于第几个 block)
    int i = idx % block_size;  // Current offset within the block   (当前在 block 内的偏移量)

   
    int Nx = 16, Ny = 16; // , Nt = 4;
    int t = i / (Ny * Nx);
    int rem = i % (Ny * Nx);
    int y = rem / Nx;
    int x = rem % Nx;

    // Channel 0: Mag
    double mag = sqrt(d_out_batch[idx].x * d_out_batch[idx].x + d_out_batch[idx].y * d_out_batch[idx].y);

    // Channel 1: RefMag (handles symmetrical flipping)   (Channel 1: RefMag (处理对称翻转))
    int ref_t = (2 - t) % 4; if (ref_t < 0) ref_t += 4;
    int ref_y = (16 - y) % 16;
    int ref_x = (8 - x) % 16; if (ref_x < 0) ref_x += 16;
    int idx_ref = b * block_size + (ref_t * Ny * Nx + ref_y * Nx + ref_x);
    double mag_ref = sqrt(d_out_batch[idx_ref].x * d_out_batch[idx_ref].x + d_out_batch[idx_ref].y * d_out_batch[idx_ref].y);

    // Write into TensorRT input tensor VRAM   (写入 TensorRT 输入张量显存)
    // Data shape: [num_blocks, 2, Nt, Ny, Nx]   (数据形状: [num_blocks, 2, Nt, Ny, Nx])
    int out_idx_0 = b * (2 * block_size) + 0 * block_size + i; // Channel 0 offset   (Channel 0 偏移)
    int out_idx_1 = b * (2 * block_size) + 1 * block_size + i; // Channel 1 offset   (Channel 1 偏移)

   
    d_trt_input[out_idx_0] = (float)(mag / 128.0);
    d_trt_input[out_idx_1] = (float)(mag_ref / 128.0);
}

// Kernel 1: let 5000 threads each calculate the DC (average) of their own block   (核函数 1：让 5000 个线程各自算自己 Block 的 DC (平均值))
__global__ void calcDCKernel(const uint16_t* d_cvbs_f0, const uint16_t* d_cvbs_f1, bool pad_f0, bool pad_f1,
                             const int* d_ledger_y, const int* d_ledger_x, double* d_ledger_dc, 
                             int num_blocks, int activeStartX, int activeEndX) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= num_blocks) return;

    int y = d_ledger_y[b]; int x = d_ledger_x[b];
    double blockDC = 0.0; int pixelCount = 0;

    for (int t = 0; t < 4; ++t) {
        if ((t < 2) ? pad_f0 : pad_f1) continue;
        const uint16_t* cvbs = (t < 2) ? d_cvbs_f0 : d_cvbs_f1;
        bool isOddField = (t % 2 != 0);

        for (int dy = 0; dy < 16; ++dy) {
            int absY = y + dy;
            if (absY < 40 || absY >= 525) continue;
            if ((absY % 2 != 0) == isOddField) {
                for (int dx = 0; dx < 16; ++dx) {
                    int absX = x + dx;
                    if (absX >= activeStartX && absX < activeEndX) {
                        blockDC += cvbs[absY * 910 + absX];
                        pixelCount++;
                    }
                }
            }
        }
    }
    d_ledger_dc[b] = (pixelCount > 0) ? (blockDC / pixelCount) : 0.0;
}

// Kernel 2: instantly complete DC removal, windowing, and packing   (核函数 2：瞬间完成去直流、加窗、打包)
__global__ void packAndWindowKernel(const uint16_t* d_cvbs_f0, const uint16_t* d_cvbs_f1, bool pad_f0, bool pad_f1,
                                    cufftDoubleComplex* d_in_batch, const int* d_ledger_y, const int* d_ledger_x, const double* d_ledger_dc,
                                    const double* d_winT, const double* d_winY, const double* d_winX,
                                    int num_blocks, int block_size, int activeStartX, int activeEndX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks * block_size) return;

    int b = idx / block_size; int i = idx % block_size;
    int t = i / 256; int rem = i % 256; int dy = rem / 16; int dx = rem % 16;

    int absY = d_ledger_y[b] + dy;
    int absX = d_ledger_x[b] + dx;
    bool isOddField = (t % 2 != 0);
    bool isPad = (t < 2) ? pad_f0 : pad_f1;
    
    if (isPad || absY < 40 || absY >= 525 || absX < activeStartX || absX >= activeEndX || (absY % 2 != 0) != isOddField) {
        d_in_batch[idx].x = 0.0;
    } else {
        const uint16_t* cvbs = (t < 2) ? d_cvbs_f0 : d_cvbs_f1;
        double val = (double)cvbs[absY * 910 + absX];
        d_in_batch[idx].x = (val - d_ledger_dc[b]) * d_winT[t] * d_winY[dy] * d_winX[dx];
    }
    d_in_batch[idx].y = 0.0;
}

// Kernel 3: Safe overlap-add (OLA) in VRAM   (核函数 3：显存内安全叠接相加 (OLA))
__global__ void olaKernel(const cufftDoubleComplex* d_in_batch, double* d_accChroma_f0, double* d_accChroma_f1,
                          double* d_weightSum_f0, double* d_weightSum_f1, bool pad_f0, bool pad_f1,
                          const int* d_ledger_y, const int* d_ledger_x, const double* d_winT, const double* d_winY, const double* d_winX,
                          int num_blocks, int block_size, int activeStartX, int activeEndX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks * block_size) return;

    int b = idx / block_size; int i = idx % block_size;
    int t = i / 256; int rem = i % 256; int dy = rem / 16; int dx = rem % 16;

    if ((t < 2) ? pad_f0 : pad_f1) return;

    int absY = d_ledger_y[b] + dy;
    int absX = d_ledger_x[b] + dx;

    if (absY >= 40 && absY < 525 && (absY % 2 != 0) == (t % 2 != 0) && absX >= activeStartX && absX < activeEndX) {
        double val = d_in_batch[idx].x / (double)block_size;
        double w = d_winT[t] * d_winY[dy] * d_winX[dx];
        int frame_idx = absY * 910 + absX;

        // Atomic addition: prevent conflicts from tens of thousands of threads writing to the same pixel simultaneously!   (原子加法：防止几万个线程同时写入同一个像素发生冲突！)
        atomicAdd((t < 2) ? &d_accChroma_f0[frame_idx] : &d_accChroma_f1[frame_idx], val * w);
        atomicAdd((t < 2) ? &d_weightSum_f0[frame_idx] : &d_weightSum_f1[frame_idx], w * w);
    }
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- Basic constants ---   (--- 基础常量 ---)
const int FIELD_WIDTH = 910;
const int FIELD_HEIGHT = 263;
const int FRAME_WIDTH = 910;
const int FRAME_HEIGHT = 526; // 263 * 2

// Vertical Active region   (垂直 Active 区域 )
const int ACTIVE_START_Y = 40;
const int ACTIVE_END_Y = 525;

// 3D window parameters   (3D 窗口参数)
const int Nx = 16, Ny = 16, Nt = 4;
const int STEP_X = 8, STEP_Y = 8; // 50% spatial overlap   (50% 空间重叠)


// --- Data structures ---   (--- 数据结构 ---)
struct FrameBuffer {
    bool isPadding;
    std::vector<uint16_t> cvbs;
    std::vector<double> accChroma;
    std::vector<double> weightSum;

    // GPU VRAM pointers   (GPU 显存指针)
    uint16_t* d_cvbs = nullptr;
    double* d_accChroma = nullptr;
    double* d_weightSum = nullptr;

    FrameBuffer() : isPadding(false) {
        cvbs.resize(FRAME_WIDTH * FRAME_HEIGHT, 0);
        accChroma.resize(FRAME_WIDTH * FRAME_HEIGHT, 0.0);
        weightSum.resize(FRAME_WIDTH * FRAME_HEIGHT, 0.0);
        
        // Pre-allocate space in VRAM directly during initialization   (初始化时直接在显存里开辟好空间)
        cudaMalloc((void**)&d_cvbs, FRAME_WIDTH * FRAME_HEIGHT * sizeof(uint16_t));
        cudaMalloc((void**)&d_accChroma, FRAME_WIDTH * FRAME_HEIGHT * sizeof(double));
        cudaMalloc((void**)&d_weightSum, FRAME_WIDTH * FRAME_HEIGHT * sizeof(double));
    }

    void resetOLA() {
        isPadding = false;
        std::fill(accChroma.begin(), accChroma.end(), 0.0);
        std::fill(weightSum.begin(), weightSum.end(), 0.0);
        // VRAM is also cleared synchronously   (显存也同步清零)
        cudaMemset(d_accChroma, 0, FRAME_WIDTH * FRAME_HEIGHT * sizeof(double));
        cudaMemset(d_weightSum, 0, FRAME_WIDTH * FRAME_HEIGHT * sizeof(double));
    }
    
   
};

// --- Helper function: compute 1D index ---   (--- 辅助函数：计算一维索引 ---)
inline int IDX3(int t, int y, int x, int _Nt, int _Ny, int _Nx) {
    return (t * _Ny * _Nx) + (y * _Nx) + x;
}

// --- File reading and interlacing ---   (--- 文件读取与交织 ---)
bool readInterlacedFrame(std::ifstream& is, FrameBuffer& frame) {
    std::vector<uint16_t> field0(FIELD_WIDTH * FIELD_HEIGHT);
    std::vector<uint16_t> field1(FIELD_WIDTH * FIELD_HEIGHT);

    // Try to read the first field   (尝试读取第一场)
    if (!is.read(reinterpret_cast<char*>(field0.data()), field0.size() * sizeof(uint16_t))) {
        std::cerr << " Failed to read Field 0! Either EOF reached or file is too small.\n";
        return false;
    }
    // Try to read the second field   (尝试读取第二场)
    if (!is.read(reinterpret_cast<char*>(field1.data()), field1.size() * sizeof(uint16_t))) {
        std::cerr << " Failed to read Field 1! EOF reached while expecting second field.\n";
        return false;
    }

    // Interlace into frame (Even/Odd Lines)   (交织为帧 (Even/Odd Lines))
    for (int y = 0; y < FIELD_HEIGHT; ++y) {
        if (y * 2 < FRAME_HEIGHT) {
            std::copy(&field0[y * FIELD_WIDTH], &field0[(y + 1) * FIELD_WIDTH], &frame.cvbs[y * 2 * FRAME_WIDTH]);
        }
        if (y * 2 + 1 < FRAME_HEIGHT) {
            std::copy(&field1[y * FIELD_WIDTH], &field1[(y + 1) * FIELD_WIDTH], &frame.cvbs[(y * 2 + 1) * FRAME_WIDTH]);
        }
    }
    return true;
}

// --- Core 3D filter ---   (--- 核心 3D 滤镜 ---)
// --- Batch ledger structure ---   (--- 批处理账本结构体 ---)
struct BlockMeta {
    int y;
    int x;
    double blockDC;
};

// --- Core 3D filter (Batched high-performance version) ---   (--- 核心 3D 滤镜 (Batched 高性能版) ---)
void processSplit3D(FrameBuffer& f0, FrameBuffer& f1, Ort::Session& session, int activeStartX, int activeEndX) {

    // 1. Generate 3D sine window   (1. 生成 3D 正弦窗)
    std::vector<double> winX(Nx), winY(Ny), winT(Nt);
    for (int i = 0; i < Nx; ++i) winX[i] = sin(M_PI * (i + 0.5) / Nx);
    for (int i = 0; i < Ny; ++i) winY[i] = sin(M_PI * (i + 0.5) / Ny);
    for (int i = 0; i < Nt; ++i) winT[i] = sin(M_PI * (i + 0.5) / Nt);

    int startY_loop = ACTIVE_START_Y - (Ny / 2);
    int startX_loop = activeStartX - (Nx / 2);

    // 2. Precisely calculate the total number of blocks for the current frame (guarantees absolute safety of batch memory allocation)   (2. 精准计算当前帧的 Block 总数 (保证批处理内存分配绝对安全))
    int num_blocks = 0;
    for (int y = startY_loop; y < ACTIVE_END_Y; y += STEP_Y) {
        for (int x = startX_loop; x < activeEndX; x += STEP_X) {
            num_blocks++;
        }
    }

    if (num_blocks == 0) return; // Defensive check   (防御性判断)


   // 3. Static memory pool and cuFFT configuration   (3. 静态内存池与 cuFFT 配置)
    size_t block_size = Nt * Ny * Nx; // 4 * 16 * 16 = 1024

    // Host (CPU) memory: used for data assembly in our previous for loop   (Host (CPU) 内存：用于我们之前的 for 循环组装数据)
    static cufftDoubleComplex* h_in_batch = nullptr;
    static cufftDoubleComplex* h_out_batch = nullptr;
    
    // Device (GPU) VRAM: used for high-speed cuFFT computation   (Device (GPU) 显存：用于 cuFFT 高速计算)
    static cufftDoubleComplex* d_in_batch = nullptr;
    static cufftDoubleComplex* d_out_batch = nullptr;
    
    static cufftHandle p_fwd;
    static cufftHandle p_inv;
    static bool is_plan_created = false;
    static int cached_num_blocks = 0;

    if (cached_num_blocks != num_blocks) {
        if (h_in_batch) {
            free(h_in_batch); free(h_out_batch);
            cudaFree(d_in_batch); cudaFree(d_out_batch);
            if (is_plan_created) {
                cufftDestroy(p_fwd); cufftDestroy(p_inv);
            }
        }

        size_t bytes = sizeof(cufftDoubleComplex) * num_blocks * block_size;
        
        // Allocate CPU memory   (分配 CPU 内存)
        h_in_batch = (cufftDoubleComplex*)malloc(bytes);
        h_out_batch = (cufftDoubleComplex*)malloc(bytes);
        
        // Allocate GPU VRAM   (分配 GPU 显存)
        cudaMalloc((void**)&d_in_batch, bytes);
        cudaMalloc((void**)&d_out_batch, bytes);

        if (!h_in_batch || !d_in_batch) {
            throw std::runtime_error("Fatal Error: Memory allocation failed!");
        }

        // Create cuFFT plan (CUFFT_Z2Z stands for Double-Precision Complex to Complex)   (创建 cuFFT 计划 (CUFFT_Z2Z 代表 Double-Precision Complex to Complex))
        int n[] = { Nt, Ny, Nx };
        cufftPlanMany(&p_fwd, 3, n, NULL, 1, block_size, NULL, 1, block_size, CUFFT_Z2Z, num_blocks);
        cufftPlanMany(&p_inv, 3, n, NULL, 1, block_size, NULL, 1, block_size, CUFFT_Z2Z, num_blocks);
        is_plan_created = true;
        cached_num_blocks = num_blocks;
    }

    std::vector<BlockMeta> batchLedger;
   // ================= Brand new phase one: Data packing and accounting (handed over to GPU) =================   (================= 全新階段一：資料打包與記帳 (交給 GPU) =================)
    // 1. CPU is only responsible for calculating coordinates, does not touch pixels   (1. CPU 只負責算一下座標，不碰像素)
    std::vector<int> h_ledger_y(num_blocks), h_ledger_x(num_blocks);
    int b_idx = 0;
    for (int y = startY_loop; y < ACTIVE_END_Y; y += STEP_Y) {
        for (int x = startX_loop; x < activeEndX; x += STEP_X) {
            h_ledger_y[b_idx] = y; h_ledger_x[b_idx] = x; b_idx++;
        }
    }

    // 2. Statically allocate GPU VRAM for coordinates and window functions (allocated only once)   (2. 靜態分配座標與窗函數的 GPU 顯存 (只分配一次))
    static int *d_ledger_y = nullptr, *d_ledger_x = nullptr;
    static double *d_ledger_dc = nullptr, *d_winX = nullptr, *d_winY = nullptr, *d_winT = nullptr;
    static int cached_ledger_blocks = 0;
    
    if (cached_ledger_blocks != num_blocks) {
        if (d_ledger_y) { cudaFree(d_ledger_y); cudaFree(d_ledger_x); cudaFree(d_ledger_dc); cudaFree(d_winX); cudaFree(d_winY); cudaFree(d_winT); }
        cudaMalloc((void**)&d_ledger_y, num_blocks * sizeof(int));
        cudaMalloc((void**)&d_ledger_x, num_blocks * sizeof(int));
        cudaMalloc((void**)&d_ledger_dc, num_blocks * sizeof(double));
        cudaMalloc((void**)&d_winX, Nx * sizeof(double));
        cudaMalloc((void**)&d_winY, Ny * sizeof(double));
        cudaMalloc((void**)&d_winT, Nt * sizeof(double));
        cached_ledger_blocks = num_blocks;
    }

    // 3. Copy coordinates and window functions to the graphics card (only a few KB per frame, completed instantly)   (3. 把座標和窗函數拷貝給顯卡 (單格只有幾 KB，瞬間完成))
    cudaMemcpy(d_ledger_y, h_ledger_y.data(), num_blocks * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ledger_x, h_ledger_x.data(), num_blocks * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_winX, winX.data(), Nx * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_winY, winY.data(), Ny * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_winT, winT.data(), Nt * sizeof(double), cudaMemcpyHostToDevice);

    // 4. Launch Kernel 1: let GPU calculate DC   (4. 發射 Kernel 1：讓 GPU 算 DC)
    int blocksForDC = (num_blocks + 255) / 256;
    calcDCKernel<<<blocksForDC, 256>>>(f0.d_cvbs, f1.d_cvbs, f0.isPadding, f1.isPadding, 
                                       d_ledger_y, d_ledger_x, d_ledger_dc, num_blocks, activeStartX, activeEndX);

    // 5. Launch Kernel 2: let GPU instantly complete DC removal, windowing, and packing   (5. 發射 Kernel 2：讓 GPU 瞬間完成去直流、加窗、打包)
    int total_elements = num_blocks * block_size;
    int blocksForAll = (total_elements + 255) / 256;
    packAndWindowKernel<<<blocksForAll, 256>>>(f0.d_cvbs, f1.d_cvbs, f0.isPadding, f1.isPadding, d_in_batch,
                                               d_ledger_y, d_ledger_x, d_ledger_dc, d_winT, d_winY, d_winX,
                                               num_blocks, block_size, activeStartX, activeEndX);
    cudaDeviceSynchronize(); // Ensure data packing is complete   (確保資料打包完畢)


    // ================= Phase two: High-concurrency inference and IOBinding =================   (================= 階段二：高併發推理與 IOBinding =================)

    cufftExecZ2Z(p_fwd, d_in_batch, d_out_batch, CUFFT_FORWARD);

    size_t tensor_elements = num_blocks * 2 * block_size;
    static float* d_trt_input = nullptr;
    static float* d_mask = nullptr;
    static int cached_tensor_elements = 0;
    if (cached_tensor_elements != tensor_elements) {
        if (d_trt_input) cudaFree(d_trt_input);
        if (d_mask) cudaFree(d_mask);
        cudaMalloc((void**)&d_trt_input, tensor_elements * sizeof(float));
        cudaMalloc((void**)&d_mask, total_elements * sizeof(float)); 
        cached_tensor_elements = tensor_elements;
    }

    // Launch Mag Kernel, and let TensorRT read VRAM directly   (發射 Mag Kernel，並由 TensorRT 直接讀取顯存)
    calcMagnitudeKernel<<<blocksForAll, 256>>>(d_out_batch, d_trt_input, num_blocks, block_size);
    cudaDeviceSynchronize(); 

    Ort::IoBinding iobinding(session);
    Ort::MemoryInfo mem_info_cuda("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    
    std::vector<int64_t> input_shape = { num_blocks, 2, Nt, Ny, Nx };
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info_cuda, d_trt_input, tensor_elements, input_shape.data(), input_shape.size());
    iobinding.BindInput("input", input_tensor);

    std::vector<int64_t> output_shape = { num_blocks, 1, Nt, Ny, Nx };
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(mem_info_cuda, d_mask, total_elements, output_shape.data(), output_shape.size());
    iobinding.BindOutput("output", output_tensor);

    session.Run(Ort::RunOptions{ nullptr }, iobinding);

    applyMaskKernel<<<blocksForAll, 256>>>(d_out_batch, d_mask, total_elements);
    cudaDeviceSynchronize();


    // ================= Brand new phase three: In-VRAM overlap-add (OLA) =================   (================= 全新階段三：顯存內疊接相加 (OLA) =================)
    cufftExecZ2Z(p_inv, d_out_batch, d_in_batch, CUFFT_INVERSE);

    // Launch Kernel 3: let GPU instantly complete the overlapping accumulation of all pixels, writing to d_accChroma VRAM!   (發射 Kernel 3：讓 GPU 瞬間完成所有像素的重疊累加，寫進 d_accChroma 显存中！)
    olaKernel<<<blocksForAll, 256>>>(d_in_batch, f0.d_accChroma, f1.d_accChroma, f0.d_weightSum, f1.d_weightSum,
                                     f0.isPadding, f1.isPadding, d_ledger_y, d_ledger_x, d_winT, d_winY, d_winX,
                                     num_blocks, block_size, activeStartX, activeEndX);
    cudaDeviceSynchronize();
} 

// --- Finalization and writing ---
enum class OutputMode {
    Tbc,
    RawY,
    RawYc,
    Y4m
};

enum class TbcPipeMode {
    None,
    Y,
    C,
    YcAlt,
    YcStack
};

struct OutputState {
    OutputMode mode = OutputMode::Tbc;
    TbcPipeMode tbcPipeMode = TbcPipeMode::None;
    std::ofstream osLuma;
    std::ofstream osChroma;
    std::ofstream osRaw;
    std::ofstream osY4m;
    std::ostream* rawStream = nullptr;
    std::ostream* tbcPipeStream = nullptr;
    std::unique_ptr<Y4mNtscWriter> y4mWriter;
    bool fullFrame = false;
    int firstLine = 40;
    int lastLine = 525;
};

const char* tbcPipeModeToString(TbcPipeMode mode) {
    if (mode == TbcPipeMode::Y) return "Y";
    if (mode == TbcPipeMode::C) return "C";
    if (mode == TbcPipeMode::YcAlt) return "YC_ALT";
    if (mode == TbcPipeMode::YcStack) return "YC_STACK";
    return "NONE";
}

void packFramePlaneToTbcFields(const std::vector<uint16_t>& framePlane, std::vector<uint16_t>& field0Out, std::vector<uint16_t>& field1Out) {
    field0Out.assign(FIELD_WIDTH * FIELD_HEIGHT, 0);
    field1Out.assign(FIELD_WIDTH * FIELD_HEIGHT, 0);
    for (int y = 0; y < FIELD_HEIGHT; ++y) {
        int frameY0 = y * 2;
        int frameY1 = frameY0 + 1;
        if (frameY0 < FRAME_HEIGHT) std::copy(&framePlane[frameY0 * FRAME_WIDTH], &framePlane[(frameY0 + 1) * FRAME_WIDTH], &field0Out[y * FIELD_WIDTH]);
        if (frameY1 < FRAME_HEIGHT) std::copy(&framePlane[frameY1 * FRAME_WIDTH], &framePlane[(frameY1 + 1) * FRAME_WIDTH], &field1Out[y * FIELD_WIDTH]);
    }
}

void writeTbcFrameChunk(std::ostream& os, const std::vector<uint16_t>& field0, const std::vector<uint16_t>& field1) {
    os.write(reinterpret_cast<const char*>(field0.data()), static_cast<std::streamsize>(field0.size() * sizeof(uint16_t)));
    os.write(reinterpret_cast<const char*>(field1.data()), static_cast<std::streamsize>(field1.size() * sizeof(uint16_t)));
}

void writeCroppedRawPlane(std::ostream& os, const std::vector<uint16_t>& plane, int xStart, int xEnd, int yStart, int yEnd) {
    const int outputWidth = xEnd - xStart;
    for (int y = yStart; y < yEnd; ++y) {
        const uint16_t* rowPtr = &plane[y * FRAME_WIDTH + xStart];
        os.write(reinterpret_cast<const char*>(rowPtr), static_cast<std::streamsize>(outputWidth * sizeof(uint16_t)));
    }
}

void finalizeAndWriteOutput(FrameBuffer& frame, OutputState& outputState, int activeStartX, int activeEndX) {
    if (frame.isPadding) return; // Absolute defense

    std::vector<uint16_t> lumaOut(FRAME_WIDTH * FRAME_HEIGHT);
    std::vector<uint16_t> chromaOut(FRAME_WIDTH * FRAME_HEIGHT);

    for (int y = 0; y < FRAME_HEIGHT; ++y) {
        for (int x = 0; x < FRAME_WIDTH; ++x) {
            int idx = y * FRAME_WIDTH + x;
            // If on the left side of the active picture (including Sync and Color Burst) or right side (Front Porch)
            if (x < activeStartX || x >= activeEndX) {
                // Copy original CVBS signal as is
                lumaOut[idx] = frame.cvbs[idx];
                chromaOut[idx] = frame.cvbs[idx];
            }
            else {
                // Active image area: use the result separated by the neural network 3D
                double chromaVal = 0.0;
                if (frame.weightSum[idx] > 0.00001) {
                    chromaVal = frame.accChroma[idx] / frame.weightSum[idx];
                }

                double lumaVal = frame.cvbs[idx] - chromaVal;
                // Clamp and add Chroma's 32768 neutral gray offset
                lumaOut[idx] = std::min(std::max((int)std::round(lumaVal), 0), 65535);
                chromaOut[idx] = std::min(std::max((int)std::round(chromaVal + 32768.0), 0), 65535);
            }
        }
    }

    if (outputState.mode == OutputMode::Tbc) {
        std::vector<uint16_t> lumaField0, lumaField1, chromaField0, chromaField1;
        packFramePlaneToTbcFields(lumaOut, lumaField0, lumaField1);
        packFramePlaneToTbcFields(chromaOut, chromaField0, chromaField1);

        if (outputState.tbcPipeMode == TbcPipeMode::None) {
            writeTbcFrameChunk(outputState.osLuma, lumaField0, lumaField1);
            writeTbcFrameChunk(outputState.osChroma, chromaField0, chromaField1);
            if (!outputState.osLuma || !outputState.osChroma) throw std::runtime_error("Failed while writing TBC output files.");
            return;
        }

        if (!outputState.tbcPipeStream) {
            throw std::runtime_error("TBC pipe output stream is not initialized.");
        }

        if (outputState.tbcPipeMode == TbcPipeMode::Y) {
            writeTbcFrameChunk(*outputState.tbcPipeStream, lumaField0, lumaField1);
        }
        else if (outputState.tbcPipeMode == TbcPipeMode::C) {
            writeTbcFrameChunk(*outputState.tbcPipeStream, chromaField0, chromaField1);
        }
        else {
            writeTbcFrameChunk(*outputState.tbcPipeStream, lumaField0, lumaField1);
            writeTbcFrameChunk(*outputState.tbcPipeStream, chromaField0, chromaField1);
        }

        if (!(*outputState.tbcPipeStream)) {
            throw std::runtime_error("Failed while writing TBC pipe output stream.");
        }
        return;
    }

    if (outputState.mode == OutputMode::Y4m) {
        if (!outputState.y4mWriter) {
            throw std::runtime_error("Y4M output writer is not initialized.");
        }
        outputState.y4mWriter->writeFrame(lumaOut, chromaOut);
        return;
    }

    if (!outputState.rawStream) {
        throw std::runtime_error("Raw output stream is not initialized.");
    }

    const int outputXStart = outputState.fullFrame ? 0 : activeStartX;
    const int outputXEnd = outputState.fullFrame ? FRAME_WIDTH : activeEndX;
    const int outputYStart = outputState.fullFrame ? 0 : outputState.firstLine;
    const int outputYEnd = outputState.fullFrame ? FRAME_HEIGHT : outputState.lastLine;

    if (outputXStart < 0 || outputXEnd > FRAME_WIDTH || outputXStart >= outputXEnd || outputYStart < 0 || outputYEnd > FRAME_HEIGHT || outputYStart >= outputYEnd) {
        throw std::runtime_error("Invalid raw output crop range.");
    }

    if (outputState.fullFrame) {
        outputState.rawStream->write(reinterpret_cast<const char*>(lumaOut.data()), static_cast<std::streamsize>(lumaOut.size() * sizeof(uint16_t)));
        if (outputState.mode == OutputMode::RawYc) {
            outputState.rawStream->write(reinterpret_cast<const char*>(chromaOut.data()), static_cast<std::streamsize>(chromaOut.size() * sizeof(uint16_t)));
        }
    } else {
        writeCroppedRawPlane(*outputState.rawStream, lumaOut, outputXStart, outputXEnd, outputYStart, outputYEnd);
        if (outputState.mode == OutputMode::RawYc) {
            writeCroppedRawPlane(*outputState.rawStream, chromaOut, outputXStart, outputXEnd, outputYStart, outputYEnd);
        }
    }
    if (!(*outputState.rawStream)) {
        throw std::runtime_error("Failed while writing raw output stream.");
    }
}

bool tryGetExecutableDirectory(const char* argv0, std::filesystem::path& executableDir, std::string& errorMessage) {
#ifdef _WIN32
    std::vector<wchar_t> exePathBuffer(32768, L'\0');
    DWORD copied = GetModuleFileNameW(nullptr, exePathBuffer.data(), static_cast<DWORD>(exePathBuffer.size()));
    if (copied == 0) {
        errorMessage = "[Error] Failed to resolve executable path via GetModuleFileNameW. Win32 error: " + std::to_string(GetLastError()) + ".";
        return false;
    }
    if (copied >= exePathBuffer.size()) {
        errorMessage = "[Error] Executable path is too long to resolve safely.";
        return false;
    }
    executableDir = std::filesystem::path(std::wstring(exePathBuffer.data(), copied)).parent_path();
    return true;
#else
    if (argv0 == nullptr || argv0[0] == '\0') {
        errorMessage = "[Error] argv[0] is empty, cannot resolve executable directory.";
        return false;
    }
    std::error_code ec;
    std::filesystem::path executablePath = std::filesystem::absolute(std::filesystem::path(argv0), ec);
    if (ec) {
        errorMessage = "[Error] Failed to resolve executable directory from argv[0]: " + ec.message() + ".";
        return false;
    }
    executableDir = executablePath.parent_path();
    return true;
#endif
}

bool resolveModelPath(const char* argv0, const std::string& modelPathArg, bool modelSpecified, std::filesystem::path& resolvedModelPath, std::string& errorMessage) {
    std::error_code ec;
    std::filesystem::path currentDir = std::filesystem::current_path(ec);
    if (ec) {
        errorMessage = "[Error] Failed to resolve current working directory: " + ec.message() + ".";
        return false;
    }
    currentDir = currentDir.lexically_normal();

    if (modelSpecified) {
        std::filesystem::path explicitModelPath = std::filesystem::absolute(std::filesystem::path(modelPathArg), ec);
        if (ec) {
            errorMessage = "[Error] Failed to resolve --model path '" + modelPathArg + "': " + ec.message() + ".";
            return false;
        }
        explicitModelPath = explicitModelPath.lexically_normal();

        bool explicitExists = std::filesystem::exists(explicitModelPath, ec);
        if (ec) {
            errorMessage = "[Error] Failed to check --model path '" + explicitModelPath.string() + "': " + ec.message() + ".";
            return false;
        }
        if (!explicitExists) {
            errorMessage = "[Error] Model file specified by --model was not found: " + explicitModelPath.string();
            return false;
        }

        bool explicitRegularFile = std::filesystem::is_regular_file(explicitModelPath, ec);
        if (ec) {
            errorMessage = "[Error] Failed to inspect --model path '" + explicitModelPath.string() + "': " + ec.message() + ".";
            return false;
        }
        if (!explicitRegularFile) {
            errorMessage = "[Error] Model path specified by --model is not a regular file: " + explicitModelPath.string();
            return false;
        }

        resolvedModelPath = explicitModelPath;
        return true;
    }

    std::filesystem::path executableDir;
    if (!tryGetExecutableDirectory(argv0, executableDir, errorMessage)) return false;
    executableDir = executableDir.lexically_normal();

    std::filesystem::path exeCandidate = (executableDir / "chroma_net.onnx").lexically_normal();
    std::filesystem::path cwdCandidate = (currentDir / "chroma_net.onnx").lexically_normal();

    bool exeCandidateRegularFile = std::filesystem::is_regular_file(exeCandidate, ec);
    if (ec) {
        errorMessage = "[Error] Failed to inspect default model path '" + exeCandidate.string() + "': " + ec.message() + ".";
        return false;
    }
    if (exeCandidateRegularFile) {
        resolvedModelPath = exeCandidate;
        return true;
    }

    bool cwdCandidateRegularFile = std::filesystem::is_regular_file(cwdCandidate, ec);
    if (ec) {
        errorMessage = "[Error] Failed to inspect default model path '" + cwdCandidate.string() + "': " + ec.message() + ".";
        return false;
    }
    if (cwdCandidateRegularFile) {
        resolvedModelPath = cwdCandidate;
        return true;
    }

    errorMessage = "[Error] ONNX model file was not found. Searched: '" + exeCandidate.string() + "' and '" + cwdCandidate.string() + "'.";
    return false;
}

void printUsage(const char* exeName) {
    std::cerr << "Usage: " << exeName << " [--input <path>] [--model <path>] [--gpu <num>] [--trt_mpi <num>] [--trt_mss <num>] [--start-frame <num>] [--end-frame <num>] [--av-start <num>] [--av-end <num>] [--width <num>] [--out-mode tbc|raw_y|raw_yc|y4m] [--tbc-pipe-mode <y|c|yc_alt|yc_stack>] [--json <path>] [--full-frame] [--force-limited] [--levels <black:white|ntsc|ntscj>] [--chroma-gain <number>] [--first-line <num>] [--last-line <num>] [--lines <num>] [-q] [--out <path|->] [input.tbc]\n";
    std::cerr << "Defaults: --out-mode tbc, --gpu 0, --trt_mpi 1000, --trt_mss 1, --start-frame 0, --end-frame 0, --av-start 132, --av-end 896, --lines 480\n";
}

// Parse CLI integer values only when the full token is consumed.
bool parseIntValue(const std::string& value, int& result) {
    try {
        std::size_t parsedLength = 0;
        result = std::stoi(value, &parsedLength);
        return parsedLength == value.size();
    }
    catch (const std::exception&) {
        return false;
    }
}

// Parse CLI floating-point values only when the full token is consumed.
bool parseDoubleValue(const std::string& value, double& result) {
    try {
        std::size_t parsedLength = 0;
        result = std::stod(value, &parsedLength);
        return parsedLength == value.size() && std::isfinite(result);
    }
    catch (const std::exception&) {
        return false;
    }
}

// Parse Y4M level presets or explicit black:white scaling levels.
bool parseLevelsValue(const std::string& value, int& black16bIre, int& white16bIre, std::string& errorMessage) {
    std::string normalized;
    normalized.reserve(value.size());
    for (char c : value) normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    if (normalized == "ntsc") {
        black16bIre = 18048;
        white16bIre = 51200;
        return true;
    }
    if (normalized == "ntscj") {
        black16bIre = 15360;
        white16bIre = 51200;
        return true;
    }
    const std::size_t separator = value.find(':');
    if (separator == std::string::npos || separator == 0 || separator == value.size() - 1 || value.find(':', separator + 1) != std::string::npos) {
        errorMessage = "Expected --levels value to be ntsc, ntscj, or <black>:<white>.";
        return false;
    }
    const std::string blackValue = value.substr(0, separator);
    const std::string whiteValue = value.substr(separator + 1);
    if (!parseIntValue(blackValue, black16bIre) || !parseIntValue(whiteValue, white16bIre)) {
        errorMessage = "Custom --levels values must be integers.";
        return false;
    }
    if (white16bIre <= black16bIre) {
        errorMessage = "--levels white value must be greater than black value.";
        return false;
    }
    return true;
}

// --- Main program ---
int main(int argc, char** argv) {
    int activeVideoStart = 132;
    int activeVideoEnd = 896;
    bool avStartSpecified = false;
    bool avEndSpecified = false;
    std::string inFile = "input.tbc"; // Default processing if no parameters are passed
    OutputMode outputMode = OutputMode::Tbc;
    TbcPipeMode tbcPipeMode = TbcPipeMode::None;
    bool fullFrame = false;
    bool forceLimited = false;
    bool levelsSpecified = false;
    int black16bIreOverride = 0;
    int white16bIreOverride = 0;
    double chromaGain = 1.0;
    bool chromaGainSpecified = false;
    int firstLine = 40;
    int lastLine = 525;
    int lines = 480;
    int width = 0;
    bool firstLineSpecified = false;
    bool lastLineSpecified = false;
    bool linesSpecified = false;
    bool widthSpecified = false;
    std::string jsonPath;
    bool jsonSpecified = false;
    std::string outPath;
    bool outPathSpecified = false;
    bool inputSpecified = false;
    std::string modelPathArg;
    bool modelSpecified = false;
    int gpuId = 0;
    int trtMaxPartitionIterations = 1000;
    int trtMinSubgraphSize = 1;
    int startFrame = 0;
    int endFrame = 0;
    bool endFrameSpecified = false;
    bool quietProgressLog = false;
    std::filesystem::path executableDir;
    std::string executableDirError;
    if (!tryGetExecutableDirectory(argv[0], executableDir, executableDirError)) {
        std::cerr << executableDirError << "\n";
        return -1;
    }
    executableDir = executableDir.lexically_normal();

    std::vector<std::string> positionalArgs;
    int argIndex = 1;
    while (argIndex < argc) {
        std::string arg = argv[argIndex];
        auto nextValueAvailable = [&]() { return (argIndex + 1 < argc) && (std::string(argv[argIndex + 1]).rfind("--", 0) != 0); };

        if (arg == "--input") {
            if (!nextValueAvailable()) {
                std::cerr << "[Error] Missing value after --input.\n";
                printUsage(argv[0]);
                return -1;
            }
            inFile = argv[++argIndex];
            inputSpecified = true;
        }
        else if (arg == "--model") {
            if (!nextValueAvailable()) {
                std::cerr << "[Error] Missing value after --model.\n";
                printUsage(argv[0]);
                return -1;
            }
            modelPathArg = argv[++argIndex];
            modelSpecified = true;
        }
        else if (arg == "--gpu") {
            if (!nextValueAvailable()) {
                std::cerr << "[Error] Missing value after --gpu.\n";
                printUsage(argv[0]);
                return -1;
            }
            try {
                gpuId = std::stoi(argv[++argIndex]);
                if (gpuId < 0) {
                    std::cerr << "[Error] --gpu must be a non-negative integer.\n";
                    printUsage(argv[0]);
                    return -1;
                }
            }
            catch (const std::exception& e) {
                std::cerr << "[Error] Invalid --gpu value: " << e.what() << "\n";
                printUsage(argv[0]);
                return -1;
            }
        }
        else if (arg == "--trt_mpi") {
            if (!nextValueAvailable()) {
                std::cerr << "[Error] Missing value after --trt_mpi.\n";
                printUsage(argv[0]);
                return -1;
            }
            try {
                trtMaxPartitionIterations = std::stoi(argv[++argIndex]);
                if (trtMaxPartitionIterations <= 0) {
                    std::cerr << "[Error] --trt_mpi must be a positive integer.\n";
                    printUsage(argv[0]);
                    return -1;
                }
            }
            catch (const std::exception& e) {
                std::cerr << "[Error] Invalid --trt_mpi value: " << e.what() << "\n";
                printUsage(argv[0]);
                return -1;
            }
        }
        else if (arg == "--trt_mss") {
            if (!nextValueAvailable()) {
                std::cerr << "[Error] Missing value after --trt_mss.\n";
                printUsage(argv[0]);
                return -1;
            }
            try {
                trtMinSubgraphSize = std::stoi(argv[++argIndex]);
                if (trtMinSubgraphSize <= 0) {
                    std::cerr << "[Error] --trt_mss must be a positive integer.\n";
                    printUsage(argv[0]);
                    return -1;
                }
            }
            catch (const std::exception& e) {
                std::cerr << "[Error] Invalid --trt_mss value: " << e.what() << "\n";
                printUsage(argv[0]);
                return -1;
            }
        }
        else if (arg == "--start-frame") {
            if (!nextValueAvailable()) {
                std::cerr << "[Error] Missing value after --start-frame.\n";
                printUsage(argv[0]);
                return -1;
            }
            try {
                startFrame = std::stoi(argv[++argIndex]);
                if (startFrame < 0) {
                    std::cerr << "[Error] --start-frame must be a non-negative integer.\n";
                    printUsage(argv[0]);
                    return -1;
                }
            }
            catch (const std::exception& e) {
                std::cerr << "[Error] Invalid --start-frame value: " << e.what() << "\n";
                printUsage(argv[0]);
                return -1;
            }
        }
        else if (arg == "--end-frame") {
            if (!nextValueAvailable()) {
                std::cerr << "[Error] Missing value after --end-frame.\n";
                printUsage(argv[0]);
                return -1;
            }
            try {
                endFrame = std::stoi(argv[++argIndex]);
                endFrameSpecified = true;
                if (endFrame < 0) {
                    std::cerr << "[Error] --end-frame must be a non-negative integer.\n";
                    printUsage(argv[0]);
                    return -1;
                }
            }
            catch (const std::exception& e) {
                std::cerr << "[Error] Invalid --end-frame value: " << e.what() << "\n";
                printUsage(argv[0]);
                return -1;
            }
        }
        else if (arg == "--av-start") {
            if (!nextValueAvailable()) {
                std::cerr << "[Error] Missing value after --av-start.\n";
                printUsage(argv[0]);
                return -1;
            }
            try {
                activeVideoStart = std::stoi(argv[++argIndex]);
                avStartSpecified = true;
            }
            catch (const std::exception& e) {
                std::cerr << "[Error] Invalid --av-start value: " << e.what() << "\n";
                printUsage(argv[0]);
                return -1;
            }
        }
        else if (arg == "--av-end") {
            if (!nextValueAvailable()) {
                std::cerr << "[Error] Missing value after --av-end.\n";
                printUsage(argv[0]);
                return -1;
            }
            try {
                activeVideoEnd = std::stoi(argv[++argIndex]);
                avEndSpecified = true;
            }
            catch (const std::exception& e) {
                std::cerr << "[Error] Invalid --av-end value: " << e.what() << "\n";
                printUsage(argv[0]);
                return -1;
            }
        }
        else if (arg == "--width") {
            if (!nextValueAvailable()) {
                std::cerr << "[Error] Missing value after --width.\n";
                printUsage(argv[0]);
                return -1;
            }
            try {
                width = std::stoi(argv[++argIndex]);
                widthSpecified = true;
                if (width <= 0) {
                    std::cerr << "[Error] --width must be a positive integer.\n";
                    printUsage(argv[0]);
                    return -1;
                }
            }
            catch (const std::exception& e) {
                std::cerr << "[Error] Invalid --width value: " << e.what() << "\n";
                printUsage(argv[0]);
                return -1;
            }
        }
        else if (arg == "--out-mode") {
            if (!nextValueAvailable()) {
                std::cerr << "[Error] Missing value after --out-mode.\n";
                printUsage(argv[0]);
                return -1;
            }
            std::string value = argv[++argIndex];
            if (value == "tbc") outputMode = OutputMode::Tbc;
            else if (value == "raw_y") outputMode = OutputMode::RawY;
            else if (value == "raw_yc") outputMode = OutputMode::RawYc;
            else if (value == "y4m") outputMode = OutputMode::Y4m;
            else {
                std::cerr << "[Error] Unknown --out-mode value: " << value << "\n";
                printUsage(argv[0]);
                return -1;
            }
        }
        else if (arg == "--tbc-pipe-mode") {
            if (!nextValueAvailable()) {
                std::cerr << "[Error] Missing value after --tbc-pipe-mode.\n";
                printUsage(argv[0]);
                return -1;
            }
            std::string value = argv[++argIndex];
            if (value == "y") tbcPipeMode = TbcPipeMode::Y;
            else if (value == "c") tbcPipeMode = TbcPipeMode::C;
            else if (value == "yc_alt") tbcPipeMode = TbcPipeMode::YcAlt;
            else if (value == "yc_stack") tbcPipeMode = TbcPipeMode::YcStack;
            else {
                std::cerr << "[Error] Unknown --tbc-pipe-mode value: " << value << "\n";
                printUsage(argv[0]);
                return -1;
            }
        }
        else if (arg == "--out") {
            if (!nextValueAvailable()) {
                std::cerr << "[Error] Missing value after --out.\n";
                printUsage(argv[0]);
                return -1;
            }
            outPathSpecified = true;
            outPath = argv[++argIndex];
        }
        else if (arg == "--json") {
            if (!nextValueAvailable()) {
                std::cerr << "[Error] Missing value after --json.\n";
                printUsage(argv[0]);
                return -1;
            }
            jsonPath = argv[++argIndex];
            jsonSpecified = true;
        }
        else if (arg == "--full-frame") {
            fullFrame = true;
        }
        else if (arg == "--force-limited") {
            forceLimited = true;
        }
        else if (arg == "--levels") {
            if (!nextValueAvailable()) {
                std::cerr << "[Error] Missing value after --levels.\n";
                printUsage(argv[0]);
                return -1;
            }
            std::string value = argv[++argIndex];
            std::string levelsError;
            if (!parseLevelsValue(value, black16bIreOverride, white16bIreOverride, levelsError)) {
                std::cerr << "[Error] " << levelsError << "\n";
                printUsage(argv[0]);
                return -1;
            }
            levelsSpecified = true;
        }
        else if (arg == "--chroma-gain") {
            if (!nextValueAvailable()) {
                std::cerr << "[Error] Missing value after --chroma-gain.\n";
                printUsage(argv[0]);
                return -1;
            }
            std::string value = argv[++argIndex];
            if (!parseDoubleValue(value, chromaGain)) {
                std::cerr << "[Error] Invalid --chroma-gain value.\n";
                printUsage(argv[0]);
                return -1;
            }
            if (chromaGain < 0.0) {
                std::cerr << "[Error] Chroma gain must not be less than 0.\n";
                printUsage(argv[0]);
                return -1;
            }
            chromaGainSpecified = true;
        }
        else if (arg == "--first-line") {
            if (!nextValueAvailable()) {
                std::cerr << "[Error] Missing value after --first-line.\n";
                printUsage(argv[0]);
                return -1;
            }
            try {
                firstLine = std::stoi(argv[++argIndex]);
                firstLineSpecified = true;
            }
            catch (const std::exception& e) {
                std::cerr << "[Error] Invalid --first-line value: " << e.what() << "\n";
                printUsage(argv[0]);
                return -1;
            }
        }
        else if (arg == "--last-line") {
            if (!nextValueAvailable()) {
                std::cerr << "[Error] Missing value after --last-line.\n";
                printUsage(argv[0]);
                return -1;
            }
            try {
                lastLine = std::stoi(argv[++argIndex]);
                lastLineSpecified = true;
            }
            catch (const std::exception& e) {
                std::cerr << "[Error] Invalid --last-line value: " << e.what() << "\n";
                printUsage(argv[0]);
                return -1;
            }
        }
        else if (arg == "--lines") {
            if (!nextValueAvailable()) {
                std::cerr << "[Error] Missing value after --lines.\n";
                printUsage(argv[0]);
                return -1;
            }
            try {
                lines = std::stoi(argv[++argIndex]);
                linesSpecified = true;
                if (lines <= 0) {
                    std::cerr << "[Error] --lines must be a positive integer.\n";
                    printUsage(argv[0]);
                    return -1;
                }
            }
            catch (const std::exception& e) {
                std::cerr << "[Error] Invalid --lines value: " << e.what() << "\n";
                printUsage(argv[0]);
                return -1;
            }
        }
        else if (arg == "-q") {
            quietProgressLog = true;
        }
        else if (arg.rfind("--", 0) == 0) {
            std::cerr << "[Error] Unknown option: " << arg << "\n";
            printUsage(argv[0]);
            return -1;
        }
        else {
            positionalArgs.push_back(arg);
        }
        argIndex++;
    }

    if (!inputSpecified && !positionalArgs.empty()) {
        inFile = positionalArgs[0];
    }

    size_t allowedPositionalCount = inputSpecified ? 0 : 1;
    if (positionalArgs.size() > allowedPositionalCount) {
        std::cerr << "[Error] Unexpected positional arguments. Only one positional input is allowed when --input is omitted.\n";
        printUsage(argv[0]);
        return -1;
    }

    LdJsonMetadata metadata;
    std::string metadataPathResolved;
    std::string metadataError;
    bool metadataLoaded = false;

    if (outputMode == OutputMode::Tbc && (fullFrame || firstLineSpecified || lastLineSpecified || linesSpecified)) {
        std::cerr << "[Error] --full-frame, --first-line, --last-line, and --lines are not valid with --out-mode tbc.\n";
        printUsage(argv[0]);
        return -1;
    }

    if (forceLimited && outputMode != OutputMode::Y4m) {
        std::cerr << "[Error] --force-limited is only valid with --out-mode y4m.\n";
        printUsage(argv[0]);
        return -1;
    }
    if (levelsSpecified && outputMode != OutputMode::Y4m) {
        std::cerr << "[Error] --levels is only valid with --out-mode y4m.\n";
        printUsage(argv[0]);
        return -1;
    }
    if (chromaGainSpecified && outputMode != OutputMode::Y4m) {
        std::cerr << "[Error] --chroma-gain is only valid with --out-mode y4m.\n";
        printUsage(argv[0]);
        return -1;
    }

    if (tbcPipeMode != TbcPipeMode::None) {
        if (outputMode != OutputMode::Tbc) {
            std::cerr << "[Error] --tbc-pipe-mode can only be used with --out-mode tbc.\n";
            printUsage(argv[0]);
            return -1;
        }
        if (!outPathSpecified || outPath != "-") {
            std::cerr << "[Error] --tbc-pipe-mode requires --out -.\n";
            printUsage(argv[0]);
            return -1;
        }
    }

    if (outputMode == OutputMode::Tbc && tbcPipeMode == TbcPipeMode::None && outPathSpecified) {
        if (outPath == "-") std::cerr << "[Error] --out - requires --tbc-pipe-mode when --out-mode is tbc.\n";
        else std::cerr << "[Error] --out is only valid for raw modes, or for TBC piping with --tbc-pipe-mode and --out -.\n";
        printUsage(argv[0]);
        return -1;
    }

    metadataPathResolved = jsonSpecified ? jsonPath : (inFile + ".json");
    if (loadLdJsonMetadata(metadataPathResolved, metadata, metadataError)) {
        metadataLoaded = true;
        if (!avStartSpecified) activeVideoStart = metadata.videoParameters.activeVideoStart;
        if (!avEndSpecified) activeVideoEnd = metadata.videoParameters.activeVideoEnd;
        if (!chromaGainSpecified && metadata.videoParameters.chromaGain >= 0.0) chromaGain = metadata.videoParameters.chromaGain;
    } else if (outputMode == OutputMode::Y4m) {
        std::cerr << "[Error] " << metadataError << "\n";
        return -1;
    }

    if (!fullFrame && !lastLineSpecified) {
        long long derivedLastLine = static_cast<long long>(firstLine) + static_cast<long long>(lines);
        if (derivedLastLine > 2147483647LL) {
            std::cerr << "[Error] Derived --last-line exceeds supported integer range.\n";
            return -1;
        }
        lastLine = static_cast<int>(derivedLastLine);
    }
    if (!avEndSpecified && widthSpecified) {
        long long derivedActiveVideoEnd = static_cast<long long>(activeVideoStart) + static_cast<long long>(width);
        if (derivedActiveVideoEnd > 2147483647LL) {
            std::cerr << "[Error] Derived --av-end exceeds supported integer range.\n";
            return -1;
        }
        activeVideoEnd = static_cast<int>(derivedActiveVideoEnd);
    }

    if (activeVideoStart >= activeVideoEnd) {
        std::cerr << "[Error] Invalid active horizontal range after metadata/CLI merge.\n";
        return -1;
    }
    if (activeVideoStart < 0 || activeVideoEnd > FRAME_WIDTH) {
        std::cerr << "[Error] Active horizontal range is out of bounds. Valid range is [0, " << FRAME_WIDTH << "].\n";
        return -1;
    }
    if (outputMode != OutputMode::Tbc && !fullFrame && (firstLine < 0 || lastLine > FRAME_HEIGHT || firstLine >= lastLine)) {
        std::cerr << "[Error] Resolved vertical range must satisfy 0 <= first-line < last-line <= " << FRAME_HEIGHT << ".\n";
        printUsage(argv[0]);
        return -1;
    }

    std::filesystem::path resolvedModelPath;
    std::string resolvedModelPathError;
    if (!resolveModelPath(argv[0], modelPathArg, modelSpecified, resolvedModelPath, resolvedModelPathError)) {
        std::cerr << resolvedModelPathError << "\n";
        return -1;
    }

    std::string baseName = inFile.substr(0, inFile.find_last_of('.'));
    std::string lumaFile = baseName + "_Y.tbc";
    std::string chromaFile = baseName + "_C.tbc";
    if (!outPathSpecified) {
        if (outputMode == OutputMode::RawY) outPath = baseName + "_Y.raw";
        else if (outputMode == OutputMode::RawYc) outPath = baseName + "_YC.raw";
        else if (outputMode == OutputMode::Y4m) outPath = baseName + ".y4m";
    }

    bool writeRawToStdout = ((outputMode == OutputMode::RawY || outputMode == OutputMode::RawYc) && outPath == "-");
    bool writeTbcPipeToStdout = (outputMode == OutputMode::Tbc && tbcPipeMode != TbcPipeMode::None);
    bool writeY4mToStdout = (outputMode == OutputMode::Y4m && outPath == "-");
    std::ostream& log = std::cerr;

#ifdef _WIN32
    if ((writeRawToStdout || writeTbcPipeToStdout || writeY4mToStdout) && _setmode(_fileno(stdout), _O_BINARY) == -1) {
        std::cerr << "[Error] Failed to switch stdout to binary mode for piped output.\n";
        return -1;
    }
#endif

    log << "Target Input File: " << inFile << "\n";
    log << "Metadata Input: " << metadataPathResolved << (metadataLoaded ? " (loaded)\n" : " (not loaded, using defaults)\n");
    if (!metadataLoaded && !metadataError.empty()) {
        log << "Metadata Load Note: " << metadataError << "\n";
    }
    if (outputMode == OutputMode::Tbc) {
        if (writeTbcPipeToStdout) {
            log << "Output Mode: TBC_PIPE_" << tbcPipeModeToString(tbcPipeMode) << "\n";
            log << "TBC Pipe Output: stdout (-)\n";
            if (tbcPipeMode == TbcPipeMode::YcAlt) log << "Interpretation: 910x526 stream with alternating Y then C frame chunks.\n";
            if (tbcPipeMode == TbcPipeMode::YcStack) log << "Interpretation: 910x1052 stacked stream (Y top, C bottom), out-of-spec TBC geometry.\n";
        } else {
            log << "Output Mode: TBC (dual files)\n";
            log << "Luma Output: " << lumaFile << "\n";
            log << "Chroma Output: " << chromaFile << "\n";
        }
    } else if (outputMode == OutputMode::Y4m) {
        if (!metadataLoaded) {
            std::cerr << "[Error] Y4M metadata is not initialized.\n";
            return -1;
        }
        log << "Output Mode: Y4M\n";
        log << "Y4M Output: " << (writeY4mToStdout ? "stdout (-)" : outPath) << "\n";
        log << "Frame Area: " << (fullFrame ? "full" : "active") << "\n";
        log << "Y4M Range Clamp: " << (forceLimited ? "legal" : "headroom/footroom") << "\n";
        log << "Y4M Levels: black " << (levelsSpecified ? black16bIreOverride : metadata.videoParameters.black16bIre) << ", white " << (levelsSpecified ? white16bIreOverride : metadata.videoParameters.white16bIre) << (levelsSpecified ? " (CLI override)\n" : " (metadata)\n");
        log << "Y4M Chroma Gain: " << chromaGain << (chromaGainSpecified ? " (CLI override)\n" : ((metadata.videoParameters.chromaGain >= 0.0) ? " (metadata)\n" : " (default)\n"));
        if (!fullFrame) {
            log << "First Line: " << firstLine << "\n";
            log << "Last Line: " << lastLine << "\n";
            log << "Lines: " << (lastLine - firstLine) << "\n";
            log << "Y4M Horizontal Active Range: [" << activeVideoStart << ", " << activeVideoEnd << ")\n";
            log << "Width: " << (activeVideoEnd - activeVideoStart) << "\n";
        }
    } else {
        log << "Output Mode: " << ((outputMode == OutputMode::RawY) ? "RAW_Y" : "RAW_YC") << "\n";
        log << "Raw Output: " << (writeRawToStdout ? "stdout (-)" : outPath) << "\n";
        log << "Frame Area: " << (fullFrame ? "full" : "active") << "\n";
        if (!fullFrame) {
            log << "First Line: " << firstLine << "\n";
            log << "Last Line: " << lastLine << "\n";
            log << "Lines: " << (lastLine - firstLine) << "\n";
            log << "Raw Horizontal Active Range: [" << activeVideoStart << ", " << activeVideoEnd << ")\n";
            log << "Width: " << (activeVideoEnd - activeVideoStart) << "\n";
        }
    }

    std::ifstream is(inFile, std::ios::binary);
    if (!is.is_open()) {
        std::cerr << "[Error] Cannot open input file! Check if the file exists and is not locked.\n";
        return -1;
    }

    // Check file size
    is.seekg(0, std::ios::end);
    long long fileSize = is.tellg();
    log << "File opened successfully. Size: " << fileSize << " bytes.\n";

    // Calculate the number of bytes needed for one frame: 910 * 263 * 2 fields * 2 bytes (uint16)
    const long long frameBytes = FIELD_WIDTH * FIELD_HEIGHT * 2 * 2;
    log << "Bytes required for ONE frame: " << frameBytes << " bytes.\n";
    if (fileSize < 0) {
        std::cerr << "[Error] Failed to determine input file size.\n";
        return -1;
    }
    if (frameBytes <= 0) {
        std::cerr << "[Error] Invalid frame byte size configuration.\n";
        return -1;
    }

    const long long totalFramesAvailable = fileSize / frameBytes;
    if (totalFramesAvailable <= 0) {
        std::cerr << "[Error] File is too small to contain even ONE frame!\n";
        return -1;
    }
    if (startFrame >= totalFramesAvailable) {
        std::cerr << "[Error] --start-frame " << startFrame << " is out of range. Available frame indices: 0.." << (totalFramesAvailable - 1) << ".\n";
        return -1;
    }
    const bool limitEndFrame = endFrameSpecified && endFrame > 0;
    if (limitEndFrame && endFrame >= totalFramesAvailable) {
        std::cerr << "[Error] --end-frame " << endFrame << " is out of range. Available frame indices: 0.." << (totalFramesAvailable - 1) << ".\n";
        return -1;
    }
    if (limitEndFrame && endFrame < startFrame) {
        std::cerr << "[Error] --end-frame must be greater than or equal to --start-frame when both are active.\n";
        return -1;
    }

    const long long decodeStartFrame = (startFrame > 0) ? static_cast<long long>(startFrame - 1) : 0LL;
    if (decodeStartFrame > (std::numeric_limits<long long>::max() / frameBytes)) {
        std::cerr << "[Error] --start-frame produced an overflow while computing byte offset.\n";
        return -1;
    }
    const long long decodeStartByteOffset = decodeStartFrame * frameBytes;
    is.clear();
    is.seekg(static_cast<std::streamoff>(decodeStartByteOffset), std::ios::beg); // Seek to requested decode start.
    if (!is.good()) {
        std::cerr << "[Error] Failed to seek to requested --start-frame position.\n";
        return -1;
    }

    const bool suppressPreRollOutput = (startFrame > 0);
    log << "Input Frames Available: " << totalFramesAvailable << "\n";
    log << "Start Frame (0-based): " << startFrame << "\n";
    log << "End Frame: " << (limitEndFrame ? std::to_string(endFrame) : "unlimited") << "\n";

    OutputState outputState;
    outputState.mode = outputMode;
    outputState.tbcPipeMode = tbcPipeMode;
    outputState.fullFrame = fullFrame;
    outputState.firstLine = firstLine;
    outputState.lastLine = lastLine;
    if (outputMode == OutputMode::Tbc) {
        if (writeTbcPipeToStdout) {
            outputState.tbcPipeStream = &std::cout;
        } else {
            outputState.osLuma.open(lumaFile, std::ios::binary);
            outputState.osChroma.open(chromaFile, std::ios::binary);
            if (!outputState.osLuma.is_open() || !outputState.osChroma.is_open()) {
                std::cerr << "[Error] Failed to open one or both TBC output files for writing.\n";
                return -1;
            }
        }
    } else if (outputMode == OutputMode::Y4m) {
        std::ostream* y4mStream = nullptr;
        if (writeY4mToStdout) {
            y4mStream = &std::cout;
        } else {
            outputState.osY4m.open(outPath, std::ios::binary);
            if (!outputState.osY4m.is_open()) {
                std::cerr << "[Error] Failed to open Y4M output file for writing: " << outPath << "\n";
                return -1;
            }
            y4mStream = &outputState.osY4m;
        }
        try {
            Y4mNtscConfig y4mConfig;
            y4mConfig.fullFrame = fullFrame;
            y4mConfig.forceLimited = forceLimited;
            y4mConfig.levelsOverride = levelsSpecified;
            y4mConfig.black16bIreOverride = black16bIreOverride;
            y4mConfig.white16bIreOverride = white16bIreOverride;
            y4mConfig.activeVideoStartOverride = fullFrame ? -1 : activeVideoStart;
            y4mConfig.activeVideoEndOverride = fullFrame ? -1 : activeVideoEnd;
            y4mConfig.firstLine = firstLine;
            y4mConfig.lastLine = lastLine;
            y4mConfig.frameIndexOffset = static_cast<std::size_t>(startFrame);
            y4mConfig.chromaGain = chromaGain;
            outputState.y4mWriter = std::make_unique<Y4mNtscWriter>(metadata, y4mConfig, *y4mStream);
        }
        catch (const std::exception& e) {
            std::cerr << "[Error] Failed to initialize Y4M writer: " << e.what() << "\n";
            return -1;
        }
    } else if (writeRawToStdout) {
        outputState.rawStream = &std::cout;
    } else {
        outputState.osRaw.open(outPath, std::ios::binary);
        if (!outputState.osRaw.is_open()) {
            std::cerr << "[Error] Failed to open raw output file for writing: " << outPath << "\n";
            return -1;
        }
        outputState.rawStream = &outputState.osRaw;
    }

    // Initialize ONNX (with exception handling)
    log << "Initializing ONNX Runtime...\n";
    log << "Model Path: " << resolvedModelPath.string() << "\n";
    log << "GPU Device: " << gpuId << "\n";
    log << "TensorRT trt_max_partition_iterations: " << trtMaxPartitionIterations << "\n";
    log << "TensorRT trt_min_subgraph_size: " << trtMinSubgraphSize << "\n";
    std::filesystem::path trtCachePath = (executableDir / "trt_cache").lexically_normal();
    std::error_code trtCachePathEc;
    std::filesystem::create_directories(trtCachePath, trtCachePathEc);
    if (trtCachePathEc) {
        std::cerr << "[Error] Failed to create TensorRT cache directory '" << trtCachePath.string() << "': " << trtCachePathEc.message() << ".\n";
        return -1;
    }
    std::string trtCachePathString = trtCachePath.string();
    log << "TensorRT Engine Cache: " << trtCachePathString << "\n";
    auto trtCacheHasFiles = [](const std::filesystem::path& cachePath) {
        std::error_code ec;
        std::filesystem::recursive_directory_iterator it(cachePath, std::filesystem::directory_options::skip_permission_denied, ec);
        std::filesystem::recursive_directory_iterator end;
        while (!ec && it != end) {
            std::error_code entryEc;
            if (it->is_regular_file(entryEc) && !entryEc) return true;
            it.increment(ec);
        }
        return false;
    };
    bool tensorRtCacheHasFiles = trtCacheHasFiles(trtCachePath);
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "nnTransform3D");
    Ort::SessionOptions session_options;
    bool tensorRtProviderAppended = false;

    // --- TensorRT + CUDA Fallback Configuration ---
    try {
        // 1. Configure TensorRT
        OrtTensorRTProviderOptions trt_options{};
        trt_options.device_id = gpuId;
        trt_options.trt_max_partition_iterations = trtMaxPartitionIterations;
        trt_options.trt_min_subgraph_size = trtMinSubgraphSize;

        // Enable FP16, leverage Tensor Core for immense speedup
        trt_options.trt_fp16_enable = 1;

        // Enable Engine cache
        // It takes a few minutes for TensorRT to compile ONNX into an Engine for the first time.
        // With cache enabled, only the first run will be slow; subsequent startups will only take a few seconds.
        trt_options.trt_engine_cache_enable = 1;
        trt_options.trt_engine_cache_path = trtCachePathString.c_str();

        // Append TensorRT Provider
        session_options.AppendExecutionProvider_TensorRT(trt_options);
        tensorRtProviderAppended = true;
        log << "TensorRT Execution Provider appended successfully." << std::endl;

        // 2. Configure CUDA as fallback
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = gpuId;
        cuda_options.arena_extend_strategy = 0;
        session_options.AppendExecutionProvider_CUDA(cuda_options);
        log << "CUDA Fallback Provider appended successfully." << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to append CUDA provider: " << e.what() << std::endl;
        std::cerr << "Falling back to CPU..." << std::endl;
    }

    std::unique_ptr<Ort::Session> session;
    try {
        if (tensorRtProviderAppended && !tensorRtCacheHasFiles) {
            log << "No cached TensorRT engine found. TensorRT will compile a new engine, this may take a few minutes!\n";
        }
        session = std::make_unique<Ort::Session>(env, resolvedModelPath.c_str(), session_options);
        log << "ONNX Session loaded successfully.\n";
    }
    catch (const std::exception& e) {
        std::cerr << "[Error] ONNX Runtime Exception: " << e.what() << "\n";
        return -1;
    }

    FrameBuffer frame0, frame1;

    // --- 1. Startup phase (LookBehind) ---
    log << "Entering Phase 1: LookBehind (Reading first frame)...\n";
    frame0.isPadding = true;
    if (!readInterlacedFrame(is, frame1)) {
        std::cerr << "[Error] Failed at initial frame read. Exiting.\n";
        return 0;
    }
    // Send the newly read first frame into GPU
    cudaMemcpy(frame1.d_cvbs, frame1.cvbs.data(), FRAME_WIDTH * FRAME_HEIGHT * sizeof(uint16_t), cudaMemcpyHostToDevice);

    log << "First frame read successfully. Executing first 3D split...\n";
    processSplit3D(frame0, frame1, *session, activeVideoStart, activeVideoEnd);

    std::swap(frame0, frame1);
    frame1.resetOLA();

    // --- 2. Main loop ---
    log << "Entering Phase 2: Main Processing Loop...\n";
    bool dropNextOutputFrame = suppressPreRollOutput;
    int decodedFrameCount = 1;
    int emittedFrameCount = 0;
    long long currentOutputFrameIndex = decodeStartFrame;
    bool endFrameReachedWithLookahead = false;
    while (readInterlacedFrame(is, frame1)) {
        decodedFrameCount++;
        try {
            // Immediately send to GPU upon reading each frame
            cudaMemcpy(frame1.d_cvbs, frame1.cvbs.data(), FRAME_WIDTH * FRAME_HEIGHT * sizeof(uint16_t), cudaMemcpyHostToDevice);
            processSplit3D(frame0, frame1, *session, activeVideoStart, activeVideoEnd);
            // After processing, fetch back the overlap-added Chroma and Weight from GPU to CPU
            cudaMemcpy(frame0.accChroma.data(), frame0.d_accChroma, FRAME_WIDTH * FRAME_HEIGHT * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(frame0.weightSum.data(), frame0.d_weightSum, FRAME_WIDTH * FRAME_HEIGHT * sizeof(double), cudaMemcpyDeviceToHost);
            if (dropNextOutputFrame) {
                log << "Skipping pre-roll output frame at source index " << currentOutputFrameIndex << ".\n";
                dropNextOutputFrame = false;
            } else {
                // Write back to disk or raw stream
                finalizeAndWriteOutput(frame0, outputState, activeVideoStart, activeVideoEnd);
                emittedFrameCount++;
            }
            if (limitEndFrame && currentOutputFrameIndex >= endFrame) {
                endFrameReachedWithLookahead = true;
                break;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "\n[Fatal Crash at Decode Frame " << decodedFrameCount << "] " << e.what() << "\n";
            std::cerr << "Emergency saving and exiting...\n";
            break;
        }

        std::swap(frame0, frame1);
        frame1.resetOLA();

        currentOutputFrameIndex++;
        if (!quietProgressLog && emittedFrameCount > 0 && emittedFrameCount % 100 == 0) {
            log << "[Info] Processed " << emittedFrameCount << " frames...\n";
        }
    }

    // --- 3. Finalization phase (LookAhead) ---
    if (endFrameReachedWithLookahead) {
        log << "End frame reached with real lookahead; skipping final padding phase.\n";
    } else {
        log << "Entering Phase 3: LookAhead (Finalizing last frame)...\n";
        frame1.isPadding = true;
        processSplit3D(frame0, frame1, *session, activeVideoStart, activeVideoEnd);

        // Bring the final frame results back to CPU
        cudaMemcpy(frame0.accChroma.data(), frame0.d_accChroma, FRAME_WIDTH * FRAME_HEIGHT * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(frame0.weightSum.data(), frame0.d_weightSum, FRAME_WIDTH * FRAME_HEIGHT * sizeof(double), cudaMemcpyDeviceToHost);
        if (dropNextOutputFrame) {
            log << "Skipping pre-roll output frame at source index " << currentOutputFrameIndex << ".\n";
        } else if (!limitEndFrame || currentOutputFrameIndex <= endFrame) {
            finalizeAndWriteOutput(frame0, outputState, activeVideoStart, activeVideoEnd);
            emittedFrameCount++;
        }
    }

    log << "All Done! Total decode frames: " << decodedFrameCount << ", emitted frames: " << emittedFrameCount << "\n";
    return 0;
}
