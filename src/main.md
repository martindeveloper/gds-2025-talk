---
marp: true
theme: fr-light
paginate: false
transition: fade
---

# x86, ARM, and Performance Basics 🔧
## CPU Architecture Internals
*Martin Pernica*

---

## Introduction
- **Why CPU Knowledge Matters** 🎯  
  - Modern games push CPUs to their limits
- **Multi-Architecture Reality** 🌐  
  - x86-64: PC gaming, PlayStation 5, Xbox Series X/S (AMD Zen 2)
  - ARM: Nintendo Switch (Custom NVIDIA SoC), mobile gaming, Apple Silicon Macs
- **Goal** 🎮  
  - Understand CPU overall architecture better and how it affects performance

---

## CPU Architecture Basics
- **Core Components** ⚙️  
  - **ALU (Arithmetic Logic Unit)**: Integer/FP math, bitwise operations
  - **Control Unit**: Instruction fetch, decode, dispatch, retirement
  - **Load-Store Unit**: Memory access coordination
  - **Branch Predictor**: Speculative execution control
- **x86-64 Specifics** 🖥️  
  - 16 general-purpose registers (RAX-R15)
  - 16-32 vector registers (XMM/YMM/ZMM depending on instruction set)
  - RFLAGS register for condition codes
- **ARM64 (AArch64) Specifics** 📱  
  - 31 GPRs: X0-X30 (64-bit), W0-W30 (32-bit views)
  - PSTATE for processor state
  - No segmentation; weaker default memory ordering than x86 (explicit barriers often required)

---

## Pipelining and Out-of-Order Execution
- **Pipeline Stages**
  - **Frontend** In-order
    - **Fetch**: Instruction retrieval from L1I cache and branch prediction
    - **Decode**: Both x86 and ARM decode to micro-ops; x86 has more complex variable-length decoding
    - **Rename**: Register renaming for out-of-order execution
  - **Backend** Out-of-order
    - **Schedule**: Issue to execution units when operands ready
    - **Execute**: ALU, FPU, Load/Store operations
    - **Retire**: Commit results in program order (in-order again)
- **Superscalar Execution** multiple instructions per cycle
  - **x86**: Peak decode/issue width ~4–6 µops/cycle (varies by generation)
  - **ARM**: Peak decode/issue width ~3–8 µops/cycle (implementation-dependent)

---

## Instruction Pipeline (Generic)

<div class="mermaid">
graph LR
    subgraph "Front End (In-Order)"
        BP[Branch Predictor<br>Predicts Next PC]
        IF[Instruction Fetch<br>L1I Cache]
        ID[Instruction Decode<br>Decode to µops]
        RN[Register Rename<br>Map to Physical Regs<br>Allocate ROB Entry]
        BP --> IF
        IF --> ID
        ID --> RN
    end
    subgraph "Back End (Out-of-Order)"
        SC[Scheduler<br>Wait for Operands<br>Issue When Ready]
        subgraph "Execution Units"
            ALU[ALU/Integer]
            FPU[FPU/SIMD]
            LS[Load/Store Unit]
        end
        SC --> ALU
        SC --> FPU
        SC --> LS
    end
    subgraph "Memory"
        L1D[L1D Cache]
        LS --> L1D
    end
    subgraph "Retirement (In-Order)"
        ROB[Reorder Buffer<br>Track In-Flight Instructions]
        CM[Commit<br>Update Architectural State<br>In Program Order]
        ROB --> CM
    end
    RN --> SC
    RN --> ROB
    ALU --> ROB
    FPU --> ROB
    L1D --> ROB
    style BP fill:#f0e6f6,stroke:#333
    style IF fill:#c4a7e7,stroke:#333
    style ID fill:#c4a7e7,stroke:#333
    style RN fill:#c4a7e7,stroke:#333
    style SC fill:#ea9a97,stroke:#333
    style ALU fill:#f8e1e0,stroke:#333
    style FPU fill:#f8e1e0,stroke:#333
    style LS fill:#f8e1e0,stroke:#333
    style L1D fill:#9ccfd8,stroke:#333
    style ROB fill:#ebd9a3,stroke:#333
    style CM fill:#ebd9a3,stroke:#333
</div>

---

## x86 Instruction Set Evolution

### x86 SIMD/Vector Extensions Timeline 📅
- **1997**: **MMX** - 64-bit registers, integer operations only
- **1999**: **SSE** - 128-bit XMM registers, single-precision floating point (4 floats)
- **2001**: **SSE2** - Double-precision FP, integer operations on 128-bit vectors
- **2004**: **SSE3** - Horizontal operations, complex arithmetic
- **2006**: **SSSE3** - Additional packed operations, improved shuffles
- **2007**: **SSE4.1/4.2** - Dot products, string processing, CRC32
- **2011**: **AVX** - 256-bit YMM registers (8 floats), non-destructive operations
- **2013**: **AVX2** - Integer operations on 256-bit vectors; **FMA3** (Fused Multiply-Add) introduced in the same CPU generation
- **2017**: **AVX-512** - 512-bit ZMM registers (16 floats), masking, embedded rounding
- **2019**: **AVX-VNNI** - Vector Neural Network Instructions for AI inference
- **2023**: **Intel AMX** - Advanced Matrix Extensions for AI acceleration
- **2024**: **AVX10** - Unified 256/512-bit vector ISA (different from AVX-512)

---

## x86-64 CPU Architecture Overview

<div class="mermaid">
graph LR
    subgraph "x86-64 CPU Core"
        FU[Fetch Unit]
        UC[µop Cache]
        DU[Decode Unit]
        RAT[Register Rename]
        subgraph "Backend"
            RS[Scheduler/RS]
            ROB[Reorder Buffer]
        end
        subgraph "Execution Units"
            ALU[ALUs x4]
            AGU[AGU x3]
            FPU[FP/Vector x2]
            BRU[Branch Unit]
        end
        PRF[Physical Register File<br/>~300 registers]
        subgraph "Memory Subsystem"
            LSQ[Load/Store Queue]
            L1D[L1D 32KB]
            L2[L2 1-2MB]
            L3[L3 Shared]
        end
        L1I[L1I 32KB]
        BP[Branch Predictor]
    end
    RAM[DDR4/DDR5]
    BP --> FU
    FU --> L1I
    L1I --> DU
    FU --> UC
    UC --> RAT
    DU --> UC
    RAT --> RS
    RAT --> ROB
    RS --> ALU
    RS --> AGU
    RS --> FPU
    RS --> BRU
    ALU --> PRF
    FPU --> PRF
    AGU --> LSQ
    LSQ --> L1D
    ROB --> RAT
    L1D --> L2
    L2 --> L3
    L3 --> RAM
    style FU fill:#c4a7e7
    style DU fill:#c4a7e7
    style UC fill:#c4a7e7
    style RAT fill:#f6c177
    style RS fill:#ea9a97
    style ROB fill:#ea9a97
    style L1I fill:#9ccfd8
    style L1D fill:#9ccfd8
    style L2 fill:#9ccfd8
    style L3 fill:#9ccfd8
</div>

---

## ARM Instruction Set Evolution

### ARM SIMD/Vector Extensions Timeline 📅
- **2005**: **NEON (Advanced SIMD)** - 128-bit quad registers, parallel operations
- **2016**: **ARMv8.1-A** - Advanced SIMD improvements, half-precision FP
- **2017**: **SVE (Scalable Vector Extension)** - 128–2048 bit scalable vectors (initial spec)
- **2019**: **ARMv8.2-A** - Dot product instructions, enhanced half-precision arithmetic
- **2021**: **ARMv9.0-A (feat. SVE2)** - Scalable Vector Extension 2, enhanced security
- **2022**: **ARMv9.2-A + SME** - Scalable Matrix Extension for AI/ML workloads

### Gaming Performance Impact 🎮
- **Early 2000s**: MMX/SSE for basic audio/graphics processing
- **2010s**: AVX/NEON for physics simulations, AI pathfinding
- **2020s**: AVX-512/SVE2 for select CPU paths; AMX/SME for AI/ML

---

## ARM64 (AArch64) CPU Architecture Overview

<div class="mermaid">
graph LR
    subgraph "ARM64 CPU Core"
        FU[Fetch Unit]
        DU[Decode Unit<br/>6-10 wide]
        RAT[Rename/Allocate]
        subgraph "Backend"
            RS[Issue Queues]
            ROB[Reorder Buffer]
        end
        subgraph "Execution Units"
            ALU[Integer ALUs x4-6]
            BRU[Branch Unit x2]
            FPU[FP/ASIMD x4]
            LSU[Load/Store x2-3]
        end
        PRF[Physical Registers<br/>Int + FP/Vec]
        subgraph "Memory Subsystem"
            LSQ[Load/Store Queue]
            L1D[L1D 64-128KB]
            L2[L2 1-16MB]
            L3[L3/SLC]
            TLB[TLB]
        end
        L1I[L1I 64-192KB]
        MOP[Macro-op Cache]
        BP[Branch Predictor]
    end
    RAM[LPDDR5/DDR5]
    BP --> FU
    FU --> L1I
    L1I --> DU
    FU --> MOP
    MOP --> RAT
    DU --> MOP
    RAT --> RS
    RAT --> ROB
    RS --> ALU
    RS --> BRU
    RS --> FPU
    RS --> LSU
    ALU --> PRF
    FPU --> PRF
    LSU --> LSQ
    LSQ --> L1D
    LSQ --> TLB
    ROB --> RAT
    L1D --> L2
    L2 --> L3
    L3 --> RAM
    style FU fill:#c4a7e7
    style DU fill:#c4a7e7
    style MOP fill:#c4a7e7
    style RAT fill:#f6c177
    style RS fill:#ea9a97
    style ROB fill:#ea9a97
    style L1I fill:#9ccfd8
    style L1D fill:#9ccfd8
    style L2 fill:#9ccfd8
    style L3 fill:#9ccfd8
</div>

---

## Architecture Comparison: x86-64 vs ARM64

| Feature | x86-64 | ARM64 |
|---------|--------|-------|
| **Instruction Set** | CISC (Complex) | RISC (Reduced) |
| **Instruction Width** | Variable (1–15 bytes) | Fixed (32-bit) |
| **General Registers** | 16 (RAX-R15) | 31 (X0-X30) |
| **Vector Registers** | 16–32 XMM/YMM/ZMM (128–512-bit) | 32 V (128-bit NEON), Scalable (SVE/SVE2) |
| **Addressing Modes** | Complex (9+ modes) | Simple (3 modes) |
| **Branch Prediction** | Advanced, multi-level (e.g., TAGE, perceptron) | Advanced (e.g., multi-level, neural on high-perf cores) |
| **Power Efficiency** | Generally lower (esp. at low wattages) | Generally higher (better perf/watt) |
| **Decode Width** | Typically 4–6 instructions/cycle (up to 8) | Typically 3–8 instructions/cycle (varies by design) |
| **Gaming Performance** | Historically strong, mature ecosystem | Excellent perf/watt; Apple Silicon often approaches or exceeds x86 |

---

## Code Example: Register Use

**C Code:**
```c
// Update player position with velocity and clamp to bounds
void move_player(float* pos_x, float vel_x, float delta_time, float min_x, float max_x) {
    *pos_x += vel_x * delta_time;
    *pos_x = fmaxf(min_x, fminf(*pos_x, max_x));
}
```

**x86-64 Assembly (System V ABI - Linux, macOS, PS5):**
```nasm
; Parameters: pos_x (RDI), vel_x (XMM0), delta_time (XMM1), min_x (XMM2), max_x (XMM3)
move_player:
    movss xmm4, [rdi]      ; Load *pos_x
    mulss xmm0, xmm1       ; vel_x * delta_time
    addss xmm4, xmm0       ; *pos_x += displacement
    minss xmm4, xmm3       ; Clamp upper bound: min(pos, max_x)
    maxss xmm4, xmm2       ; Clamp lower bound: max(pos, min_x)
    movss [rdi], xmm4      ; Store *pos_x
    ret
```

**Note:** `minss` then `maxss` - order matters for correct clamping

---

## Cache Memory Hierarchy
- **Modern Cache Structure** 🏗️  
  - **L1**: 32–64KB/core, 3–5 cycles latency (instruction + data split)
  - **L2**: 512KB–4MB/core, 10–14 cycles (unified instruction/data)
  - **L3**: 8–128MB shared, 25–60 cycles (last-level cache)
  - **RAM**: 8–64GB, 100–200+ cycles (DRAM access penalty, highly variable)
- **Cache Line Behavior** 📦  
  - **Size**: 64 bytes (both x86 and ARM)
  - **Coherency**: MESI protocol (or variants like MOESI/MESIF)
  - **Prefetching**: Hardware detects sequential/strided patterns
- **Gaming Performance Impact** 🎮  
  - Cache miss = ~100–300 cycles lost
  - L3 cache affects texture streaming and pop-in
  - Data layout (SoA vs AoS) impacts physics performance

---

## Cache-Optimal Data Structures

### Problem: Array of Structures (AoS)
```c
// Bad: Array of Structures (AoS) - poor cache usage ❌
struct Particle_AoS {
    float x, y, z;           // Position (12 bytes)
    float vx, vy, vz;        // Velocity (12 bytes)  
    float r, g, b, a;        // Color (16 bytes)
    float life, mass;        // Properties (8 bytes)
    int active;              // State (4 bytes)
    char padding[12];        // Align to 64 bytes total
};

void update_positions_aos(Particle_AoS* particles, int count) {
    for (int i = 0; i < count; i++) {
        particles[i].x += particles[i].vx;  // Loads entire 64-byte particle
        particles[i].y += particles[i].vy;  // Already in cache
        particles[i].z += particles[i].vz;  // Still in cache
    }
}
```

---

## Cache-Optimal: Structure of Arrays (SoA)

### Solution: Separate Arrays for Better Cache Usage
```c
// Good: Structure of Arrays (SoA) - cache-friendly ✅
struct ParticleSystem_SoA {
    float* x;        // All X positions together
    float* y;        // All Y positions together  
    float* z;        // All Z positions together
    float* vx;       // All X velocities together
    // ... separate arrays for each component
};

void update_positions_soa(ParticleSystem_SoA* sys, int count) {
    for (int i = 0; i < count; i++) {
        sys->x[i] += sys->vx[i];  // Sequential access, optimal cache usage
        sys->y[i] += sys->vy[i];  // Arrays fit nicely in cache lines
        sys->z[i] += sys->vz[i];
    }
}
```

**Performance**: SoA can be 2-4x faster for vectorized operations! 🚀

---

## Instruction-Level Parallelism (ILP)

- CPU can actually execute multiple instructions per cycle
- Each core has multiple execution units which can be used in parallel
  - e.g., 4–12+ ports on modern Intel/AMD/Apple cores: ALUs, FP units, load/store units, branch units, etc.

### Problem: Dependent Instruction Chains
```c
// Bad: Dependent chain (serialized execution) ❌
void update_health_bad(Player* players, int count) {
    for (int i = 0; i < count; i++) {
        players[i].health -= players[i].damage_taken; // Health update
        if (players[i].health < 0) { // Depends on previous line
            players[i].health = 0;
        }
    }
}
```

- True data dependencies (read-after-write) create stalls: the second instruction cannot even out-of-order CPUs must wait for the result of the subtraction before they can do the comparison → limits ILP to ~1 instruction per cycle in this chain
- Modern CPUs extract ILP via out-of-order execution, register renaming, and speculative execution, but they are fundamentally bounded by the longest dependency chain in your code (critical path latency often matters more than instruction count)

---

## Instruction-Level Parallelism (ILP)

### Solution: Independent Operations
```c
// Good: Independent operations (ILP-friendly) ✅
void update_health_good(Player* players, int count) {
    for (int i = 0; i < count; i++) {
        int damage = players[i].damage_taken;
        int current_health = players[i].health;
        
        // Independent calculations - can execute in parallel
        int new_health = current_health - damage;
        int clamped_health = (new_health < 0) ? 0 : new_health;
        
        players[i].health = clamped_health;
        players[i].damage_taken = 0; // Reset for next frame
    }
}
```

---

## ILP Assembly: Out-of-Order Execution

### x86-64: Parallel Instruction Execution
```nasm
; x86-64: Demonstrates ILP within the loop body
; struct Player { int health; int damage_taken; int shield; int status; }; // 16 bytes
update_health_good:
    test    esi, esi        ; Check count
    jle     .done           ; Return if count <= 0
    xor     ecx, ecx        ; i = 0

.loop:
    ; --- These loads can execute in parallel (different load ports) ---
    mov     eax, [rdi]      ; eax = players[i].health
    mov     edx, [rdi + 4]  ; edx = players[i].damage_taken
    
    ; --- Arithmetic (independent from stores below) ---
    sub     eax, edx        ; new_health = health - damage
    xor     r8d, r8d        ; r8d = 0 (for clamping) ; --- may be executed in parallel with mov above
    
    ; --- Branchless clamping ---
    test    eax, eax        ; Check if new_health < 0
    cmovl   eax, r8d        ; If negative, use 0
    
    ; --- These stores can dispatch in parallel (store buffer) ---
    mov     [rdi], eax      ; players[i].health = clamped_health
    mov     [rdi + 4], r8d  ; players[i].damage_taken = 0
    
    ; --- Loop control ---
    add     rdi, 16         ; Advance to next Player
    inc     ecx             ; i++
    cmp     ecx, esi        ; Compare with count
    jl      .loop           ; Continue if i < count

.done:
    ret

; ILP opportunities in this code:
; 1. Two loads execute on different load ports simultaneously
; 2. Arithmetic operations overlap with memory operations
; 3. Store buffer allows parallel store dispatch
; 4. Branch prediction enables next iteration fetch during execution
```

---

## Branch Prediction and Speculative Execution
- **Branch Predictor Types**  
  - **Static**: Always taken/not-taken (simple loops)
  - **Dynamic**: Two-level adaptive predictors, perceptron-based, TAGE
  - **Indirect**: Target prediction for function pointers/virtual calls
- **Prediction Accuracy**  
  - **Modern CPUs**: 95–98% accuracy on typical code
  - **Gaming code**: Often 85–90% due to complex AI/physics logic
- **Misprediction Penalty**  
  - **x86**: Typically 15–20 cycles (Intel Alder Lake, AMD Zen 4)
  - **ARM**: Typically 12–20 cycles (Apple M3/M4, ARM Cortex-X4)
  - **Impact**: 1% misprediction rate = ~5% performance loss (rule of thumb)
- **Speculative Execution Security**  
  - Spectre/Meltdown vulnerabilities and mitigations
  - Modern CPUs have hardware mitigations; performance impact varies by workload

---

## Branch Optimization: Branchless Programming

### Problem: Unpredictable Branches
```c
typedef struct {
    int x, y, z, w;    // Position + padding (16 bytes)
    int type;          // Entity type (offset 16)
    int health;        // Health value (offset 20)
} Entity;              // Total: 24 bytes

// Bad: Unpredictable branching in collision detection ❌
bool check_collision_bad(Entity* entities, int count) {
    for (int i = 0; i < count; i++) {
        if (entities[i].type == ENEMY) {           // Unpredictable
            if (entities[i].x > 100) {             // Nested branch
                if (entities[i].health > 0) {      // More branching
                    return true;
                }
            }
        }
    }
    return false;
}
```
---

## Branch Optimization: Branchless Programming

### Solution: Branchless with Bit Manipulation
```c
// Good: Branchless with bit manipulation ✅
int count_active_enemies(Entity* entities, int count) {
    int active_count = 0;
    for (int i = 0; i < count; i++) {
        // Branchless: convert boolean to 0/1
        int is_enemy = (entities[i].type == ENEMY);
        int is_alive = (entities[i].health > 0);
        int is_in_bounds = (entities[i].x > 100);
        
        // Combine conditions without branches
        active_count += is_enemy & is_alive & is_in_bounds;
    }
    return active_count;
}
```

---

## Assembly: Conditional Moves vs. Branches

### x86-64: Conditional Moves Avoid Mispredictions
```nasm
; x86-64: Conditional moves avoid branches
; struct Entity { int x; int y; int z; int w; int type; int health; }; // 24 bytes
;                 ^0    ^4    ^8    ^12   ^16    ^20
count_active_enemies:
	xor     eax, eax        ; active_count = 0
	test    rsi, rsi        ; Check count
	jle     .done           ; Return if count <= 0
.loop:
    mov edx, [rdi+16]       ; Load entity type
    cmp edx, 1              ; Compare with ENEMY
    sete dl                 ; Set dl = 1 if enemy
    
    mov r8d, [rdi+20]       ; Load health
    test r8d, r8d           ; Test if > 0
    setg r8b                ; Set r8b = 1 if alive
    
    mov r9d, [rdi]          ; Load x position
    cmp r9d, 100            ; Compare with 100
    setg r9b                ; Set r9b = 1 if in bounds
    
    and dl, r8b             ; Combine conditions
    and dl, r9b             ; All must be true
    movzx edx, dl           ; Zero-extend to 32-bit
    add eax, edx            ; Add to count (branchless)
    
    add rdi, 24             ; Next entity
    dec rsi                 ; Decrement counter
    jnz .loop               ; Continue if not zero

.done:
	ret
```

---

## Thread-Level Parallelism (TLP)

- **When Single-Core Isn't Enough**: Saturated or stalled cores need Thread-Level Parallelism (TLP)
- **Multithreading for Throughput**: Increases total work performed by utilizing multiple cores
- **Simultaneous Multithreading (SMT / Hyper-Threading)**:
  - Allows the core to issue from multiple hardware threads in the same cycle
  - Helps keep execution units busy by filling pipeline bubbles from stalls
  - Not true parallelism-better utilization of single-core resources
- **Shared Resources & Trade-offs**:
  - SMT threads share L1/L2 caches and branch predictors
  - Thread interference can reduce branch prediction accuracy
  - Cache contention from data eviction between threads

---

## Hybrid Architectures: P-Cores & E-Cores
- **The Rise of Heterogeneity**
  - **Intel Hybrid (Alder Lake onwards)**: Performance-cores (P-Cores) + Efficiency-cores (E-Cores)
  - **ARM big.LITTLE**: Similar concept, widely used in mobile and now server/desktop
- **Core Characteristics**
  - **P-Cores (Performance)**: Wide, out-of-order, high frequency, SMT/Hyper-Threading
    - *Ideal for*: Latency-sensitive tasks (main game loop, render thread, physics, AI)
  - **E-Cores (Efficiency)**: Narrower, simpler OoO or in-order, lower frequency, power-efficient
    - *Ideal for*: Background tasks (audio, asset streaming, networking, non-critical systems)
---

## Hybrid Architectures: P-Cores & E-Cores

- **Developer Strategies**
  - **Job Systems**: Design task schedulers to be aware of core types
  - **OS Schedulers**: Modern OSes (Windows 11, macOS, Linux) are increasingly P/E-core aware
    - Provide hints via QoS classes or thread priorities
  - **Manual Thread Affinity**: Pin critical threads to P-Cores, background threads to E-Cores
    - `SetThreadAffinityMask` (Windows) - use sparingly; can fight the scheduler on hybrid systems
      - Prefer `SetThreadInformation` with ThreadPowerThrottling / QoS hints where possible
    - `pthread_setaffinity_np` (Linux)
    - `DispatchQueue(label: "myqueue", qos: .userInteractive)` or `pthread_attr_set_qos_class_np` (macOS)

---

## Synchronization and Game Threading Patterns
- **Synchronization Primitives**  
  - **Atomic operations**: Lock-free programming, memory ordering
  - **Memory barriers**: Enforce cross-thread ordering guarantees
  - **Fast mutexes**: `futex` (Linux), `WaitOnAddress` (Windows), `os_unfair_lock` (macOS)
- **Game Threading Patterns**  
  - **Main thread**: Game logic, input handling
  - **Render thread**: GPU command submission
  - **Worker threads**: Physics, AI, audio processing
  - **Job systems**: Fine-grained task parallelism

---

## Code Example: Lock-Free Programming
```c
// Lock-free circular buffer (single producer, single consumer)
#include <stdatomic.h>

#define BUFFER_SIZE 256 // Example size

typedef struct {
    atomic_size_t head;
    atomic_size_t tail;
    void* buffer[BUFFER_SIZE];
} LockFreeQueue;

// Producer (audio thread)
bool enqueue(LockFreeQueue* queue, void* item) {
    size_t current_tail = atomic_load_explicit(&queue->tail, memory_order_relaxed);
    size_t next_tail = (current_tail + 1) % BUFFER_SIZE;
    
    // Check if queue is full
    if (next_tail == atomic_load_explicit(&queue->head, memory_order_acquire)) {
        return false;  // Queue full
    }
    
    queue->buffer[current_tail] = item;
    atomic_store_explicit(&queue->tail, next_tail, memory_order_release);
    return true;
}

// Consumer (main thread)
bool dequeue(LockFreeQueue* queue, void** item) {
    size_t current_head = atomic_load_explicit(&queue->head, memory_order_relaxed);
    
    // Check if queue is empty
    if (current_head == atomic_load_explicit(&queue->tail, memory_order_acquire)) {
        return false;  // Queue empty
    }
    
    *item = queue->buffer[current_head];
    atomic_store_explicit(&queue->head, (current_head + 1) % BUFFER_SIZE, 
                         memory_order_release);
    return true;
}
```

---

## Memory Consistency: When Lock-Free Code Breaks

### The Problem: Different Memory Models
- **x86-64**: Total Store Order (TSO) - strong memory model
- **ARM64**: Weak memory ordering - requires explicit barriers
- **Real Impact**: Code that works on Intel/AMD may break on Apple Silicon! ⚠️

### x86-64 (TSO)
Stores become visible to other cores in program order, and loads can't pass older stores to the same address. This "almost sequential" behavior often makes racy code seem to work.

### ARM64 (weak ordering)
Hardware can reorder both loads and stores aggressively. A producer might publish an index before data is visible, or a consumer might read stale values. You need acquire/release or explicit barriers-your algorithm must be correct from the C++ memory model alone, not rely on stronger hardware guarantees.

---

## Memory Consistency: When Lock-Free Code Breaks

### Lock-Free Queue: No Synchronization (Broken)
```c
#define BUFFER_SIZE 256

// ⚠️ WRONG: No synchronization - undefined behavior in C/C++! x86 TSO often masks the bug, but it's not guaranteed.
typedef struct {
    size_t head;    // Consumer index (no synchronization!)
    size_t tail;    // Producer index (no synchronization!)
    void* buffer[BUFFER_SIZE];
} BrokenQueue;

// Producer thread - data race, undefined behavior
bool broken_enqueue(BrokenQueue* queue, void* item) {
    size_t current_tail = queue->tail;
    size_t next_tail = (current_tail + 1) % BUFFER_SIZE;
    
    if (next_tail == queue->head) return false;
    
    queue->buffer[current_tail] = item;  // No ordering guarantee!
    queue->tail = next_tail;             // Compiler/CPU can reorder
    return true;
}

// Consumer thread - data race, undefined behavior
bool broken_dequeue(BrokenQueue* queue, void** item) {
    size_t current_head = queue->head;
    
    if (current_head == queue->tail) return false;  // Empty check
    
    *item = queue->buffer[current_head];  // Might read garbage!
    queue->head = current_head + 1;
    return true;
}
```

---

## Memory Consistency: The ARM Fix

### Solution: Explicit Memory Barriers
```c
// ARM-safe version with memory barriers
#include <stdatomic.h>
#define BUFFER_SIZE 256

typedef struct {
    atomic_size_t head;
    atomic_size_t tail;  
    void* buffer[BUFFER_SIZE];
} SafeQueue;

bool safe_enqueue(SafeQueue* queue, void* item) {
    size_t current_tail = atomic_load_explicit(&queue->tail, memory_order_relaxed);
    size_t next_tail = (current_tail + 1) % BUFFER_SIZE;
    
    if (next_tail == atomic_load_explicit(&queue->head, memory_order_acquire)) return false;
    
    queue->buffer[current_tail] = item;
    // Release semantics: all previous stores complete before this store
    atomic_store_explicit(&queue->tail, next_tail, memory_order_release);
    return true;
}

bool safe_dequeue(SafeQueue* queue, void** item) {
    size_t current_head = atomic_load_explicit(&queue->head, memory_order_relaxed);
    
    if (current_head == atomic_load_explicit(&queue->tail, memory_order_acquire)) return false;
    
    *item = queue->buffer[current_head];
    atomic_store_explicit(&queue->head, (current_head + 1) % BUFFER_SIZE, memory_order_release);
    return true;
}

// Alternative: Manual memory barrier (ARM assembly), dmb ish  // Data Memory Barrier - Inner Shareable domain, C++ equivalent: std::atomic_thread_fence(std::memory_order_release);
```

---

## SIMD: Single Instruction, Multiple Data
- **SIMD Evolution**  
  - **x86**: MMX → SSE → SSE2/3/4 → AVX → AVX2 → AVX-512
  - **ARM**: NEON (128-bit) → SVE/SVE2 (scalable, 128–2048 bit)
- **Vector Widths**  
  - **SSE**: 128-bit (4 floats, 8 shorts, 16 bytes)
  - **AVX2**: 256-bit (8 floats, 16 shorts, 32 bytes)
  - **AVX-512**: 512-bit (16 floats, 32 shorts, 64 bytes)
  - **ARM NEON**: 128-bit (4 floats, 8 shorts, 16 bytes)
- **Game Applications**  
  - Matrix transformations: 4x4 matrices fit perfectly in 4x 128-bit vectors
  - Audio processing: DSP effects on multiple channels
  - Physics: Parallel constraint solving, collision detection
  - Graphics: Vertex transformations, color space conversions

---

## Code Example: SIMD 4x4 Matrix-Vector Multiplication

### x86 SSE Implementation (4x4 Matrix * 4D Vector)
```c
#include <immintrin.h>

// Standard pattern used by glm, DirectXMath, etc.
// Matrix is row-major: result = vec * matrix (row-vector pre-multiply)
void mat4_vec4_mul_sse(const float* matrix, const float* vec, float* result) {
    __m128 v = _mm_load_ps(vec);
    
    // Broadcast each vector component and multiply with matrix row
    __m128 r0 = _mm_load_ps(&matrix[0]);
    __m128 r1 = _mm_load_ps(&matrix[4]);
    __m128 r2 = _mm_load_ps(&matrix[8]);
    __m128 r3 = _mm_load_ps(&matrix[12]);
    
    // Multiply-add pattern: sum of (row[i] * vec[i]) for each row
    __m128 result_vec = _mm_mul_ps(r0, _mm_shuffle_ps(v, v, _MM_SHUFFLE(0,0,0,0)));
    result_vec = _mm_add_ps(result_vec, _mm_mul_ps(r1, _mm_shuffle_ps(v, v, _MM_SHUFFLE(1,1,1,1))));
    result_vec = _mm_add_ps(result_vec, _mm_mul_ps(r2, _mm_shuffle_ps(v, v, _MM_SHUFFLE(2,2,2,2))));
    result_vec = _mm_add_ps(result_vec, _mm_mul_ps(r3, _mm_shuffle_ps(v, v, _MM_SHUFFLE(3,3,3,3))));
    
    _mm_store_ps(result, result_vec);
}
```

---

## SIMD Matrix-Vector Multiplication: ARM NEON

### ARM NEON Implementation (4x4 Matrix * 4D Vector)
```c
#include <arm_neon.h>

// Standard pattern using vfmaq_laneq_f32 (FMA)
// Matrix is row-major: result = vec * matrix (row-vector pre-multiply)
void mat4_vec4_mul_neon(const float* matrix, const float* vec, float* result) {
    float32x4_t v = vld1q_f32(vec);
    
    float32x4_t r0 = vld1q_f32(&matrix[0]);
    float32x4_t r1 = vld1q_f32(&matrix[4]);
    float32x4_t r2 = vld1q_f32(&matrix[8]);
    float32x4_t r3 = vld1q_f32(&matrix[12]);
    
    // Multiply-add: broadcast vec[i] and FMA with row
    float32x4_t result_vec = vmulq_laneq_f32(r0, v, 0);
    result_vec = vfmaq_laneq_f32(result_vec, r1, v, 1);
    result_vec = vfmaq_laneq_f32(result_vec, r2, v, 2);
    result_vec = vfmaq_laneq_f32(result_vec, r3, v, 3);
    
    vst1q_f32(result, result_vec);
}
```

---

## Future-Proof SIMD: ARM SVE2 (Scalable Vector Extensions)

### Beyond Fixed-Width SIMD
- **Traditional SIMD**: Fixed width (128-bit NEON, 256-bit AVX2, 512-bit AVX-512)
- **ARM SVE/SVE2**: Scalable from 128-bit to 2048-bit at runtime!
- **Reality Check**:
  - **Current**: Apple M-series use 128-bit NEON
  - **Server ARM**: Neoverse implements SVE/SVE2 with varying widths
  - **Future**: Wider SVE2 adoption in high-performance ARM SoCs

---

## Future-Proof SIMD: ARM SVE2 (Scalable Vector Extensions)

### Vector-Length-Agnostic Code
```c
// Same C code compiles for ANY vector width (128-bit to 2048-bit)
#include <arm_sve.h>

void transform_vertices_sve(float* positions, float* matrices, float* output, size_t vertex_count) {
    // Get the vector length at runtime (count of 32-bit float elements per vector)
    size_t vl_f32 = svcntw(); 
    
    for (size_t i = 0; i < vertex_count; i += vl_f32) {
        // Create predicate for remaining elements (handles partial vectors at the end)
        svbool_t pred = svwhilelt_b32(i, vertex_count);
        
        // Load vertex X positions - auto-adapts to vector width
        // Assuming positions is SoA for X: [x0, x1, x2, ...]
        // This example processes a single component to highlight SVE2 VL-agnostic loops
        svfloat32_t pos_x_vals = svld1_f32(pred, &positions[i]);
        // svfloat32_t y = svld1_f32(pred, &positions[i * 4 + 1]); 
        // svfloat32_t z = svld1_f32(pred, &positions[i * 4 + 2]);
        // svfloat32_t w = svld1_f32(pred, &positions[i * 4 + 3]);
        
        // Matrix multiply (simplified for slide: e.g., scale X by matrix[0][0])
        // svfloat32_t matrix_el = svdup_n_f32_z(pred, matrices[0]); // Example broadcast
        // svfloat32_t result_x = svmul_f32_z(pred, x, matrix_el);
        svfloat32_t scale_val = svdup_n_f32_z(pred, matrices[0]); // Example: broadcast a scale factor
        svfloat32_t result_x = svmul_f32_z(pred, pos_x_vals, scale_val);

        // Store results - automatically handles vector width
        svst1_f32(pred, &output[i], result_x);
        // svst1_f32(pred, &output[i * 4 + 1], result_y);
    }
}
```

---

## SVE2: Same Code, Different Vector Widths

### Compilation Magic: One Source, Multiple Targets
```bash
# Same source code, different vector widths

# 128-bit vectors
clang -march=armv8.2-a+sve2 -msve-vector-bits=128 vertices.c -o vertices_128

# 256-bit vectors
clang -march=armv8.2-a+sve2 -msve-vector-bits=256 vertices.c -o vertices_256

# 512-bit vectors
clang -march=armv8.2-a+sve2 -msve-vector-bits=512 vertices.c -o vertices_512

# Runtime detection
clang -march=armv8.2-a+sve2 -msve-vector-bits=scalable vertices.c -o vertices_vla
```

Compilers target a **conservative** baseline by default (SSE2, often NEON), so you need to explicitly enable newer vector extensions (AVX2, SVE2, etc.) for your target hardware.

---

## Profiling Tools

- **Windows**
  - Superluminal (frame-focused CPU profiler widely used in game dev)
  - Visual Studio Performance Profiler, PIX
  - Windows Performance Recorder + Windows Performance Analyzer (ADK)
  - Intel VTune Profiler, NVIDIA Nsight Systems, AMD uProf, Perfetto
- **macOS & iOS**
  - Xcode Instruments (Time Profiler, Energy, Metal System Trace)
- **Android**
  - Android Studio Profiler (CPU, Memory, Network)
  - Qualcomm Trepn Profiler, GameBench
- **Linux**
  - perf (events), gprof, BPF-based bcc/bpftrace tools
  - Valgrind/Callgrind, Sysprof

---

## Conclusion
- **Key Takeaways** 🎯  
  - **Cache**: Optimize locality with SoA vs AoS patterns
  - **Pipelining**: Avoid stalls with independent instruction chains  
  - **Branches**: Go branchless for predictable performance
  - **Threading**: Use P-cores + E-cores intelligently with job systems
  - **SIMD**: Parallelize with AVX/NEON, future-proof with SVE2
  - **Memory Ordering**: Test lock-free code on both x86 TSO and ARM weak models
- **Critical Warnings** ⚠️  
  - x86 lock-free code may break on ARM devices!
  - Test with speculative execution mitigations enabled
  - Profile on target hardware, not just dev machines
  - Use PGO for performance tuning
  - Consider power efficiency for portable devices

---

## People who inspired this talk 🙏

- **Matt Godbolt** - Compiler Explorer changed how we all learn assembly
- **Casey Muratori** - Handmade Hero, performance-aware programming advocacy
- **Mike Acton** - Data-Oriented Design, the SoA vs AoS
- **Fabian Giesen** - SIMD and optimization deep dives
- **Agner Fog** - CPU optimization manuals and instruction tables
- **Travis Downs** - Modern CPU performance analysis
- **Daniel Lemire** - Branchless techniques and SIMD wizardry
- **Jeff Preshing** - Lock-free programming and memory ordering clarity
...and many more! ❤️

**Resources:**
- [Wookash Podcast](https://wooka.sh) - Łukasz Ściga podcast going deep with guests
- [Compiler Explorer](https://godbolt.org) - See what your code compiles to
- [FFmpeg](https://ffmpeg.org) - Source code is amazing inspiration and newly started assembly lessons

---

# Q&A 🤔

---

## Thank you!

Martin Pernica
🐦 @martindeveloper
📩 martin.pernica@flying-rat.studio

---
