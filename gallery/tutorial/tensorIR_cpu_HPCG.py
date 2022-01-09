import os
import time
import tvm
import math
import numpy as np
from tvm.script import tir as T


@T.prim_func
def GenerateProblem(Aval: T.handle, Aidx: T.handle, B: T.handle, X:T.handle, n: T.int32, nnzinrow: T.int32):
    T.func_attr({"global_symbol": "GenerateProblem", "tir.noalias": True})
    aval = T.match_buffer(Aval, (n*n*n, nnzinrow), dtype="float64")
    aidx = T.match_buffer(Aidx, (n*n*n, nnzinrow), dtype="int32")
    b = T.match_buffer(B, (n*n*n), dtype="float64")
    x = T.match_buffer(X, (n*n*n), dtype="float64")
    for i in range(n*n*n):
        with T.block("outer"):
            vi = T.axis.spatial(n*n*n, i)
            idx = T.alloc_buffer((3), dtype="int32")
            idx[0] = vi // (n * n)
            idx[1] = (vi - idx[0] * (n * n)) // n
            idx[2] = vi % n
            nnz = T.alloc_buffer((1), dtype="int32")
            num_boundry = T.alloc_buffer((1), dtype="int32")
            num_boundry[0] = 0
            nnz[0] = 0
            for bz in range(3):
                for by in range(3):
                    for bx in range(3):
                        with T.block("inner"):
                            vz, vy, vx = T.axis.remap("SSS", [bz, by, bx])
                            cur_col = T.alloc_buffer((1), dtype="int32")
                            cur_col[0] = vi + (bz-1) * (n*n) + (by-1) * n + (bx-1)
                            if cur_col[0] == vi:
                                aval[vi, nnz[0]] = T.float64(26.0)
                                aidx[vi, nnz[0]] = cur_col[0]
                            elif idx[0] + vz == 0 or idx[0] + vz == n+1 or idx[1] + vy == 0 or idx[1] + vy == n+1 or idx[2] + vx == 0 or idx[2] + vx == n+1:
                                aval[vi, nnz[0]] = T.float64(0.0)
                                aidx[vi, nnz[0]] = 0
                                num_boundry[0] += 1 
                            else:
                                aval[vi, nnz[0]] = T.float64(-1.0)
                                aidx[vi, nnz[0]] = cur_col[0]
                            nnz[0] += 1
            b[vi] = T.float64(26.0) - (T.float64(nnz[0] - 1 - num_boundry[0])) 
            x[vi] = T.float64(0)                

@T.prim_func
def CopyVec(From: T.handle, To: T.handle, len: T.int32):
    T.func_attr({"global_symbol": "CopyVec", "tir.noalias": True})
    from_vec = T.match_buffer(From, (len), dtype="float64")
    to_vec = T.match_buffer(To, (len), dtype="float64")
    for i in range(len):
        with T.block("outer"):
            vi = T.axis.spatial(len, i)
            to_vec[vi] = from_vec[vi]

@T.prim_func
def ZeroVec(X: T.handle, len: T.int32):
    T.func_attr({"global_symbol": "ZeroVec", "tir.noalias": True})
    x = T.match_buffer(X, (len), dtype="float64")

    for i in range(len):
         with T.block("outer"):
            vi = T.axis.S(len, i)
            x[vi] = T.float64(0.0)

@T.prim_func
def Waxpby(alpha: T.float64, A: T.handle, beta: T.float64, B: T.handle, W: T.handle, len: T.int32):
    T.func_attr({"global_symbol": "Waxpby", "tir.noalias": True})
    a = T.match_buffer(A, (len), dtype="float64")
    b = T.match_buffer(B, (len), dtype="float64")
    w = T.match_buffer(W, (len), dtype="float64")
    for i in range(len):
        with T.block("outer"):
            vi = T.axis.spatial(len, i)
            w[vi] = alpha * a[vi] + beta * b[vi]

@T.prim_func
def DotProduct(A: T.handle, B: T.handle, Res: T.handle, num_threads: T.int32, len: T.int32):
    T.func_attr({"global_symbol": "DotProduct", "tir.noalias": True})
    a = T.match_buffer(A, (len), dtype="float64")
    b = T.match_buffer(B, (len), dtype="float64")
    r = T.match_buffer(Res, (1), dtype="float64")
    res = T.alloc_buffer((num_threads), dtype="float64")

    for t in range(num_threads):
        with T.block("outer"):
            vt =  T.axis.S(num_threads, t) 
            tmp = T.alloc_buffer((1), dtype="float64")
            tmp[0] = T.float64(0)
            for i in range(len/num_threads):
                with T.block("inner"):
                    vi =  T.axis.S(len/num_threads, i) 
                    tmp[0] += a[vt * len/num_threads + vi] * b[vt * len/num_threads + vi]
            res[vt] = tmp[0]

    r[0] =  T.float64(0)
    for j in range(num_threads):
        r[0] += res[j]

@T.prim_func
def DotProduct_rfactor(A: T.handle, B: T.handle, Res: T.handle, len: T.int32):
    T.func_attr({"global_symbol": "DotProduct_rfactor", "tir.noalias": True})
    a = T.match_buffer(A, (len), dtype="float64")
    b = T.match_buffer(B, (len), dtype="float64")
    r = T.match_buffer(Res, (1), dtype="float64")

    for i in range(len):
         with T.block("outer"):
            vi = T.axis.reduce(len, i)
            with T.init():
                r[0] = T.float64(0.0)
            r[0] += a[vi] * b[vi]

@T.prim_func
def Spmv(Aval: T.handle, Aidx: T.handle, X: T.handle, Y: T.handle, n: T.int32, nnzinrow: T.int32):
    T.func_attr({"global_symbol": "Spmv", "tir.noalias": True})
    aval = T.match_buffer(Aval, (n*n*n, nnzinrow), dtype="float64")
    aidx = T.match_buffer(Aidx, (n*n*n, nnzinrow), dtype="int32")
    x = T.match_buffer(X, (n*n*n), dtype="float64")
    y = T.match_buffer(Y, (n*n*n), dtype="float64")
    for i in range(n*n*n):
        with T.block("outer"):
            vi = T.axis.spatial(n*n*n, i)
            sum = T.alloc_buffer((1), dtype="float64")
            sum[0] = T.float64(0)
            for j in range(nnzinrow):
                with T.block("inner"):
                    vj = T.axis.spatial(nnzinrow, j)
                    sum[0] += aval[vi,vj] * x[aidx[vi,vj]]
            y[vi] = sum[0]

@T.prim_func
def Symgs(Aval: T.handle, Aidx: T.handle, X: T.handle, R: T.handle, n: T.int32, nnzinrow: T.int32):
    T.func_attr({"global_symbol": "Symgs", "tir.noalias": True})
    aval = T.match_buffer(Aval, (n*n*n, nnzinrow), dtype="float64")
    aidx = T.match_buffer(Aidx, (n*n*n, nnzinrow), dtype="int32")
    x = T.match_buffer(X, (n*n*n), dtype="float64")
    r = T.match_buffer(R, (n*n*n), dtype="float64")
    for i in range(n*n*n):
        with T.block("forward_outer"):
            vi = T.axis.spatial(n*n*n, i)
            sum = T.alloc_buffer((1), dtype="float64")
            sum[0] = r[vi]
            for j in range(nnzinrow):
                with T.block("forward_inner"):
                    vj = T.axis.spatial(nnzinrow, j)
                    sum[0] -= aval[vi,vj] * x[aidx[vi,vj]]
            sum[0] += aval[vi, nnzinrow/2] * x[vi]
            x[vi] = sum[0] / aval[vi, nnzinrow/2]

    for i in range(n*n*n):
        with T.block("backward_outer"):
            vi = T.axis.spatial(n*n*n, n*n*n - 1 - i)
            sum1 = T.alloc_buffer((1), dtype="float64")
            sum1[0] = r[vi]
            for j in range(nnzinrow):
                with T.block("backward_inner"):
                    vj = T.axis.spatial(nnzinrow, j)
                    sum1[0] -= aval[vi,vj] * x[aidx[vi,vj]]
            sum1[0] += aval[vi, nnzinrow/2] * x[vi]
            x[vi] = sum1[0] / aval[vi, nnzinrow/2]


@T.prim_func
def Symgs_ls(Aval: T.handle, Aidx: T.handle, X: T.handle, R: T.handle, n: T.int32, nnzinrow: T.int32):
    T.func_attr({"global_symbol": "Symgs", "tir.noalias": True})
    aval = T.match_buffer(Aval, (n*n*n, nnzinrow), dtype="float64")
    aidx = T.match_buffer(Aidx, (n*n*n, nnzinrow), dtype="int32")
    x = T.match_buffer(X, (n*n*n), dtype="float64")
    r = T.match_buffer(R, (n*n*n), dtype="float64")
    for i in range(n*n*n):
        with T.block("forward_outer"):
            vi = T.axis.spatial(n*n*n, i)
            sum = T.alloc_buffer((1), dtype="float64")
            sum[0] = r[vi]
            for j in range(nnzinrow):
                with T.block("forward_inner"):
                    vj = T.axis.spatial(nnzinrow, j)
                    sum[0] -= aval[vi,vj] * x[aidx[vi,vj]]
            sum[0] += aval[vi, nnzinrow/2] * x[vi]
            x[vi] = sum[0] / aval[vi, nnzinrow/2]

    for i in range(n*n*n):
        with T.block("backward_outer"):
            vi = T.axis.spatial(n*n*n, i)
            sum1 = T.alloc_buffer((1), dtype="float64")
            sum1[0] = r[vi]
            for j in range(nnzinrow):
                with T.block("backward_inner"):
                    vj = T.axis.spatial(nnzinrow, j)
                    sum1[0] -= aval[vi,vj] * x[aidx[vi,vj]]
            sum1[0] += aval[vi, nnzinrow/2] * x[vi]
            x[vi] = sum1[0] / aval[vi, nnzinrow/2]

@T.prim_func
def Restriction(Rc: T.handle, R: T.handle, Axf: T.handle, nc: T.int32):
    T.func_attr({"global_symbol": "Restriction", "tir.noalias": True})
    rc = T.match_buffer(Rc, (nc*nc*nc), dtype="float64")
    r = T.match_buffer(R, (nc*2*nc*2*nc*2), dtype="float64")
    axf = T.match_buffer(Axf, (nc*2*nc*2*nc*2), dtype="float64")

    for i,j,k in T.grid(nc,nc,nc):
        with T.block("outer"):
            vi,vj,vk = T.axis.remap("SSS",[i,j,k])
            rc[vi*nc*nc + vj*nc + vk] = r[(2*vi)*(nc*2)*(nc*2)+(2*vj)*(nc*2)+(2*vk)] - axf[(2*vi)*(nc*2)*(nc*2)+(2*vj)*(nc*2)+(2*vk)]

@T.prim_func
def Prolongation(Xf: T.handle, Xc: T.handle, nc: T.int32):
    T.func_attr({"global_symbol": "Prolongation", "tir.noalias": True})
    xf = T.match_buffer(Xf, (nc*2*nc*2*nc*2), dtype="float64")
    xc = T.match_buffer(Xc, (nc*nc*nc), dtype="float64")

    for i,j,k in T.grid(nc,nc,nc):
        with T.block("outer"):
            vi,vj,vk = T.axis.remap("SSS",[i,j,k])
            vc = T.axis.S(nc*nc*nc, (vi)*nc*nc+(vj)*nc+(vk))
            vf = T.axis.S(nc*2*nc*2*nc*2, (vi*2)*(nc*2)*(nc*2)+(vj*2)*(nc*2)+(vk*2))
            xf[vf] = xf[vf] + xc[vc]

class HPCG_Example():
    def __init__(self, dim=3, N=256, n_mg_levels=4, real="float64", ctx = tvm.cpu()):
        # grid parameters 
        self.use_multigrid = True
        self.ls_opt = True
        self.ctx = ctx
        self.num_threads = 32

        # only apply 3-dim cube problem temporarily
        self.N = N
        self.n_mg_levels = n_mg_levels
        self.dim = dim
        self.real = real
        self.nrow = N**dim

        self.nnzinrow = 27

        self.N_ext = self.N // 2  # number of ext cells set so that that total grid size is still power of 2
        self.N_tot = 2 * self.N
        self.pre_and_post_smoothing = 2
        self.bottom_smoothing = 50

        self.b = tvm.nd.empty((self.N**3,), dtype=self.real, device=self.ctx)
        self.x = tvm.nd.empty((self.N**3,), dtype=self.real, device=self.ctx)

        self.r = tvm.nd.empty((self.N**3,), dtype=self.real, device=self.ctx)
        self.z = tvm.nd.empty((self.N**3,), dtype=self.real, device=self.ctx)
        self.p = tvm.nd.empty((self.N**3,), dtype=self.real, device=self.ctx)
        self.Ap= tvm.nd.empty((self.N**3,), dtype=self.real, device=self.ctx)


        self.Aval = []
        self.Aidx = []
        self.rc = []
        self.xc = []
        self.Axf = []
        for level in range(self.n_mg_levels):
            self.Aval.append(tvm.nd.empty(((self.N//(2**level))**3, self.nnzinrow), dtype=self.real, device=self.ctx))
            self.Aidx.append(tvm.nd.empty(((self.N//(2**level))**3, self.nnzinrow), dtype="int32", device=self.ctx))
            if level > 0:
               self.rc.append(tvm.nd.empty(((self.N//(2**level))**3, ), dtype=self.real, device=self.ctx)) 
               self.xc.append(tvm.nd.empty(((self.N//(2**level))**3, ), dtype=self.real, device=self.ctx)) 
            if level != (self.n_mg_levels-1) :
               self.Axf.append(tvm.nd.empty(((self.N//(2**level))**3, ), dtype=self.real, device=self.ctx)) 


        self.stencil_list = []
        self.level_number = 0
        self.level_num = tvm.nd.NDArray
        self.level_idx = tvm.nd.empty((self.nrow,), dtype="int32", device=self.ctx)
        self.symgs_ls_mod = tvm.runtime.Module

        self.generate_problem_mod = tvm.runtime.Module
        self.copy_vec_mod = tvm.runtime.Module
        self.zero_vec_mod = tvm.runtime.Module
        self.dot_mod = tvm.runtime.Module
        self.waxpby_mod = tvm.runtime.Module
        self.spmv_mod = tvm.runtime.Module
        self.symgs_mod = tvm.runtime.Module
        self.restric_mod = tvm.runtime.Module
        self.prolongation_mod = tvm.runtime.Module


    def generate_problem(self, Aval, Aidx, B, X, n):
        self.generate_problem_mod(Aval, Aidx, B, X, n, self.nnzinrow)

    def copy(self, from_vec, to_vec):
        self.copy_vec_mod(from_vec, to_vec, self.nrow)
    
    def zero_vec(self, vec, level):
        self.zero_vec_mod(vec, (self.N//(2**level))**self.dim)

    def waxpby(self, alpha, x, beta, y, w):
        self.waxpby_mod(alpha, x, beta, y, w, self.nrow)

    def dot_rfactor(self, x, y):
        res = tvm.nd.empty((1,), dtype=self.real, device=self.ctx)
        self.dot_mod(x, y, res, self.nrow)
        return res.asnumpy()[0]
    
    def spmv(self, Aval, Aidx, x, y, level):
        self.spmv_mod(Aval, Aidx, x, y, self.N//(2**level), self.nnzinrow)

    def symgs(self, Aval, Aidx, x, r, level):
        self.symgs_mod(Aval, Aidx, x, r, self.N//(2**level), self.nnzinrow)

    def restrication(self, rc, rf, Axf, level):
        self.restric_mod(rc, rf, Axf, self.N//(2**level)//2)

    def prolongation(self, xf, xc, level):
        self.prolongation_mod(xf, xc, self.N//(2**level)//2)

    def MG(self, r, x):
        rf = r 
        xf = x
        for level in range(self.n_mg_levels-1):
            if level != 0 :
                rf = self.rc[level-1]
                xf = self.xc[level-1]
            self.zero_vec(xf, level)
            if self.ls_opt and level == 0:
                self.symgs_ls_mod(self.Aval[level], self.Aidx[level], xf, rf, self.N, self.nnzinrow, self.level_num, self.level_idx)
            # else:
            #     self.symgs(self.Aval[level], self.Aidx[level], xf, rf, level)
            self.spmv(self.Aval[level], self.Aidx[level], xf, self.Axf[level], level)
            self.restrication(self.rc[level], rf, self.Axf[level], level)
            
        xf = self.xc[-1]
        rf = self.rc[-1]
        bot = self.n_mg_levels-1
        self.zero_vec(xf, bot)
        # self.symgs(self.Aval[bot], self.Aidx[bot], xf, rf, bot)

        for level in reversed(range(self.n_mg_levels-1)):
            if level != 0 :
                rf = self.rc[level-1]
                xf = self.xc[level-1]
            else :
                rf = r
                xf = x
            self.prolongation(xf, self.xc[level], level)
            if self.ls_opt and level == 0:
                self.symgs_ls_mod(self.Aval[level], self.Aidx[level], xf, rf, self.N, self.nnzinrow, self.level_num, self.level_idx)
            # else:
            #     self.symgs(self.Aval[level], self.Aidx[level], xf, rf, level)

    def init(self):
        for level in range(self.n_mg_levels):
            if level == 0 :
                self.generate_problem(self.Aval[level], self.Aidx[level], self.b, self.x, self.N)
            else:
                tmp = tvm.nd.empty(((self.N//(2**level))**3,), dtype=self.real, device=self.ctx)
                self.generate_problem(self.Aval[level], self.Aidx[level], tmp, tmp, self.N//(2**level))

    def slove(self,
              max_iters=50,
              eps=1e-12,
              abs_tol=1e-12,
              rel_tol=1e-12,
              verbose=False):
        t = time.time()

        normr = 0.0
        rtz = 0.0
        oldrtz = 0.0
        iter = 0
        
        self.copy(self.x, self.p)
        self.spmv(self.Aval[0], self.Aidx[0], self.p, self.Ap, 0)
        self.waxpby(1.0, self.b, -1.0, self.Ap, self.r)
        normr = math.sqrt(self.dot_rfactor(self.r, self.r))
        normr0 = normr

        while iter < max_iters and normr/normr0 > abs_tol:
            if self.use_multigrid == False:
                self.copy(self.r, self.z)
            else:
                self.MG(self.r, self.z)
            
            if iter == 0:
                self.copy(self.z, self.p)
                rtz = self.dot_rfactor(self.r, self.z)
            else:
                oldrtz = rtz
                rtz = self.dot_rfactor(self.r, self.z)
                beta = rtz/oldrtz
                self.waxpby(1.0, self.z, beta, self.p, self.p)

            self.spmv(self.Aval[0], self.Aidx[0], self.p, self.Ap, 0)  
            alpha = rtz / self.dot_rfactor(self.p, self.Ap) 
            self.waxpby(1.0, self.x, alpha, self.p, self.x)
            self.waxpby(1.0, self.r, -alpha, self.Ap, self.r)
            normr = math.sqrt(self.dot_rfactor(self.r, self.r))

            iter += 1
            print("iter",iter, " norm=", normr/normr0, "used time=", f'{time.time() - t:.3f} s')

    def generate_module(self):
        # GenerateProblem
        sch = tvm.tir.Schedule(GenerateProblem)
        i, = sch.get_loops(sch.get_block("outer"))
        spl_idx = sch.split(i, factors=[self.num_threads,None])
        sch.parallel(spl_idx[0])
        bz,by,bx = sch.get_loops(sch.get_block("inner"))
        sch.unroll(bz)
        sch.unroll(by)
        sch.unroll(bx)

        # print(sch.mod.script()) 
        # mod = tvm.build(sch.mod, target="c --unpacked-api")
        self.generate_problem_mod = tvm.build(sch.mod, target="llvm -opt-level=3")
        # print(mod.get_source())
        # CopyVec
        sch = tvm.tir.Schedule(CopyVec)
        i, = sch.get_loops(sch.get_block("outer"))
        spl_idx = sch.split(i, factors=[self.num_threads, None])
        sch.parallel(spl_idx[0])
        self.copy_vec_mod = tvm.build(sch.mod, target="llvm -opt-level=3")

        # ZeroVec
        sch = tvm.tir.Schedule(ZeroVec)

        i, = sch.get_loops(sch.get_block("outer"))
        spl_idx = sch.split(i, factors=[self.num_threads, None])
        sch.parallel(spl_idx[0])
        self.zero_vec_mod = tvm.build(sch.mod, target="llvm -opt-level=3")

        # DotProduct
        sch = tvm.tir.Schedule(DotProduct_rfactor)

        i, = sch.get_loops(sch.get_block("outer"))
        i_out, i_in  = sch.split(i, factors=[self.num_threads,None])
        rf_block = sch.rfactor(i_out, 0)
        ii, jj = sch.get_loops(rf_block)
        sch.parallel(ii)

        self.dot_mod = tvm.build(sch.mod, target="llvm -opt-level=3")

        # Waxpby
        sch = tvm.tir.Schedule(Waxpby)

        i, = sch.get_loops(sch.get_block("outer"))
        spl_idx = sch.split(i, factors=[self.num_threads,None])
        sch.parallel(spl_idx[0])

        self.waxpby_mod = tvm.build(sch.mod, target="llvm -opt-level=3")

        # Spmv
        sch = tvm.tir.Schedule(Spmv)

        i, = sch.get_loops(sch.get_block("outer"))
        spl_idx = sch.split(i, factors=[self.num_threads,None])
        sch.parallel(spl_idx[0])

        self.spmv_mod = tvm.build(sch.mod, target="llvm -opt-level=3")

        # Symgs
        sch = tvm.tir.Schedule(Symgs)

        self.symgs_mod = tvm.build(sch.mod, target="llvm -opt-level=3")

        # Restrication
        sch = tvm.tir.Schedule(Restriction)

        i,j,k = sch.get_loops(sch.get_block("outer"))
        fuse = sch.fuse(i,j,k)
        spl_idx = sch.split(fuse, factors=[self.num_threads,None])
        sch.parallel(spl_idx[0])

        self.restric_mod = tvm.build(sch.mod, target="llvm -opt-level=3")

        # Prolongation
        sch = tvm.tir.Schedule(Prolongation)

        i,j,k = sch.get_loops(sch.get_block("outer"))
        fuse = sch.fuse(i,j,k)
        spl_idx = sch.split(fuse, factors=[self.num_threads,None])
        sch.parallel(spl_idx[0], force = 1)
        self.prolongation_mod = tvm.build(sch.mod, target="llvm -opt-level=3")

    def level_schedule_analysis(self):
        if os.path.exists("level_num"+str(self.N)+".npy") and os.path.exists("level_idx"+str(self.N)+".npy"):
            self.level_idx = tvm.nd.array(np.load("level_idx"+str(self.N)+".npy"), self.ctx)
            self.level_num = tvm.nd.array(np.load("level_num"+str(self.N)+".npy"), self.ctx)
            self.level_number = len(np.load("level_num"+str(self.N)+".npy")) - 1
        else :
            for i in range(-1,2,1):
                for j in range(-1,2,1):
                    for k in range(-1,2,1):
                        self.stencil_list.append((i,j,k))
            print(self.stencil_list)
            level_idx = np.zeros((self.nrow,2), dtype="int32")
            level_idx[:,0] = np.arange(self.nrow) 
            t0 = time.time()
            for i in range(self.N):
                for j in range(self.N):
                    for k in range(self.N):
                        cur_idx =  i * (self.N ** 2) + j * self.N + k
                        level_idx[cur_idx][1] = 0
                        max_level = -1
                        for n in range(len(self.stencil_list)):
                            bi = self.stencil_list[n][0]
                            bj = self.stencil_list[n][1]
                            bk = self.stencil_list[n][2]
                            if  i+bi >=0 and i+bi < self.N and j + bj >=0 and j + bj < self.N and k + bk >=0 and k + bk < self.N:
                                idx = (i+bi) * (self.N ** 2) + (j + bj) * self.N + k + bk
                                if idx < cur_idx and level_idx[idx][1] > max_level:
                                    max_level =  level_idx[idx][1]
                        level_idx[cur_idx][1] = max_level + 1
            self.level_number = level_idx[-1][1] + 1
            level_num = np.zeros((self.level_number+1), dtype="int32")
            sort = level_idx[level_idx[:,1].argsort()]
            cur_idx = 1
            for i in range(1, self.nrow):
                if(sort[i-1][1] != sort[i][1]):
                    level_num[cur_idx] = i 
                    cur_idx += 1
            level_num[cur_idx] = self.nrow 
            
            self.level_idx = tvm.nd.array(sort[:,0].astype("int32"), self.ctx)
            self.level_num = tvm.nd.array(level_num.astype("int32"), self.ctx)
            np.save("level_num"+str(self.N)+".npy", level_num.astype("int32"))
            np.save("level_idx"+str(self.N)+".npy", sort[:,0].astype("int32"))

        # level schedule primitive
        append_list = [tvm.tir.decl_buffer((self.level_number+1), name="level_num", dtype="int32"),  tvm.tir.decl_buffer((self.nrow), name="level_idx", dtype="int32")]
        func = Symgs_ls.append_params(append_list)
        sch = tvm.tir.Schedule(func)
        loop, = sch.get_loops(sch.get_block("forward_outer"))
        sch.level_schedule(loop, self.level_number, append_list[0], append_list[1])
        level, level_num  = sch.get_loops(sch.get_block("forward_outer"))
        sch.parallel(level_num, force = 1)
        loop, = sch.get_loops(sch.get_block("backward_outer"))
        sch.level_schedule(loop, -self.level_number, append_list[0], append_list[1]) # negative number for reversed loop
        level, level_num  = sch.get_loops(sch.get_block("backward_outer"))
        sch.parallel(level_num, force = 1)
        print(sch.mod.script()) 
        
        # mod = tvm.build(sch.mod, target="c --unpacked-api")
        # print(mod.get_source())
        self.symgs_ls_mod = tvm.build(sch.mod, target="llvm -opt-level=3")

    def run(self, verbose=False):
        t0 = time.time()
        self.generate_module()
        print(f'Generate_module time: {time.time() - t0:.3f} s')

        t1 = time.time()
        self.init()
        if self.ls_opt:
            self.level_schedule_analysis()
            print(f'Init and level schedule analysis time: {time.time() - t1:.3f} s')

        t2 = time.time()
        self.slove(max_iters=50, verbose=verbose)
        print(f'Slove time: {time.time() - t2:.3f} s')

if __name__ == '__main__':
    solver = HPCG_Example(dim=3, N=128)
    t = time.time()
    solver.run(verbose=True)
    print(f'Total solver time: {time.time() - t:.3f} s')



