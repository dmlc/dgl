import torch as th
import dgl
import torch.nn as nn
from dgl.nn.functional import edge_softmax
import time
import numpy as np
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
# time.sleep(10)
# n = 2703
# nnz = 5429
# rows = np.random.randint(0, n, nnz)
# cols = np.random.randint(0, n, nnz)

# values = th.randn((nnz,3)).requires_grad_(True)

# print(values)
# values = th.unsqueeze(values, -1)
# g = dgl.graph((cols, rows))
# g.edata['e'] = values
# edata = g.edata['e']


data = CoraGraphDataset()
g = data[0]
n_edges = data.graph.number_of_edges()
values = th.randn((n_edges,1000)).requires_grad_(True)
g.edata['e'] = values
edata = g.edata['e']
# c_sparse = dgl.ops.edge_softmax(g, values)
# t3 = c_sparse
# print(t3)
# g = dgl.graph((th.tensor([0, 0, 0, 1, 1, 2]), th.tensor([0, 1, 2, 1, 2, 2])))
# g.create_formats_()
# g = g.formats(['csc'])
# edata = th.tensor([[1,2,4], [2,4,8], [3,6,9], [4,8,16], [5,10,20], [6,12,24]],dtype=th.float,requires_grad=True).float()
# edata = th.tensor([[1,2], [2,4], [3,6], [4,8], [5,10], [6,12]], dtype=th.float,requires_grad=True).float()
# edata = th.tensor([[1,1], [1,2], [1,3], [1,4], [1,5], [1,6]],dtype=th.float,requires_grad=True).float()
# edata = th.tensor([2,4,6,8,10,12]).float()
# edata = th.tensor([[1], [2], [3], [4], [5], [6]],dtype=th.float,requires_grad=True).float()
# c = time.time()
class Model(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,g, ndata):
        return edge_softmax(g, ndata)
loss_fcn = nn.CrossEntropyLoss()
m = Model()
a = m(g, edata)
# a = edge_softmax(g, edata)
# print(a)
loss = loss_fcn(a, edata)
loss.backward()

# print(time.time() - c)
# print(a)
# // NDArray test;
# // std::cout<<"SPMM"<<std::endl;
# // std::cout<<"element of op is:"<<op<<std::endl;
# // std::cout<<"element of reduce_op is:"<<reduce_op<<std::endl;
# // std::cout<<"element of U is:"<<U.NumElements()<<std::endl;
# // if(U.NumElements() != 0){
# //     for (int64_t i = 0; i < U.NumElements(); ++i) {
# //        std::cout<< U.Ptr<float>()[i] << ", ";
# //     }
# //     std::cout<<std::endl;
# // }
# // std::cout<<"shape of E is:"<<*(E->shape)<<std::endl;
# // std::cout<<"element of E is:"<<E.NumElements()<<std::endl;
# // if(E.NumElements() != 0){
# //     for (int64_t i = 0; i < E.NumElements(); ++i) {
# //        std::cout<< E.Ptr<float>()[i] << ", ";
# //     }
# //     std::cout<<std::endl;
# // }
# // std::cout<<"element of V is:"<<V.NumElements()<<std::endl;
# // if(V.NumElements() != 0){
# //     for (int64_t i = 0; i < V.NumElements(); ++i) {
# //        std::cout<< V.Ptr<float>()[i] << ", ";
# //     }
# //     std::cout<<std::endl;
# // }
# // std::cout<<"element of ArgU is:"<<ArgU.NumElements()<<std::endl;
# // if(ArgU.NumElements() != 0){
# //     for (int64_t i = 0; i < ArgU.NumElements(); ++i) {
# //        std::cout<< ArgU.Ptr<float>()[i] << ", ";
# //     }
# //     std::cout<<std::endl;
# // }
# // std::cout<<"element of ArgE is:"<<ArgE.NumElements()<<std::endl;
# // if(ArgE.NumElements() != 0){
# //     for (int64_t i = 0; i < ArgE.NumElements(); ++i) {
# //        std::cout<< ArgE.Ptr<float>()[i] << ", ";
# //     }
# //     std::cout<<std::endl;
# // }

# // std::cout<<"element of test is:"<<test<<std::endl;
# // for (int64_t i = 0; i < V.NumElements(); ++i) {
# //     std::cout<< V.Ptr<float>()[i] << ", ";
# //     //temp.Ptr<float>()[i] = i;
# // }
# // // std::cout<<args[3]<<std::endl;
# // std::cout<<"ttt"<<std::endl;
# // // // std::cout<<V->shape<<std::endl;

# // std::vector<int64_t> v = {graph->NumVertices(dst_vtype),E->ndim};
# // NDArray V_max = NDArray::Empty(v,V->dtype,DLContext{kDLCPU, 0});
# // for (int64_t i = 0; i < V_max.NumElements(); ++i) {
# //     V_max.Ptr<float>()[i] = 0;
# // }
# //SpMM(op, reduce_op, graph.sptr(), U, E, V_max, {ArgU, ArgE});
# // std::cout<<"right spmm is:"<<std::endl;
# // for (int64_t i = 0; i < V.NumElements(); ++i) {
# //      std::cout<< V.Ptr<float>()[i] << ", ";
# //     }
# // std::cout<<std::endl;

# // for (int64_t i = 0; i < V.NumElements(); ++i) {
# //     std::cout<< V_max.Ptr<float>()[i] << ", ";
# //     //temp.Ptr<float>()[i] = i;
# // }
# // std::cout<<std::endl;

# // std::vector<int64_t> v = {3,2};
# // NDArray temp = NDArray::Empty(v,V->dtype,DLContext{kDLCPU, 0});
# // for (int64_t i = 0; i < temp.NumElements(); ++i) {
# //     std::cout<< temp.Ptr<float>()[i] << ", ";
# //     temp.Ptr<float>()[i] = i;
# //     std::cout<<temp.Ptr<float>()[i]<<", ";
# // }
# // std::cout<<std::endl;
# //temp.Empty()
# // std::cout<<"right sddmm is:"<<std::endl;
# // for (int64_t i = 0; i < out.NumElements(); ++i) {
# //      std::cout<< out.Ptr<float>()[i] << ", ";
# //     }
# // std::cout<<std::endl;
# // const std::string temp = "sub";
# // if(op == temp){
# //   //std::cout<<"hello"<<std::endl;
# //   // std::cout<<"hello"<<std::endl;
# //   // std::cout<<out->dtype<<std::endl;
# //   // std::cout<<out.NumElements()<<std::endl;
# //   // NDArray a = out.CopyTo(DLContext{kDLCPU, 0});
# //   for (int64_t i = 0; i < out.NumElements(); ++i) {
# //     out.Ptr<float>()[i] = std::exp(out.Ptr<float>()[i]);
# //   }
# //   // std::ostringstream oss;
# //   // NDArray shuchu = a.CopyTo(DLContext{kDLCPU, 0});
# //   // oss << "array([";
# //   // for (int64_t i = 0; i < shuchu.NumElements(); ++i) {
# //   //     oss << shuchu.Ptr<float>()[i] << ", ";
# //   // }
# //   // std::cout<<oss.str()<<std::endl;

# // }
# //std::cout<<"op is:"<<op<<std::endl;
# // std::cout<<"SDDMM"<<std::endl;
# // std::cout<<"element of op is:"<<op<<std::endl;
# // std::cout<<"element of lhs is:"<<lhs.NumElements()<<std::endl;
# // if(lhs.NumElements() != 0){
# //     for (int64_t i = 0; i < lhs.NumElements(); ++i) {
# //        std::cout<< lhs.Ptr<float>()[i] << ", ";
# //     }
# //     std::cout<<std::endl;
# // }
# // std::cout<<"element of rhs is:"<<rhs.NumElements()<<std::endl;
# // if(rhs.NumElements() != 0){
# //     for (int64_t i = 0; i < rhs.NumElements(); ++i) {
# //        std::cout<< rhs.Ptr<float>()[i] << ", ";
# //     }
# //     std::cout<<std::endl;
# // }
# // std::cout<<"element of out is:"<<out.NumElements()<<std::endl;
# // if(out.NumElements() != 0){
# //     for (int64_t i = 0; i < out.NumElements(); ++i) {
# //        std::cout<< out.Ptr<float>()[i] << ", ";
# //     }
# //     std::cout<<std::endl;
# // }
# // std::cout<<"value of lhs_target is:"<<lhs_target<<std::endl;
# // std::cout<<"value of rhs_target is:"<<rhs_target<<std::endl;