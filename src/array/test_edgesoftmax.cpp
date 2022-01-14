DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelEdgeSoftmax")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef graph = args[0];
    NDArray E = args[1];
    NDArray None_test = args[2];

    // string  op;
    const std::string reduce_op_max = "max";
    const std::string op_sub = "sub";
    const std::string copy_rhs_op = "copy_rhs";
    const std::string reduce_op_sum = "sum";
    const std::string op_div = "div";
    // int  op;
    int lhs_target  = 1;
    int rhs_target  = 2;

    // spmm_max data
    NDArray U_max = None_test.CopyTo(DLContext{kDLCPU, 0});
    auto pair = graph->meta_graph()->FindEdge(0);  // only one etype in the graph.
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    std::vector<int64_t> v;
    // std::cout<<"E shape is"<<*(E->shape)<<std::endl;
    // std::cout<<"E dim is"<<E->ndim<<std::endl;
    // std::cout<<"E strides is"<<*(E->strides)<<std::endl;
    if(*(E->strides)== 1){
        v = {graph->NumVertices(dst_vtype)};
    }else{
        v = {graph->NumVertices(dst_vtype),*(E->strides)};
    }
    NDArray V_max = NDArray::Empty(v,E->dtype,DLContext{kDLCPU, 0});
    for (int64_t i = 0; i < V_max.NumElements(); ++i) {
        V_max.Ptr<float>()[i] = 0;
    }
    NDArray ArgU_max = None_test.CopyTo(DLContext{kDLCPU, 0});
    std::vector<int64_t> e_shape = {*(E->shape)};
    NDArray ArgE_max = NDArray::Empty(e_shape,ArgU_max->dtype,DLContext{kDLCPU, 0});
    for (int64_t i = 0; i < ArgE_max.NumElements(); ++i) {
        ArgE_max.Ptr<float>()[i] = 0;
    }
    
    // spmm_max 
    SpMM(copy_rhs_op, reduce_op_max, graph.sptr(), U_max, E, V_max, {ArgU_max, ArgE_max}); 
    std::cout<<"my spmm_sub result is:"<<std::endl;
    for (int64_t i = 0; i < V_max.NumElements(); ++i) {
           std::cout<< V_max.Ptr<float>()[i] << ", ";
    }
    std::cout<<std::endl;
    
    // sddmm_sub data
    NDArray out_sub = E.CopyTo(DLContext{kDLCPU, 0});
    for (int64_t i = 0; i < out_sub.NumElements(); ++i) {
        out_sub.Ptr<float>()[i] = 0;
    }

    // sddmm_sub
    SDDMM(op_sub, graph.sptr(), E, V_max, out_sub, lhs_target, rhs_target);

    // exp
    for (int64_t i = 0; i < out_sub.NumElements(); ++i) {
        out_sub.Ptr<float>()[i] = std::exp(out_sub.Ptr<float>()[i]);
    }
    std::cout<<"my sddmm_sub with exp result is:"<<std::endl;
    for (int64_t i = 0; i < out_sub.NumElements(); ++i) {
           std::cout<< out_sub.Ptr<float>()[i] << ", ";
    }
    std::cout<<std::endl;
    
    //spmm_sum data
    NDArray U_sum = None_test.CopyTo(DLContext{kDLCPU, 0});
    std::vector<int64_t> V_sum_shape;
    if(*(E->strides)== 1){
         V_sum_shape = {graph->NumVertices(dst_vtype)};
    }else{
         V_sum_shape = {graph->NumVertices(dst_vtype),*(E->strides)};
    }
    NDArray V_sum = NDArray::Empty(V_sum_shape,E->dtype,DLContext{kDLCPU, 0});
    for (int64_t i = 0; i < V_sum.NumElements(); ++i) {
        V_sum.Ptr<float>()[i] = 0;
    }
    NDArray ArgU_sum =  None_test.CopyTo(DLContext{kDLCPU, 0});
    NDArray ArgE_sum =  None_test.CopyTo(DLContext{kDLCPU, 0});

    
    // spmm_sum 
    SpMM(copy_rhs_op, reduce_op_sum, graph.sptr(), U_sum, out_sub, V_sum, {ArgU_sum, ArgE_sum});
    std::cout<<"my spmm_sum result is:"<<std::endl;
    for (int64_t i = 0; i < V_sum.NumElements(); ++i) {
           std::cout<< V_sum.Ptr<float>()[i] << ", ";
    }
    std::cout<<std::endl;

    //sddmm_div data
    NDArray out_div = E.CopyTo(DLContext{kDLCPU, 0});
    for (int64_t i = 0; i < out_div.NumElements(); ++i) {
        out_div.Ptr<float>()[i] = 0;
    }
    
    //sddmm_div 
    SDDMM(op_div, graph.sptr(), out_sub, V_sum, out_div, lhs_target, rhs_target);

    std::cout<<"my sddmm_div result is:"<<std::endl;
    for (int64_t i = 0; i < out_div.NumElements(); ++i) {
           std::cout<< out_div.Ptr<float>()[i] << ", ";
        }
    std::cout<<std::endl;
});