
namespace dgl {

class ImmutableHeteroGraph : public HeteroGraphInterface {
 public:
  /*!
   * \brief Construct an immutable heterogeneous graph from a metagraph
   * and a vector of homogeneous/bipartite graphs
   *
   * \param metagraph The immutable metagraph
   * \param relations A vector of immutable homogeneous/bipartite graphs
   */
  explicit ImmutableHeteroGraph(
      ImmutableGraph metagraph,
      std::vector<ImmutableGraph> relations);

  /*! \brief default copy constructor */
  ImmutableGraph(const ImmutableGraph& other) = default;

#ifndef _MSC_VER
  /*! \brief default move constructor */
  ImmutableGraph(ImmutableGraph&& other) = default;
#else
  ImmutableGraph(ImmutableGraph&& other) {
    this->metagraph_ = other.metagraph_;
    this->relations_ = std::move(other.relations_);
  }
#endif

  /*! \brief default assign constructor */
  ImmutableGraph& operator=(const ImmutableGraph& other) = default;

  /*! \brief default destructor */
  ~ImmutableGraph() = default;

 private:
  /*!
   * \brief the metagraph of the heterogeneous graph
   *
   * The nodes and edges of the metagraph maps to the node types
   * and edge types of the heterogeneous graph.
   */
  ImmutableGraph metagraph_;

  /*
   * A list of immutable graphs, either homogeneous or bipartite.
   */
  std::vector<ImmutableGraph> relations_;
};

};  // namespace dgl
