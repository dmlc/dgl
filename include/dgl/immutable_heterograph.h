
namespace dgl {

/*
 * TODO: replace with ImmutableBiGraph after merging #573
 */
struct ImmutableBiGraph {
  int dummy;
};

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
  ImmutableHeteroGraph(const ImmutableHeteroGraph& other) = default;

#ifndef _MSC_VER
  /*! \brief default move constructor */
  ImmutableHeteroGraph(ImmutableHeteroGraph&& other) = default;
#else
  ImmutableHeteroGraph(ImmutableHeteroGraph&& other) {
    this->metagraph_ = other.metagraph_;
    this->relations_ = std::move(other.relations_);
  }
#endif

  /*! \brief default assign constructor */
  ImmutableHeteroGraph& operator=(const ImmutableHeteroGraph& other) = default;

  /*! \brief default destructor */
  ~ImmutableHeteroGraph() = default;

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
  std::vector<ImmutableBiGraph> relations_;
};

};  // namespace dgl
