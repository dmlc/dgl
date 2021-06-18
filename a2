#define NUM_LAYERS 5
#define NUM_SAMPLED_VERTICES 64
#define VERTICES_PER_SAMPLE 64
#define NUM_SAMPLES 10000

class ImportanceSample 
{
public:
  int adjacencyMatrixLen[NUM_LAYERS];
  int adjacencyMatrixRow[NUM_LAYERS][VERTICES_PER_SAMPLE*NUM_SAMPLED_VERTICES];
  int adjacencyMatrixCol[NUM_LAYERS][VERTICES_PER_SAMPLE*NUM_SAMPLED_VERTICES];
  int adjacencyMatrixVal[NUM_LAYERS][VERTICES_PER_SAMPLE*NUM_SAMPLED_VERTICES];
};

struct ImportanceSampleApp {
  __host__ __device__ int steps() {return NUM_LAYERS;}

  __host__ __device__ 
  int stepSize(int k) {
    return NUM_SAMPLED_VERTICES;
  }

  template<typename SampleType, typename EdgeArray, typename WeightArray>
  __device__ inline
  VertexID next(int step, CSRPartition* csr, const VertexID* transits, const VertexID sampleIdx,
    SampleType* sample, 
    const float max_weight,
    EdgeArray& transitEdges, WeightArray& transitEdgeWeights,
    const EdgePos_t numEdges, const VertexID_t neighbrID, curandState* state)
  {
    EdgePos_t id = RandNumGen::rand_int(state, csr->get_n_vertices());
    for (int i = 0; i < VERTICES_PER_SAMPLE; i++) {
      VertexID transit = transits[i];
      bool hasEdge = csr->has_edge_logn(transit, id);
      int len = i*VERTICES_PER_SAMPLE + neighbrID;//::atomicAdd(&sample->adjacencyMatrixLen[step], 1);
      int ii = -1, jj = -1;
      if (hasEdge) {
        //int cooIdx = step * NUM_SAMPLED_VERTICES + len;
        ii = i; jj = neighbrID;
      }
      sample->adjacencyMatrixRow[step][len] = ii;
      sample->adjacencyMatrixCol[step][len] = jj;
      //sample->adjacencyMatrixVal[step][len] = 1.0f;
    }

    return id;
  }

  __host__ __device__ int samplingType()
  {
    return SamplingType::CollectiveNeighborhood;
  }

  __host__ __device__ OutputFormat outputFormat()
  {
    return AdjacencyMatrix;
  }

  __host__ EdgePos_t numSamples(CSR* graph)
  {
    return NUM_SAMPLES;//graph->get_n_vertices() / VERTICES_PER_SAMPLE / (graph->get_n_vertices() > 1000000 ? 100 : 1);
  }

  __host__ __device__ bool hasExplicitTransits()
  {
    return false;
  }

  template<class SampleType>
  __device__ VertexID_t stepTransits(int step, const VertexID_t sampleID, SampleType& sample, int transitIdx, curandState* randState)
  {
    return -1;
  }

  template<class SampleType>
  __host__ std::vector<VertexID_t> initialSample(int sampleIdx, CSR* graph, SampleType& sample)
  {
    std::vector<VertexID_t> initialValue;

    for (int i = 0; i < VERTICES_PER_SAMPLE; i++) {
      initialValue.push_back((sampleIdx * VERTICES_PER_SAMPLE + i) % graph->get_n_vertices());
    }

    return initialValue;
  }

  template<class SampleType>
  __host__ SampleType initializeSample(CSR* graph, const VertexID_t sampleID)
  {
    SampleType sample;
    for (int i = 0; i < NUM_LAYERS; i++) {
      sample.adjacencyMatrixLen[i] = 0;
    }
    return sample;
  }

  __host__ __device__ EdgePos_t initialSampleSize(CSR* graph)
  {
    return VERTICES_PER_SAMPLE;
  }
};