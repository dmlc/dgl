for lib in gt igraph nx dgl; do
    echo "Profiling ${lib}"
    python ${lib}_bench.py > ${lib}_bench
    bash ${lib}_bench.sh
done
