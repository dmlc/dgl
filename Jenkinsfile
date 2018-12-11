def init_git_submodule() {
}

def setup() {
}

def build_dgl() {
}

def pytorch_unit_test(dev) {
}

def mxnet_unit_test(dev) {
}

def example_test(dev) {
}

def pytorch_tutorials() {
}

def mxnet_tutorials() {
}

pipeline {
    agent none
    stages {
        stage("Lint Check") {
            agent { docker { image "dgllib/dgl-ci-lint" } }
            steps {
                setup()
            }
        }
        stage("Build") {
            parallel {
                stage("CPU Build") {
                    agent { docker { image "dgllib/dgl-ci-cpu" } }
                    steps {
                        setup()
                    }
                }
                stage("GPU Build") {
                    agent {
                        docker {
                            image "dgllib/dgl-ci-gpu"
                            args "--runtime nvidia"
                        }
                    }
                    steps {
                        setup()
                        build_dgl()
                    }
                }
                stage("MXNet CPU Build (temp)") {
                    agent { docker { image "dgllib/dgl-ci-mxnet-cpu" } }
                    steps {
                        setup()
                        build_dgl()
                    }
                }
            }
        }
//        stage("Test") {
//            parallel {
//                stage("Pytorch CPU") {
//                    agent { docker { image "dgllib/dgl-ci-cpu" } }
//                    stages {
//                        stage("TH CPU unittest") {
//                            steps { pytorch_unit_test("CPU") }
//                        }
//                        stage("TH CPU example test") {
//                            steps { example_test("CPU") }
//                        }
//                    }
//                }
//                stage("Pytorch GPU") {
//                    agent {
//                        docker {
//                            image "dgllib/dgl-ci-gpu"
//                            args "--runtime nvidia"
//                        }
//                    }
//                    stages {
//                        // TODO: have GPU unittest
//                        //stage("TH GPU unittest") {
//                        //  steps { pytorch_unit_test("GPU") }
//                        //}
//                        stage("TH GPU example test") {
//                            steps { example_test("GPU") }
//                        }
//                    }
//                    // TODO: have GPU unittest
//                    //post {
//                    //  always { junit "*.xml" }
//                    //}
//                }
//                stage("MXNet CPU") {
//                    agent { docker { image "dgllib/dgl-ci-mxnet-cpu" } }
//                    stages {
//                        stage("MX Unittest") {
//                            steps { mxnet_unit_test("CPU") }
//                        }
//                    }
//                }
//            }
//        }
        stage("Doc") {
            parallel {
                stage("TH Tutorial") {
                    agent { docker { image "dgllib/dgl-ci-cpu" } }
                    steps {
                        pytorch_tutorials()
                    }
                }
                stage("MX Tutorial") {
                    agent { docker { image "dgllib/dgl-ci-mxnet-cpu" } }
                    steps {
                        mxnet_tutorials()
                    }
                }
            }
        }
    }
}
