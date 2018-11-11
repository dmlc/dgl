#!/usr/bin/env groovy

def init_git_submodule() {
  sh "git submodule init"
  sh "git submodule update"
}

def setup() {
  init_git_submodule()
}

def build_dgl() {
  sh "if [ -d build ]; then rm -rf build; fi; mkdir build"
  dir("python") {
    sh "python3 setup.py install"
  }
  dir ("build") {
    sh "cmake .."
    sh "make -j4"
  }
}

def pytorch_unit_test(dev) {
  withEnv(["DGL_LIBRARY_PATH=${env.WORKSPACE}/build", "PYTHONPATH=${env.WORKSPACE}/python"]) {
    sh "python3 -m nose -v --with-xunit tests"
    sh "python3 -m nose -v --with-xunit tests/pytorch"
    sh "python3 -m nose -v --with-xunit tests/graph_index"
  }
}

def mxnet_unit_test(dev) {
  withEnv(["DGL_LIBRARY_PATH=${env.WORKSPACE}/build", "PYTHONPATH=${env.WORKSPACE}/python"]) {
    sh "python3 -m nose -v --with-xunit tests/mxnet"
  }
}

def example_test(dev) {
  dir ("tests/scripts") {
    withEnv(["DGL_LIBRARY_PATH=${env.WORKSPACE}/build", "PYTHONPATH=${env.WORKSPACE}/python"]) {
      sh "./test_examples.sh ${dev}"
    }
  }
}

pipeline {
  agent none
  stages {
    stage("Lint Check") {
      agent {
        docker {
          image "lingfanyu/dgl-lint"
        }
      }
      steps {
        init_git_submodule()
        sh "tests/scripts/task_lint.sh"
      }
    }
    stage("Build") {
      parallel {
        stage("CPU Build") {
          agent {
            docker {
              image "lingfanyu/dgl-cpu"
              args "-v cpu-ws:/workspace"
              args "-w /workspace"
            }
          }
          steps {
            setup()
            build_dgl()
          }
        }
        stage("GPU Build") {
          agent {
            docker {
              image "lingfanyu/dgl-gpu"
              args "--runtime nvidia"
              args "-v gpu-ws:/workspace"
              args "-w /workspace"
            }
          }
          steps {
            setup()
            build_dgl()
            sh "nvidia-smi"
          }
        }
        stage("MXNet CPU Build (temp)") {
          agent {
            docker {
              image "zhengda1936/dgl-mxnet-cpu:v3"
              args "-v mx-cpu-ws:/workspace"
              args "-w /workspace"
            }
          }
          steps {
            setup()
            build_dgl()
          }
        }
      }
    }
    stage("Test") {
      parallel {
        stage("Pytorch CPU") {
          agent {
            docker {
              image "lingfanyu/dgl-cpu"
              args "-v cpu-ws:/workspace"
              args "-w /workspace"
            }
          }
          stages {
            stage("TH CPU unittest") {
              steps { pytorch_unit_test("CPU") }
            }
            stage("TH CPU example test") {
              steps { example_test("CPU") }
            }
          }
          post {
            always { junit "*.xml" }
          }
        }
        stage("Pytorch GPU") {
          agent {
            docker {
              image "lingfanyu/dgl-gpu"
              args "--runtime nvidia"
              args "-v gpu-ws:/workspace"
              args "-w /workspace"
            }
          }
          stages {
            stage("TH GPU unittest") {
              steps { pytorch_unit_test("GPU") }
            }
            stage("TH GPU example test") {
              steps { example_test("GPU") }
            }
          }
          post {
            always { junit "*.xml" }
          }
        }
        stage("MXNet CPU") {
          agent {
            docker {
              image "zhengda1936/dgl-mxnet-cpu:v3"
              args "-v mx-cpu-ws:/workspace"
              args "-w /workspace"
            }
          }
          stages {
            stage("MX Unittest") {
              steps { mxnet_unit_test("CPU") }
            }
          }
          post {
            always { junit "*.xml" }
          }
        }
      }
    }
  }
}
