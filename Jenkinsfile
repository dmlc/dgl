#!/usr/bin/env groovy

def init_git_submodule() {
  sh "git submodule init"
  sh "git submodule update"
}

def init_git_submodule_win64() {
  bat "git submodule init"
  bat "git submodule update"
}

def build_dgl() {
  sh "bash tests/scripts/build_dgl.sh"
}

def build_dgl_win64() {
  /* Assuming that Windows slaves are already configured with MSBuild VS2017,
   * CMake and Python/pip/setuptools etc. */
  bat "CALL tests\\scripts\\build_dgl.bat"
}

def unit_test(backend, dev) {
  withEnv(["DGL_LIBRARY_PATH=${env.WORKSPACE}/build", "PYTHONPATH=${env.WORKSPACE}/python", "DGLBACKEND=${backend}"]) {
    sh "bash tests/scripts/task_unit_test.sh ${dev}"
  }
}

def unit_test_win64(backend, dev) {
  withEnv(["DGL_LIBRARY_PATH=${env.WORKSPACE}\\build", "PYTHONPATH=${env.WORKSPACE}\\python", "DGLBACKEND=${backend}"]) {
    bat "CALL tests\\scripts\\task_unit_test.bat ${dev}"
  }
}

def example_test(backend, dev) {
  withEnv(["DGL_LIBRARY_PATH=${env.WORKSPACE}/build", "PYTHONPATH=${env.WORKSPACE}/python", "DGLBACKEND=${backend}"]) {
    dir ("tests/scripts") {
      sh "bash task_example_test.sh ${dev}"
    }
  }
}

def example_test_win64(backend, dev) {
  withEnv(["DGL_LIBRARY_PATH=${env.WORKSPACE}\\build", "PYTHONPATH=${env.WORKSPACE}\\python", "DGLBACKEND=${backend}"]) {
    dir ("tests\\scripts") {
      bat "CALL task_example_test ${dev}"
    }
  }
}

def pytorch_tutorials() {
  withEnv(["DGL_LIBRARY_PATH=${env.WORKSPACE}/build", "PYTHONPATH=${env.WORKSPACE}/python"]) {
    dir ("tests/scripts") {
      sh "bash task_pytorch_tutorial_test.sh"
    }
  }
}

def mxnet_tutorials() {
  withEnv(["DGL_LIBRARY_PATH=${env.WORKSPACE}/build", "PYTHONPATH=${env.WORKSPACE}/python", "DGLBACKEND=mxnet"]) {
    dir("tests/scripts") {
      sh "bash task_mxnet_tutorial_test.sh"
    }
  }
}
pipeline {
  agent none
  stages {
    stage("Lint Check") {
      agent {
        docker { image "dgllib/dgl-ci-lint" }
      }
      steps {
        init_git_submodule()
        sh "bash tests/scripts/task_lint.sh"
      }
    }
    stage("Build") {
      parallel {
        stage("CPU Build") {
          agent {
            docker { image "dgllib/dgl-ci-cpu" }
          }
          steps {
            init_git_submodule()
            build_dgl()
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
            init_git_submodule()
            build_dgl()
          }
        }
        stage("MXNet CPU Build (temp)") {
          agent {
            docker { image "dgllib/dgl-ci-mxnet-cpu" }
          }
          steps {
            init_git_submodule()
            build_dgl()
          }
        }
        stage("CPU Build (Win64/PyTorch)") {
          agent {
            label "windows"
          }
          steps {
            init_git_submodule_win64()
            build_dgl_win64()
          }
        }
      }
    }
    stage("Test") {
      parallel {
        stage("Pytorch CPU") {
          agent {
            docker { image "dgllib/dgl-ci-cpu" }
          }
          stages {
            stage("TH CPU unittest") {
              steps { unit_test("pytorch", "CPU") }
            }
            stage("TH CPU example test") {
              steps { example_test("pytorch", "CPU") }
            }
          }
          post {
            always { junit "*.xml" }
          }
        }
        stage("Pytorch CPU (Windows)") {
          agent { label "windows" }
          stages {
            stage("TH CPU Win64 unittest") {
              steps { unit_test_win64("pytorch", "CPU") }
            }
            stage("TH CPU Win64 example test") {
              steps { example_test_win64("pytorch", "CPU") }
            }
          }
          post {
            always { junit "*.xml" }
          }
        }
        stage("Pytorch GPU") {
          agent {
            docker {
              image "dgllib/dgl-ci-gpu"
              args "--runtime nvidia"
            }
          }
          stages {
            stage("TH GPU unittest") {
              steps { unit_test("GPU") }
            }
            stage("TH GPU example test") {
              steps { example_test("pytorch", "GPU") }
            }
          }
          post {
            always { junit "*.xml" }
          }
        }
        stage("MXNet CPU") {
          agent {
            docker { image "dgllib/dgl-ci-mxnet-cpu" }
          }
          stages {
            stage("MX Unittest") {
              steps { unit_test("mxnet", "CPU") }
            }
          }
          post {
            always { junit "*.xml" }
          }
        }
      }
    }
    stage("Doc") {
      parallel {
        stage("TH Tutorial") {
          agent {
            docker { image "dgllib/dgl-ci-cpu" }
          }
          steps {
            pytorch_tutorials()
          }
        }
        stage("MX Tutorial") {
          agent {
            docker { image "dgllib/dgl-ci-mxnet-cpu" }
          }
          steps {
            mxnet_tutorials()
          }
        }
      }
    }
  }
}
