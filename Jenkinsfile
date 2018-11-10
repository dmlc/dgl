#!/usr/bin/env groovy

def init_git_submodule() {
  sh "git submodule init"
  sh "git submodule update"
}

def setup() {
  sh "easy_install nose"
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
  withEnv(["DGL_LIBRARY_PATH=${env.WORKSPACE}/build"]) {
    sh "nosetests tests -v --with-xunit"
    sh "nosetests tests/pytorch -v --with-xunit"
    sh "nosetests tests/graph_index -v --with-xunit"
  }
}

def mxnet_unit_test(dev) {
  withEnv(["DGL_LIBRARY_PATH=${env.WORKSPACE}/build"]) {
    sh "nosetests tests/mxnet -v --with-xunit"
  }
}

def example_test(dev) {
  dir ("tests/scripts") {
    withEnv(["DGL_LIBRARY_PATH=${env.WORKSPACE}/build"]) {
      sh "./test_examples.sh ${dev}"
    }
  }
}

pipeline {
  agent none
  stages {
    stage("XXX") {
      agent any
      steps {
        sh "echo ${env.WORKSPACE}"
      }
    }
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
              args "-v ${env.WORKSPACE}:/workspace"
            }
          }
          steps { dir ("/workspace") {
            setup()
            build_dgl()
          }}
        }
        stage("GPU Build") {
          agent {
            docker {
              image "lingfanyu/dgl-gpu"
              args "--runtime nvidia"
              args "-v ${env.WORKSPACE}:/workspace"
            }
          }
          steps { dir ("/workspace") {
            setup()
            build_dgl()
          }}
        }
        stage("MXNet CPU Build (temp)") {
          agent {
            docker {
              image "zhengda1936/dgl-mxnet-cpu:v3"
              args "-v ${env.WORKSPACE}:/workspace"
            }
          }
          steps { dir ("/workspace") {
            setup()
            build_dgl()
          }}
        }
      }
    }
    stage("Test") {
      parallel {
        stage("Pytorch CPU") {
          agent {
            docker {
              image "lingfanyu/dgl-cpu"
              args "-v ${env.WORKSPACE}:/workspace"
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
              args "-v ${env.WORKSPACE}:/workspace"
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
              args "-v ${env.WORKSPACE}:/workspace"
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
