#!/usr/bin/env groovy

def init_git_submodule() {
  sh "git submodule init"
  sh "git submodule update"
}

def setup() {
  init_git_submodule()
}

def build_dgl() {
  // sh "if [ -d build ]; then rm -rf build; fi; mkdir build"
  sh "rm -rf _download"
  dir ("build") {
    sh "cmake .."
    sh "make -j4"
  }
  dir("python") {
    // sh "rm -rf build *.egg-info dist"
    // sh "pip3 uninstall -y dgl"
    sh "python3 setup.py install"
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
    sh "python3 -m nose -v --with-xunit tests/graph_index"
  }
}

def example_test(dev) {
  withEnv(["DGL_LIBRARY_PATH=${env.WORKSPACE}/build", "PYTHONPATH=${env.WORKSPACE}/python"]) {
    dir ("tests/scripts") {
      sh "bash task_example_test.sh ${dev}"
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
  withEnv(["DGL_LIBRARY_PATH=${env.WORKSPACE}/build", "PYTHONPATH=${env.WORKSPACE}/python"]) {
    dir("tests/scripts") {
      sh "bash task_mxnet_tutorial_test.sh"
    }
  }
}
pipeline {
  agent none
  stages {
    stage("Lint Check") {
      agent { docker { image "dgllib/dgl-ci-lint" } }
      steps {
        init_git_submodule()
        sh "bash tests/scripts/task_lint.sh"
      }
    }
    stage("Build") {
      parallel {
        stage("CPU Build") {
          agent { 
            docker { 
              image "dgllib/dgl-ci-cpu"
              reuseNode true
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
              image "dgllib/dgl-ci-gpu"
              reuseNode true
              args "--runtime nvidia"
            }
          }
          steps {
            setup()
            build_dgl()
          }
        }
        stage("MXNet CPU Build (temp)") {
          agent { 
            docker { 
              image "dgllib/dgl-ci-mxnet-cpu"
              reuseNode true
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
              image "dgllib/dgl-ci-cpu"
              reuseNode true
            }
          }
          stages {
            stage("TH CPU unittest") {
              steps { 
                sh "ls"
                sh "pwd"
                pytorch_unit_test("CPU") 
              }
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
              image "dgllib/dgl-ci-gpu"
              reuseNode true
              args "--runtime nvidia"
            }
          }
          stages {
            // TODO: have GPU unittest
            //stage("TH GPU unittest") {
            //  steps { pytorch_unit_test("GPU") }
            //}
            stage("TH GPU example test") {
              steps { example_test("GPU") }
            }
          }
          // TODO: have GPU unittest
          //post {
          //  always { junit "*.xml" }
          //}
        }
        stage("MXNet CPU") {
          agent { 
            docker { 
              image "dgllib/dgl-ci-mxnet-cpu"
              reuseNode true
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
    /*
    stage("Doc") {
      parallel {
        stage("TH Tutorial") {
          agent { 
            docker { 
              image "dgllib/dgl-ci-cpu"
              reuseNode true
            }
          }
          steps {
            pytorch_tutorials()
          }
        }
        stage("MX Tutorial") {
          agent { 
            docker { 
              image "dgllib/dgl-ci-mxnet-cpu"
              reuseNode true
            }
          }
          steps {
            mxnet_tutorials()
          }
        }
      }
    }
  */
    stage("Build Docs"){
      steps{
        withCredentials([sshUserPrivateKey(credentialsId: "yourkeyid", keyFileVariable: 'keyfile')]) {
         stage('scp-f/b') {
           sh 'echo $yourkeyid'
           sh 'echo $keyfile'
           // sh "scp -i ${keyfile} do sth here"
         }
        }
      }
    }
  }  
}
