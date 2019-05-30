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

def cpp_unit_test_linux(){
  sh "bash tests/scripts/task_cpp_unit_test.sh"
}

def cpp_unit_test_windows(){
  bat "CALL tests\\scripts\\task_cpp_unit_test.bat"
}

def unit_test(backend, dev) {
  withEnv(["DGL_LIBRARY_PATH=${env.WORKSPACE}/build", "PYTHONPATH=${env.WORKSPACE}/python", "DGLBACKEND=${backend}"]) {
    sh "bash tests/scripts/task_unit_test.sh ${backend}"
  }
}

def unit_test_win64(backend, dev) {
  withEnv(["DGL_LIBRARY_PATH=${env.WORKSPACE}\\build", "PYTHONPATH=${env.WORKSPACE}\\python", "DGLBACKEND=${backend}"]) {
    bat "CALL tests\\scripts\\task_unit_test.bat ${backend}"
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
      agent { docker { image "dgllib/dgl-ci-lint" } }
      steps {
        init_git_submodule()
        sh "pwd"
        sh "ls -lh"
      }
    }
    stage("Build") {
      parallel {
        stage("CPU Build") {
          agent { docker { image "dgllib/dgl-ci-cpu" } }
          steps {
            ws('workspace/cpu-build') {
              //init_git_submodule()
              sh "pwd"
              sh "ls -lh"
            }
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
            ws('workspace/gpu-build') {
              //init_git_submodule()
              sh "pwd"
              sh "ls -lh"
            }
          }
        }
        //stage("CPU Build (Win64)") {
        //  agent { label "windows" }
        //  steps {
        //    ws('workspace/cpu-build-win') {
        //      init_git_submodule_win64()
        //      build_dgl_win64()
        //    }
        //  }
        //}
      }
    }
    stage("Test") {
      parallel {
        stage("CPP Test") {
          stages {
            stage("CPP Unit Test Linux") {
              agent { docker {image "dgllib/dgl-ci-cpu"} }
              steps { 
                ws('workspace/cpu-build') {
                  sh "pwd"
                  sh "ls -lh"
                }
              }
            }
            //stage("CPP Unit Test Windows") {
            //  agent { label "windows" }
            //  steps {
            //    ws('workspace/cpu-build-win') {
            //      cpp_unit_test_windows()
            //    }
            //  }
            //}
          }
        }
        stage("Pytorch CPU") {
          agent { docker { image "dgllib/dgl-ci-cpu" } }
          stages {
            stage("TH CPU unittest") {
              steps {
                ws('workspace/cpu-build') {
                  sh "pwd"
                  sh "ls -lh"
                }
              }
            }
            stage("TH CPU example test") {
              steps {
                ws('workspace/cpu-build') {
                  sh "pwd"
                  sh "ls -lh"
                }
              }
            }
          }
          post {
            always { junit "*.xml" }
          }
        }
        //stage("Pytorch CPU (Windows)") {
        //  agent { label "windows" }
        //  stages {
        //    stage("TH CPU Win64 unittest") {
        //      steps {
        //        ws('workspace/cpu-build-win') {
        //          unit_test_win64("pytorch", "CPU")
        //        }
        //      }
        //    }
        //    stage("TH CPU Win64 example test") {
        //      steps {
        //        ws('workspace/cpu-build-win') {
        //          example_test_win64("pytorch", "CPU")
        //        }
        //      }
        //    }
        //  }
        //  post {
        //    always { junit "*.xml" }
        //  }
        //}
        stage("Pytorch GPU") {
          agent {
            docker {
              image "dgllib/dgl-ci-gpu"
              args "--runtime nvidia"
            }
          }
          stages {
            // TODO: have GPU unittest
            //stage("TH GPU unittest") {
            //  steps { pytorch_unit_test("GPU") }
            //}
            stage("TH GPU example test") {
              steps {
                ws('workspace/gpu-build') {
                  sh "pwd"
                  sh "ls -lh"
                }
              }
            }
          }
          // TODO: have GPU unittest
          //post {
          //  always { junit "*.xml" }
          //}
        }
        stage("MXNet CPU") {
          agent {
            docker { image "dgllib/dgl-ci-mxnet-cpu" }
          }
          stages {
            stage("MX Unittest") {
              options {
                  timeout(time: 5, unit: 'MINUTES') 
              }
              steps {
                ws('workspace/cpu-build') {
                  sh "pwd"
                  sh "ls -lh"
                }
              }
            }
          }
          post {
            always { junit "*.xml" }
          }
        }
      }
    }
    //stage("Doc") {
    //  parallel {
    //    stage("TH Tutorial") {
    //      agent {
    //        docker { image "dgllib/dgl-ci-cpu" }
    //      }
    //      steps {
    //        ws('workspace/cpu-build') {
    //          pytorch_tutorials()
    //        }
    //      }
    //    }
    //    //stage("MX Tutorial") {
    //    //  agent {
    //    //    docker { image "dgllib/dgl-ci-mxnet-cpu" }
    //    //  }
    //    //  steps {
    //    //    mxnet_tutorials()
    //    //  }
    //    //}
    //  }
    //}
  }
}
