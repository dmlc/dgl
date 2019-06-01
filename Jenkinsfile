#!/usr/bin/env groovy

dgl_linux_libs = "build/libdgl.so, python/dgl/_ffi/_cy3/core.cpython-35m-x86_64-linux-gnu.so"

def init_git() {
  checkout scm
  sh "git submodule init"
  sh "git submodule update"
}

def init_git_win64() {
  checkout scm
  bat "git submodule init"
  bat "git submodule update"
}

// pack libraries for later use
def pack_lib(name, libs) {
  sh """
     echo "Packing ${libs} into ${name}"
     echo ${libs} | sed -e 's/,/ /g' | xargs md5sum
     """
  stash includes: libs, name: name
}

// unpack libraries saved before
def unpack_lib(name, libs) {
  unstash name
  sh """
     echo "Unpacked ${libs} from ${name}"
     echo ${libs} | sed -e 's/,/ /g' | xargs md5sum
     """
}

def build_dgl_linux(dev) {
  ws("workspace/${dev}-build") {
    init_git()
    sh "bash tests/scripts/build_dgl.sh"
    pack_lib("dgl-${dev}", dgl_linux_libs)
  }
}

def build_dgl_win64() {
  /* Assuming that Windows slaves are already configured with MSBuild VS2017,
   * CMake and Python/pip/setuptools etc. */
  bat "CALL tests\\scripts\\build_dgl.bat"
}

def cpp_unit_test_linux() {
  ws("workspace/cpp-cpu-test") {
    init_git()
    unpack_lib("dgl-cpu", dgl_linux_libs)
    sh "bash tests/scripts/task_cpp_unit_test.sh"
  }
}

def cpp_unit_test_windows() {
  bat "CALL tests\\scripts\\task_cpp_unit_test.bat"
}

def unit_test(backend, dev) {
  def wspace = "workspace/${backend}-${dev}-unittest"
  ws(wspace) {
    init_git()
    unpack_lib("dgl-${dev}", dgl_linux_libs)
    timeout(time: 2, unit: 'MINUTES') {
      sh "bash tests/scripts/task_unit_test.sh ${backend}"
    }
  }
}

//def unit_test_win64(backend, dev) {
//  withEnv(["DGL_LIBRARY_PATH=${env.WORKSPACE}\\build", "PYTHONPATH=${env.WORKSPACE}\\python", "DGLBACKEND=${backend}"]) {
//    bat "CALL tests\\scripts\\task_unit_test.bat ${backend}"
//  }
//}

def example_test(backend, dev) {
  sh "pwd"
  //def wspace = "${env.WORKSPACE}/${backend}-${dev}-exptest/"
  //def build = "${env.WORKSPACE}/${dev}-build/"
  //ws("${wspace}") {
    //withEnv(["DGL_LIBRARY_PATH=${build}/build",
             //"PYTHONPATH=${build}/python",
             //"DGLBACKEND=${backend}",
             //"DGL_DOWNLOAD_DIR=${wspace}"]) {
    withEnv(["DGL_LIBRARY_PATH=${env.WORKSPACE}/build",
             "PYTHONPATH=${env.WORKSPACE}/python",
             "DGLBACKEND=${backend}",
             "DGL_DOWNLOAD_DIR=${env.WORKSPACE}"]) {
      timeout(time: 20, unit: 'MINUTES') {
        sh "bash tests/scripts/task_example_test.sh ${dev}"
      }
    }
  //}
}

//def example_test_win64(backend, dev) {
//  withEnv(["DGL_LIBRARY_PATH=${env.WORKSPACE}\\build", "PYTHONPATH=${env.WORKSPACE}\\python", "DGLBACKEND=${backend}"]) {
//    dir ("tests\\scripts") {
//      bat "CALL task_example_test ${dev}"
//    }
//  }
//}

def tutorial_test(backend) {
  def wspace = "${env.WORKSPACE}/${backend}-tuttest/"
  def build = "${env.WORKSPACE}/cpu-build/"
  ws("${wspace}") {
    withEnv(["DGL_LIBRARY_PATH=${env.WORKSPACE}/build",
             "PYTHONPATH=${env.WORKSPACE}/python",
             "DGLBACKEND=${backend}"]) {
      timeout(time: 20, unit: 'MINUTES') {
        sh "bash tests/scripts/task_${backend}_tutorial_test.sh"
      }
    }
  }
}

pipeline {
  agent none
  stages {
    stage("Lint Check") {
      agent { docker { image "dgllib/dgl-ci-lint" } }
      steps {
        init_git()
        sh "bash tests/scripts/task_lint.sh"
      }
    }
    stage("Build") {
      agent { label "CPUNode" }
      steps {
        build_dgl_linux("cpu")
      }
    }
    stage("Test") {
      agent { label "CPUNode" }
      steps {
        unit_test("pytorch", "cpu")
      }
    }

    /*stage("Build & Test") {
      parallel {
        stage("Linux CPU") {
          agent { docker { image "dgllib/dgl-ci-cpu" } }
          stages {
            stage("Build") {
              steps { build_dgl("cpu") }
            }
            stage("CPP test") {
              steps {
                //cpp_unit_test_linux()
                sh "ls ./build"
              }
            }
            stage("TH unit test") {
              steps { unit_test("pytorch", "cpu") }
            }
            stage("TH example test") {
              steps { example_test("pytorch", "cpu") }
            }
            //stage("MX unit test") {
            //  steps { unit_test("mxnet", "cpu") }
            //}
            stage("Torch tutorial test") {
              steps { tutorial_test("pytorch") }
            }
          }
        }
        stage("Linux GPU") {
          agent {
            docker {
              image "dgllib/dgl-ci-gpu"
              args "--runtime nvidia"
            }
          }
          stages {
            stage("Build") {
              steps { build_dgl("gpu") }
            }
            stage("TH example test") {
              steps { example_test("pytorch", "gpu") }
            }
          }
        }
      }
    }*/

    /*stage("Build") {
      parallel {
        stage("CPU Build") {
          agent { docker { image "dgllib/dgl-ci-cpu" } }
          steps { build_dgl("cpu") }
        }
        stage("GPU Build") {
          agent {
            docker {
              image "dgllib/dgl-ci-gpu"
              args "--runtime nvidia"
            }
          }
          steps { build_dgl("gpu") }
        }
        //stage("CPU Build (Win64/PyTorch)") {
        //  agent {
        //    label "windows"
        //  }
        //  steps {
        //    ws ("${env.WORKSPACE}/cpu-build") {
        //      init_git_win64()
        //      build_dgl_win64()
        //    }
        //  }
        //}
      }
    }
    stage("Test") {
      parallel {
        stage("CPP Test"){
          stages{
            stage("CPP Unit Test Linux"){
              agent { docker {image "dgllib/dgl-ci-cpu"} }
              steps { 
                cpp_unit_test_linux() 
              }
            }
            //stage("CPP Unit Test Windows"){
            //  agent {
            //    label "windows"
            //  }
            //  steps {
            //    init_git_win64()
            //    cpp_unit_test_windows()
            //  }
            //}
          }
        }
        stage("Pytorch CPU") {
          agent { docker { image "dgllib/dgl-ci-cpu" } }
          stages {
            stage("TH CPU unittest") {
              steps { unit_test("pytorch", "cpu") }
            }
            stage("TH CPU example test") {
              steps { example_test("pytorch", "cpu") }
            }
          }
        }
        //stage("Pytorch CPU (Windows)") {
        //  agent { label "windows" }
        //  stages {
        //    stage("TH CPU Win64 unittest") {
        //      steps { unit_test_win64("pytorch", "CPU") }
        //    }
        //    stage("TH CPU Win64 example test") {
        //      steps { example_test_win64("pytorch", "CPU") }
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
              steps { example_test("pytorch", "gpu") }
            }
          }
        }
        stage("MXNet CPU") {
          agent { docker { image "dgllib/dgl-ci-mxnet-cpu" } }
          stages {
            stage("MX Unittest") {
              steps { unit_test("mxnet", "cpu") }
            }
          }
        }
      }
    }
    stage("Doc") {
      parallel {
        stage("TH Tutorial") {
          agent { docker { image "dgllib/dgl-ci-cpu" } }
          steps {
            tutorial_test("pytorch")
          }
        }
        //stage("MX Tutorial") {
        //  agent {
        //    docker { image "dgllib/dgl-ci-mxnet-cpu" }
        //  }
        //  steps {
        //    mxnet_tutorials()
        //  }
        //}
      }
    }
    */
  }
}
