#!/usr/bin/env groovy

dgl_linux_libs = "build/libdgl.so, build/runUnitTests, python/dgl/_ffi/_cy3/core.cpython-35m-x86_64-linux-gnu.so"

def init_git() {
  sh "rm -rf *"
  checkout scm
  sh "git submodule init"
  sh "git submodule update"
  sh "ls -lh"
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
  init_git()
  sh "bash tests/scripts/build_dgl.sh"
  pack_lib("dgl-${dev}", dgl_linux_libs)
}

def build_dgl_win64() {
  /* Assuming that Windows slaves are already configured with MSBuild VS2017,
   * CMake and Python/pip/setuptools etc. */
  bat "CALL tests\\scripts\\build_dgl.bat"
}

def cpp_unit_test_linux() {
  init_git()
  unpack_lib("dgl-cpu", dgl_linux_libs)
  sh "bash tests/scripts/task_cpp_unit_test.sh"
}

def cpp_unit_test_windows() {
  bat "CALL tests\\scripts\\task_cpp_unit_test.bat"
}

def unit_test(backend, dev) {
  init_git()
  unpack_lib("dgl-${dev}", dgl_linux_libs)
  timeout(time: 2, unit: 'MINUTES') {
    sh "bash tests/scripts/task_unit_test.sh ${backend}"
  }
}

//def unit_test_win64(backend, dev) {
//  withEnv(["DGL_LIBRARY_PATH=${env.WORKSPACE}\\build", "PYTHONPATH=${env.WORKSPACE}\\python", "DGLBACKEND=${backend}"]) {
//    bat "CALL tests\\scripts\\task_unit_test.bat ${backend}"
//  }
//}

def example_test(backend, dev) {
  init_git()
  unpack_lib("dgl-${dev}", dgl_linux_libs)
  timeout(time: 20, unit: 'MINUTES') {
    sh "bash tests/scripts/task_example_test.sh ${dev}"
  }
}

//def example_test_win64(backend, dev) {
//  withEnv(["DGL_LIBRARY_PATH=${env.WORKSPACE}\\build", "PYTHONPATH=${env.WORKSPACE}\\python", "DGLBACKEND=${backend}"]) {
//    dir ("tests\\scripts") {
//      bat "CALL task_example_test ${dev}"
//    }
//  }
//}

def tutorial_test(backend) {
  init_git()
  unpack_lib("dgl-cpu", dgl_linux_libs)
  timeout(time: 20, unit: 'MINUTES') {
    sh "bash tests/scripts/task_${backend}_tutorial_test.sh"
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
      parallel {
        stage("CPU Build") {
          agent { docker { image "dgllib/dgl-ci-cpu" } }
          steps {
            build_dgl_linux("cpu")
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
            sh "nvidia-smi"
            build_dgl_linux("gpu")
          }
        }
      }
    }
    stage("Test") {
      parallel {
        stage("C++ CPU") {
          agent { docker { image "dgllib/dgl-ci-cpu" } }
          steps {
            cpp_unit_test_linux()
          }
        }
        stage("Torch CPU") {
          agent { docker { image "dgllib/dgl-ci-cpu" } }
          stages {
            stage("Unit test") {
              steps {
                unit_test("pytorch", "cpu")
              }
            }
            stage("Example test") {
              steps {
                example_test("pytorch", "cpu")
              }
            }
            stage("Tutorial test") {
              steps {
                tutorial_test("pytorch")
              }
            }
          }
        }
        stage("Torch GPU") {
          agent {
            docker {
              image "dgllib/dgl-ci-gpu"
              args "--runtime nvidia"
            }
          }
          stages {
            stage("Unit test") {
              steps {
                //unit_test("pytorch", "gpu")
                sh "nvidia-smi"
              }
            }
            stage("Example test") {
              steps {
                example_test("pytorch", "gpu")
              }
            }
          }
        }
        stage("MXNet CPU") {
          agent { docker { image "dgllib/dgl-ci-cpu" } }
          stages {
            stage("Unit test") {
              steps {
                unit_test("mxnet", "cpu")
              }
            }
            //stage("Example test") {
            //  steps {
            //    unit_test("pytorch", "cpu")
            //  }
            //}
            //stage("Tutorial test") {
            //  steps {
            //    tutorial_test("mxnet")
            //  }
            //}
          }
        }
      }
    }
  }
}
