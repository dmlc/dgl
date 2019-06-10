#!/usr/bin/env groovy

dgl_linux_libs = "build/libdgl.so, build/runUnitTests, python/dgl/_ffi/_cy3/core.cpython-35m-x86_64-linux-gnu.so"
// Currently DGL on Windows is not working with Cython yet
dgl_win64_libs = "build\\dgl.dll, build\\runUnitTests.exe"

def init_git() {
  sh "rm -rf *"
  checkout scm
  sh "git submodule update --recursive --init"
}

def init_git_win64() {
  checkout scm
  bat "git submodule update --recursive --init"
}

// pack libraries for later use
def pack_lib(name, libs) {
  echo "Packing ${libs} into ${name}"
  stash includes: libs, name: name
}

// unpack libraries saved before
def unpack_lib(name, libs) {
  unstash name
  echo "Unpacked ${libs} from ${name}"
}

def build_dgl_linux(dev) {
  init_git()
  sh "sudo apt-get install -y ssh rsync"
  sh "pip install ansible" // Should move to docker
  ansiblePlaybook(
    playbook: './build.yml',
    inventory: './host.ini',
    credentialsId: 'remotebuild',
    extraVars: [src_dir: "${env.GIT_COMMIT}_${dev}", dev: "${dev}"])
  // sh "bash tests/scripts/build_dgl.sh ${dev}"
  pack_lib("dgl-${dev}-linux", dgl_linux_libs)
}

def build_dgl_win64(dev) {
  /* Assuming that Windows slaves are already configured with MSBuild VS2017,
   * CMake and Python/pip/setuptools etc. */
  init_git_win64()
  bat "CALL tests\\scripts\\build_dgl.bat"
  pack_lib("dgl-${dev}-win64", dgl_win64_libs)
}

def cpp_unit_test_linux() {
  init_git()
  unpack_lib("dgl-cpu-linux", dgl_linux_libs)
  sh "bash tests/scripts/task_cpp_unit_test.sh"
}

def cpp_unit_test_win64() {
  init_git_win64()
  unpack_lib("dgl-cpu-win64", dgl_win64_libs)
  bat "CALL tests\\scripts\\task_cpp_unit_test.bat"
}

def unit_test_linux(backend, dev) {
  init_git()
  unpack_lib("dgl-${dev}-linux", dgl_linux_libs)
  timeout(time: 2, unit: 'MINUTES') {
    sh "bash tests/scripts/task_unit_test.sh ${backend} ${dev}"
  }
}

def unit_test_win64(backend, dev) {
  init_git_win64()
  unpack_lib("dgl-${dev}-win64", dgl_win64_libs)
  timeout(time: 2, unit: 'MINUTES') {
    bat "CALL tests\\scripts\\task_unit_test.bat ${backend}"
  }
}

def example_test_linux(backend, dev) {
  init_git()
  unpack_lib("dgl-${dev}-linux", dgl_linux_libs)
  timeout(time: 20, unit: 'MINUTES') {
    sh "bash tests/scripts/task_example_test.sh ${dev}"
  }
}

def example_test_win64(backend, dev) {
  init_git_win64()
  unpack_lib("dgl-${dev}-win64", dgl_win64_libs)
  timeout(time: 20, unit: 'MINUTES') {
    bat "CALL tests\\scripts\\task_example_test.bat ${dev}"
  }
}

def tutorial_test_linux(backend) {
  init_git()
  unpack_lib("dgl-cpu-linux", dgl_linux_libs)
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
        stage("CPU Build (Win64)") {
          // Windows build machines are manually added to Jenkins master with
          // "windows" label as permanent agents.
          agent { label "windows" }
          steps {
            build_dgl_win64("cpu")
          }
        }
        // Currently we don't have Windows GPU build machines
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
        stage("C++ CPU (Win64)") {
          agent { label "windows" }
          steps {
            cpp_unit_test_win64()
          }
        }
        stage("Torch CPU") {
          agent { docker { image "dgllib/dgl-ci-cpu" } }
          stages {
            stage("Unit test") {
              steps {
                unit_test_linux("pytorch", "cpu")
              }
            }
            stage("Example test") {
              steps {
                example_test_linux("pytorch", "cpu")
              }
            }
            stage("Tutorial test") {
              steps {
                tutorial_test_linux("pytorch")
              }
            }
          }
        }
        stage("Torch CPU (Win64)") {
          agent { label "windows" }
          stages {
            stage("Unit test") {
              steps {
                unit_test_win64("pytorch", "cpu")
              }
            }
            stage("Example test") {
              steps {
                example_test_win64("pytorch", "cpu")
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
                sh "nvidia-smi"
                unit_test_linux("pytorch", "gpu")
              }
            }
            stage("Example test") {
              steps {
                example_test_linux("pytorch", "gpu")
              }
            }
          }
        }
        stage("MXNet CPU") {
          agent { docker { image "dgllib/dgl-ci-cpu" } }
          stages {
            stage("Unit test") {
              steps {
                unit_test_linux("mxnet", "cpu")
              }
            }
            //stage("Example test") {
            //  steps {
            //    unit_test_linux("pytorch", "cpu")
            //  }
            //}
            //stage("Tutorial test") {
            //  steps {
            //    tutorial_test_linux("mxnet")
            //  }
            //}
          }
        }
        stage("MXNet GPU") {
          agent {
            docker {
              image "dgllib/dgl-ci-gpu"
              args "--runtime nvidia"
            }
          }
          stages {
            stage("Unit test") {
              steps {
                sh "nvidia-smi"
                unit_test_linux("mxnet", "gpu")
              }
            }
            //stage("Example test") {
            //  steps {
            //    unit_test_linux("pytorch", "cpu")
            //  }
            //}
            //stage("Tutorial test") {
            //  steps {
            //    tutorial_test_linux("mxnet")
            //  }
            //}
          }
        }
      }
    }
  }
}
