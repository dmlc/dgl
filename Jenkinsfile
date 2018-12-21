#!/usr/bin/env groovy

def init_git_submodule() {
  sh "git submodule init"
  sh "git submodule update"
}

def init_git_submodule_win64() {
  bat "git submodule init"
  bat "git submodule update"
}

def setup() {
  init_git_submodule()
}

def setup_win64() {
  init_git_submodule_win64()
}

def build_dgl() {
  sh "if [ -d build ]; then rm -rf build; fi; mkdir build"
  sh "rm -rf _download"
  dir ("build") {
    sh "cmake .."
    sh "make -j4"
  }
  dir("python") {
    sh "rm -rf build *.egg-info dist"
    sh "pip3 uninstall -y dgl"
    sh "python3 setup.py install"
  }
}

def build_dgl_win64() {
  /* Assuming that Windows slaves are already configured with MSBuild VS2017,
   * CMake and Python/pip/setuptools etc. */
  bat "DEL /S /Q build"
  bat "DEL /S /Q _download"
  dir ("build") {
    bat """
    CALL "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat"
    cmake -DCMAKE_CXX_FLAGS="/DDGL_EXPORTS" -DCMAKE_CONFIGURATION_TYPES="Release" .. -G "Visual Studio 15 2017 Win64"
    msbuild dgl.sln
    COPY Release\\dgl.dll .
    """
  }
  dir ("python") {
    bat "DEL /S /Q build *.egg-info dist"
    bat "pip3 uninstall -y dgl || (call )"  // ignore error
    bat "python setup.py install --user"
  }
}

def pytorch_unit_test(dev) {
  withEnv(["DGL_LIBRARY_PATH=${env.WORKSPACE}/build", "PYTHONPATH=${env.WORKSPACE}/python"]) {
    sh "python3 -m nose -v --with-xunit tests"
    sh "python3 -m nose -v --with-xunit tests/pytorch"
    sh "python3 -m nose -v --with-xunit tests/graph_index"
  }
}

def pytorch_unit_test_win64(dev) {
  withEnv(["DGL_LIBRARY_PATH=${env.WORKSPACE}\\build", "PYTHONPATH=${env.WORKSPACE}\\python"]) {
    bat "python -m nose -v --with-xunit tests"
    bat "python -m nose -v --with-xunit tests\\pytorch"
    bat "python -m nose -v --with-xunit tests\\graph_index"
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

def example_test_win64(dev) {
  withEnv(["DGL_LIBRARY_PATH=${env.WORKSPACE}\\build", "PYTHONPATH=${env.WORKSPACE}\\python"]) {
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
            setup()
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
            setup()
            build_dgl()
          }
        }
        stage("MXNet CPU Build (temp)") {
          agent {
            docker { image "dgllib/dgl-ci-mxnet-cpu" }
          }
          steps {
            setup()
            build_dgl()
          }
        }
	stage("CPU Build (Win64/PyTorch)") {
	  agent {
            label "windows"
          }
	  steps {
	    setup_win64()
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
        stage("Pytorch CPU (Windows)") {
          agent { label "windows" }
          stages {
            stage("TH CPU Win64 unittest") {
              steps { pytorch_unit_test_win64("CPU") }
            }
	    stage("TH CPU Win64 example test") {
	      steps { example_test_win64("CPU") }
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
            docker { image "dgllib/dgl-ci-mxnet-cpu" }
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
