#!/usr/bin/env groovy

dgl_linux_libs = 'build/libdgl.so, build/runUnitTests, python/dgl/_ffi/_cy3/core.cpython-36m-x86_64-linux-gnu.so, build/tensoradapter/pytorch/*.so'
// Currently DGL on Windows is not working with Cython yet
dgl_win64_libs = "build\\dgl.dll, build\\runUnitTests.exe, build\\tensoradapter\\pytorch\\*.dll"

def init_git() {
  sh 'rm -rf *'
  checkout scm
  sh 'git submodule update --recursive --init'
}

def init_git_win64() {
  checkout scm
  bat 'git submodule update --recursive --init'
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
  sh "bash tests/scripts/build_dgl.sh ${dev}"
  sh 'ls -lh /usr/lib/x86_64-linux-gnu/'
  pack_lib("dgl-${dev}-linux", dgl_linux_libs)
}

def build_dgl_win64(dev) {
  /* Assuming that Windows slaves are already configured with MSBuild VS2017,
   * CMake and Python/pip/setuptools etc. */
  init_git_win64()
  bat "CALL tests\\scripts\\build_dgl.bat"
  pack_lib("dgl-${dev}-win64", dgl_win64_libs)
}

def cpp_unit_test_linux(dev) {
  init_git()
  unpack_lib("dgl-${dev}-linux", dgl_linux_libs)
  sh 'bash tests/scripts/task_cpp_unit_test.sh'
}

def cpp_unit_test_win64() {
  init_git_win64()
  unpack_lib('dgl-cpu-win64', dgl_win64_libs)
  bat "CALL tests\\scripts\\task_cpp_unit_test.bat"
}

def unit_test_linux(backend, dev) {
  init_git()
  unpack_lib("dgl-${dev}-linux", dgl_linux_libs)
  timeout(time: 30, unit: 'MINUTES') {
    sh "bash tests/scripts/task_unit_test.sh ${backend} ${dev}"
  }
}

def unit_test_win64(backend, dev) {
  init_git_win64()
  unpack_lib("dgl-${dev}-win64", dgl_win64_libs)
  timeout(time: 20, unit: 'MINUTES') {
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
  unpack_lib('dgl-cpu-linux', dgl_linux_libs)
  timeout(time: 20, unit: 'MINUTES') {
    sh "bash tests/scripts/task_${backend}_tutorial_test.sh"
  }
}

def is_authorized(name) {
  def authorized_user = ['VoVAllen', 'BarclayII', 'jermainewang', 'zheng-da', 'mufeili', 'Rhett-Ying', 'isratnisa']
  return (name in authorized_user)
}

pipeline {
  agent any
  triggers {
        issueCommentTrigger('@dgl-bot .*')
  }
  stages {
    stage('Regression Test Trigger') {
      agent {
        kubernetes {
          yamlFile 'docker/pods/ci-lint.yaml'
          defaultContainer 'dgl-ci-lint'
        }
      }
      when { triggeredBy 'IssueCommentCause' }
      steps {
        // container('dgl-ci-lint') {
          checkout scm
          script {
              def comment = env.GITHUB_COMMENT
              def author = env.GITHUB_COMMENT_AUTHOR
              echo("${env.GIT_URL}")
              echo("${env}")
              if (!is_authorized(author)) {
                error('Not authorized to launch regression tests')
              }
              dir('benchmark_scripts_repo') {
                checkout([$class: 'GitSCM', branches: [[name: '*/master']],
                        userRemoteConfigs: [[credentialsId: 'github', url: 'https://github.com/dglai/DGL_scripts.git']]])
              }
              sh('cp benchmark_scripts_repo/benchmark/* benchmarks/scripts/')
              def command_lists = comment.split(' ')
              def instance_type = command_lists[2].replace('.', '')
              if (command_lists.size() != 5) {
              pullRequest.comment('Cannot run the regression test due to unknown command')
              error('Unknown command')
              } else {
              pullRequest.comment("Start the Regression test. View at ${RUN_DISPLAY_URL}")
              }
              def prNumber = env.BRANCH_NAME.replace('PR-', '')
              dir('benchmarks/scripts') {
                sh('python3 -m pip install boto3')
                sh("PYTHONUNBUFFERED=1 GIT_PR_ID=${prNumber} GIT_URL=${env.GIT_URL} GIT_BRANCH=${env.CHANGE_BRANCH} python3 run_reg_test.py --data-folder ${env.GIT_COMMIT}_${instance_type} --run-cmd '${comment}'")
              }
              pullRequest.comment("Finished the Regression test. Result table is at https://dgl-asv-data.s3-us-west-2.amazonaws.com/${env.GIT_COMMIT}_${instance_type}/results/result.csv. Jenkins job link is ${RUN_DISPLAY_URL}. ")
              currentBuild.result = 'SUCCESS'
              return
          }
        // }
      }
    }
    stage('Bot Instruction') {
      agent {
        kubernetes {
          yamlFile 'docker/pods/ci-lint.yaml'
          defaultContainer 'dgl-ci-lint'
        }
      }
      steps {
        script {
          def prOpenTriggerCause = currentBuild.getBuildCauses('jenkins.branch.BranchEventCause')
          if (prOpenTriggerCause) {
            if (env.BUILD_ID == '1') {
              pullRequest.comment('To trigger regression tests: \n - `@dgl-bot run [instance-type] [which tests] [compare-with-branch]`; \n For example: `@dgl-bot run g4dn.4xlarge all dmlc/master` or `@dgl-bot run c5.9xlarge kernel,api dmlc/master`')
            }
          }
          echo('Not the first build')
        }
      }
    }
    stage('CI') {
      when { not { triggeredBy 'IssueCommentCause' } }
      stages {
        stage('Lint Check') {
          agent {
            kubernetes {
              yamlFile 'docker/pods/ci-lint.yaml'
              defaultContainer 'dgl-ci-lint'
            }
          }
          steps {
            init_git()
            sh 'bash tests/scripts/task_lint.sh'
          }
          post {
            always {
              cleanWs disableDeferredWipeout: true, deleteDirs: true
            }
          }
        }
        
        stage('Build') {
          parallel {
            stage('CPU Build') {
              agent {
                kubernetes {
                  yamlFile 'docker/pods/ci-compile-cpu.yaml'
                  defaultContainer 'dgl-ci-cpu-compile'
                }
              }
              steps {
                build_dgl_linux('cpu')
              }
              post {
                always {
                  cleanWs disableDeferredWipeout: true, deleteDirs: true
                }
              }
            }
            stage('GPU Build') {
              agent {
                kubernetes {
                  yamlFile 'docker/pods/ci-compile-gpu.yaml'
                  defaultContainer 'dgl-ci-gpu-compile'
                }
              }
              steps {
                // sh "nvidia-smi"
                build_dgl_linux('gpu')
              }
              post {
                always {
                  cleanWs disableDeferredWipeout: true, deleteDirs: true
                }
              }
            }
            stage('CPU Build (Win64)') {
              // Windows build machines are manually added to Jenkins master with
              // "windows" label as permanent agents.
              agent { label 'windows' }
              steps {
                build_dgl_win64('cpu')
              }
              post {
                always {
                  cleanWs disableDeferredWipeout: true, deleteDirs: true
                }
              }
            }
          // Currently we don't have Windows GPU build machines
          }
        }
        stage('Test') {
          parallel {
            stage('C++ CPU') {
              agent {
                kubernetes {
                  yamlFile 'docker/pods/ci-cpu.yaml'
                  defaultContainer 'dgl-ci-cpu'
                }
              }
              steps {
                cpp_unit_test_linux('cpu')
              }
              post {
                always {
                  cleanWs disableDeferredWipeout: true, deleteDirs: true
                }
              }
            }
            stage('C++ GPU') {
              agent {
                kubernetes {
                  yamlFile 'docker/pods/ci-gpu.yaml'
                  defaultContainer 'dgl-ci-gpu'
                }
              }
              steps {
                cpp_unit_test_linux('gpu')
              }
              post {
                always {
                  cleanWs disableDeferredWipeout: true, deleteDirs: true
                }
              }
            }
            stage('C++ CPU (Win64)') {
              agent { label 'windows' }
              steps {
                cpp_unit_test_win64()
              }
              post {
                always {
                  cleanWs disableDeferredWipeout: true, deleteDirs: true
                }
              }
            }
            stage('Tensorflow CPU') {
              agent {
                kubernetes {
                  yamlFile 'docker/pods/ci-cpu.yaml'
                  defaultContainer 'dgl-ci-cpu'
                }
              }
              stages {
                stage('Unit test') {
                  steps {
                    unit_test_linux('tensorflow', 'cpu')
                  }
                }
              }
              post {
                always {
                  cleanWs disableDeferredWipeout: true, deleteDirs: true
                }
              }
            }
            stage('Tensorflow GPU') {
              agent {
                kubernetes {
                  yamlFile 'docker/pods/ci-gpu.yaml'
                  defaultContainer 'dgl-ci-gpu'
                }
              }
              stages {
                stage('Unit test') {
                  steps {
                    unit_test_linux('tensorflow', 'gpu')
                  }
                }
              }
              post {
                always {
                  cleanWs disableDeferredWipeout: true, deleteDirs: true
                }
              }
            }
            stage('Torch CPU') {
              agent {
                kubernetes {
                  yamlFile 'docker/pods/ci-cpu.yaml'
                  defaultContainer 'dgl-ci-cpu'
                }
              }
              stages {
                stage('Unit test') {
                  steps {
                    unit_test_linux('pytorch', 'cpu')
                  }
                }
                stage('Example test') {
                  steps {
                    example_test_linux('pytorch', 'cpu')
                  }
                }
                stage('Tutorial test') {
                  steps {
                    sh 'ls -l /tmp/dataset/*'
                    sh 'ls -l /tmp/dataset/'
                    tutorial_test_linux('pytorch')
                  }
                }
              }
              post {
                always {
                  cleanWs disableDeferredWipeout: true, deleteDirs: true
                }
              }
            }
            stage('Torch CPU (Win64)') {
              agent { label 'windows' }
              stages {
                stage('Unit test') {
                  steps {
                    unit_test_win64('pytorch', 'cpu')
                  }
                }
                stage('Example test') {
                  steps {
                    example_test_win64('pytorch', 'cpu')
                  }
                }
              }
              post {
                always {
                  cleanWs disableDeferredWipeout: true, deleteDirs: true
                }
              }
            }
            stage('Torch GPU') {
              agent {
                kubernetes {
                  yamlFile 'docker/pods/ci-gpu.yaml'
                  defaultContainer 'dgl-ci-gpu'
                }
              }
              stages {
                stage('Unit test') {
                  steps {
                    sh 'nvidia-smi'
                    unit_test_linux('pytorch', 'gpu')
                  }
                }
                stage('Example test') {
                  steps {
                    example_test_linux('pytorch', 'gpu')
                  }
                }
              }
              post {
                always {
                  cleanWs disableDeferredWipeout: true, deleteDirs: true
                }
              }
            }
            stage('MXNet CPU') {
              agent {
                kubernetes {
                  yamlFile 'docker/pods/ci-cpu.yaml'
                  defaultContainer 'dgl-ci-cpu'
                }
              }
              stages {
                stage('Unit test') {
                  steps {
                    unit_test_linux('mxnet', 'cpu')
                  }
                }
              //stage("Tutorial test") {
              //  steps {
              //    tutorial_test_linux("mxnet")
              //  }
              //}
              }
              post {
                always {
                  cleanWs disableDeferredWipeout: true, deleteDirs: true
                }
              }
            }
            stage('MXNet GPU') {
              agent {
                kubernetes {
                  yamlFile 'docker/pods/ci-gpu.yaml'
                  defaultContainer 'dgl-ci-gpu'
                }
              }
              stages {
                stage('Unit test') {
                  steps {
                    sh 'nvidia-smi'
                    unit_test_linux('mxnet', 'gpu')
                  }
                }
              }
              post {
                always {
                  cleanWs disableDeferredWipeout: true, deleteDirs: true
                }
              }
            }
          }
        }
      }
    }
  }
  post {
    always {
      node('windows') {
        bat "rmvirtualenv ${BUILD_TAG}"
      }
    }
  }
}
