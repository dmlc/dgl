#!/usr/bin/env groovy

// CI tests are executed within Docker containers as the 'root' user. However,
// communications between Jenkins nodes are done with the 'ubuntu' user(login
// via root is disallowed on AWS EC2 instances). Therefore, we need to change
// the file permission to allow 'ubuntu' user to access the files created by
// the 'root' user. This is achieved by running 'chmod -R 777 .'.

// Summary of Jenkins nodes:
// - linux-benchmark-node: Linux CPU node for authentication and lint check.
//      number of nodes: 1
//      instance type: m5.2xlarge(8 vCPUs, 32 GB memory)
//      number of executors per node: 6
//      number of jobs running on this node per CI run: 3
// - dgl-ci-linux-cpu: Linux CPU node for building and testing.
//      number of nodes: 4
//      instance type: m6i.24xlarge(96 vCPUs, 384 GB memory)
//      number of executors per node: 6
//      number of jobs running on this node per CI run: 8
// - dgl-ci-linux-gpu: Linux GPU node for building and testing.
//      number of nodes: 4
//      instance type: g4dn.4xlarge(16 vCPUs, 64 GB memory, 1 GPU)
//      number of executors per node: 1
//      number of jobs running on this node per CI run: 4
// - dgl-ci-windows-cpu: Windows CPU node for building and testing.
//      number of nodes: 4
//      instance type: m6i.8xlarge(32 vCPUs, 128 GB memory)
//      number of executors per node: 2
//      number of jobs running on this node per CI run: 3

dgl_linux_libs = 'build/libdgl.so, build/runUnitTests, python/dgl/_ffi/_cy3/core.cpython-*-x86_64-linux-gnu.so, build/tensoradapter/pytorch/*.so, build/dgl_sparse/*.so, build/graphbolt/*.so'
// Currently DGL on Windows is not working with Cython yet
dgl_win64_libs = "build\\dgl.dll, build\\runUnitTests.exe, build\\tensoradapter\\pytorch\\*.dll, build\\dgl_sparse\\*.dll, build\\graphbolt\\*.dll"

def init_git() {
  sh "chmod -R 777 ." // Fix permission issue
  sh 'rm -rf *'
  sh "git config --global --add safe.directory '*'"
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
  timeout(time: 40, unit: 'MINUTES') {
    sh "bash tests/scripts/task_unit_test.sh ${backend} ${dev}"
  }
}

def unit_distributed_linux(backend, dev) {
  init_git()
  unpack_lib("dgl-${dev}-linux", dgl_linux_libs)
  timeout(time: 40, unit: 'MINUTES') {
    sh "bash tests/scripts/task_distributed_test.sh ${backend} ${dev}"
  }
}

def unit_test_cugraph(backend, dev) {
  init_git()
  unpack_lib("dgl-${dev}-linux", dgl_linux_libs)
  timeout(time: 15, unit: 'MINUTES') {
    sh "bash tests/scripts/cugraph_unit_test.sh ${backend}"
  }
}

def unit_test_win64(backend, dev) {
  init_git_win64()
  unpack_lib("dgl-${dev}-win64", dgl_win64_libs)
  timeout(time: 50, unit: 'MINUTES') {
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

def go_test_linux() {
  init_git()
  unpack_lib('dgl-cpu-linux', dgl_linux_libs)
  timeout(time: 20, unit: 'MINUTES') {
    sh "bash tests/scripts/task_go_test.sh"
  }
}

def is_authorized(name) {
  def devs = [
    // System:
    'dgl-bot', 'noreply',
    // Core:
    'Rhett-Ying', 'BarclayII', 'jermainewang', 'mufeili', 'isratnisa',
    'rudongyu', 'classicsong', 'HuXiangkun', 'hetong007', 'kylasa',
    'frozenbugs', 'peizhou001', 'zheng-da', 'czkkkkkk', 'thvasilo',
    // Intern:
    'pyynb', 'az15240', 'BowenYao18', 'kec020', 'Liu-rj',
    // Friends:
    'nv-dlasalle', 'yaox12', 'chang-l', 'Kh4L', 'VibhuJawa', 'kkranen',
    'TristonC', 'mfbalin',
    'bgawrych', 'itaraban', 'daniil-sizov', 'anko-intel', 'Kacper-Pietkun',
    'hankaj', 'agrabows', 'DominikaJedynak', 'RafLit', 'CfromBU',
    // Emeritus:
    'VoVAllen',
  ]
  return (name in devs)
}

def is_admin(name) {
  def admins = ['dgl-bot', 'Rhett-Ying', 'BarclayII', 'jermainewang']
  return (name in admins)
}

def regression_test_done = false

pipeline {
  agent any
  triggers {
        issueCommentTrigger('@dgl-bot.*')
  }
  stages {
    // Below 2 stages are to authenticate the change/comment author.
    // Only core developers are allowed to trigger CI.
    // Such authentication protects CI from malicious code which may bring CI instances down.
    stage('Authentication') {
      agent {
        docker {
            label 'linux-benchmark-node'
            image 'dgllib/dgl-ci-lint'
            alwaysPull true
        }
      }
      when { not { triggeredBy 'IssueCommentCause' } }
      steps {
        script {
          def author = env.CHANGE_AUTHOR
          def prOpenTriggerCause = currentBuild.getBuildCauses('jenkins.branch.BranchEventCause')
          def first_run = prOpenTriggerCause && env.BUILD_ID == '1'
          if (author && !is_authorized(author)) {
            pullRequest.comment("Not authorized to trigger CI. Please ask core developer to help trigger via issuing comment: \n - `@dgl-bot`")
            error("Authentication failed.")
          }
          if (first_run) {
            pullRequest.comment('To trigger regression tests: \n - `@dgl-bot run [instance-type] [which tests] [compare-with-branch]`; \n For example: `@dgl-bot run g4dn.4xlarge all dmlc/master` or `@dgl-bot run c5.9xlarge kernel,api dmlc/master`')
          }
        }
      }
    }
    stage('AuthenticationComment') {
      agent {
        docker {
            label 'linux-benchmark-node'
            image 'dgllib/dgl-ci-lint'
            alwaysPull true
        }
      }
      when { triggeredBy 'IssueCommentCause' }
      steps {
        script {
          def author = env.GITHUB_COMMENT_AUTHOR
          if (!is_authorized(author)) {
            pullRequest.comment("Not authorized to trigger CI via issuing comment.")
            error("Authentication failed.")
          }
        }
      }
    }
    stage('Regression Test') {
      agent {
        docker {
            label 'linux-benchmark-node'
            image 'dgllib/dgl-ci-lint'
            alwaysPull true
        }
      }
      when { triggeredBy 'IssueCommentCause' }
      steps {
          checkout scm
          script {
              def comment = env.GITHUB_COMMENT
              def command_lists = comment.split(' ')
              if (command_lists.size() == 1) {
                // CI command, not for regression
                return
              }
              if (command_lists.size() != 5) {
                pullRequest.comment('Cannot run the regression test due to unknown command')
                error('Unknown command')
              }
              def author = env.GITHUB_COMMENT_AUTHOR
              echo("${env.GIT_URL}")
              echo("${env}")
              if (!is_admin(author)) {
                error('Not authorized to launch regression tests')
              }
              dir('benchmark_scripts_repo') {
                checkout([$class: 'GitSCM', branches: [[name: '*/master']],
                        userRemoteConfigs: [[credentialsId: 'github', url: 'https://github.com/dglai/DGL_scripts.git']]])
              }
              sh('cp benchmark_scripts_repo/benchmark/* benchmarks/scripts/')
              def instance_type = command_lists[2].replace('.', '')
              pullRequest.comment("Start the Regression test. View at ${RUN_DISPLAY_URL}")
              def prNumber = env.BRANCH_NAME.replace('PR-', '')
              dir('benchmarks/scripts') {
                sh('python3 -m pip install boto3')
                sh("PYTHONUNBUFFERED=1 GIT_PR_ID=${prNumber} GIT_URL=${env.GIT_URL} GIT_BRANCH=${env.CHANGE_BRANCH} python3 run_reg_test.py --data-folder ${env.GIT_COMMIT}_${instance_type} --run-cmd '${comment}'")
              }
              pullRequest.comment("Finished the Regression test. Result table is at https://dgl-asv-data.s3-us-west-2.amazonaws.com/${env.GIT_COMMIT}_${instance_type}/results/result.csv. Jenkins job link is ${RUN_DISPLAY_URL}. ")
              currentBuild.result = 'SUCCESS'
              regression_test_done = true
          }
      }
    }
    stage('CI') {
      when { expression { !regression_test_done } }
      stages {
        stage('Abort Previous CI') {
          steps {
            script {
              if (env.BRANCH_NAME != "master") {
                // Jenkins will abort an older build if a newer build already
                // passed a higher milestone.
                // https://www.jenkins.io/doc/pipeline/steps/pipeline-milestone-step/
                def buildNumber = env.BUILD_NUMBER as int
                for (int i = 1; i <= buildNumber; i++) {
                  milestone(i)
                }
              }
            }
          }
        }

        stage('Lint Check') {
          agent {
            docker {
              label "linux-benchmark-node"
              image "dgllib/dgl-ci-lint"
              alwaysPull true
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
                docker {
                  label "dgl-ci-linux-cpu"
                  image "dgllib/dgl-ci-cpu:v240511_1440"
                  args "-u root"
                  alwaysPull true
                }
              }
              steps {
                build_dgl_linux('cpu')
              }
              post {
                always {
                  sh "chmod -R 777 ." // Fix permission issue
                  cleanWs disableDeferredWipeout: true, deleteDirs: true
                }
              }
            }
            stage('GPU Build') {
              agent {
                docker {
                  label "dgl-ci-linux-cpu"
                  image "dgllib/dgl-ci-gpu:cu121_v240511_1440"
                  args "-u root"
                  alwaysPull true
                }
              }
              steps {
                // sh "nvidia-smi"
                build_dgl_linux('gpu')
              }
              post {
                always {
                  sh "chmod -R 777 ." // Fix permission issue
                  cleanWs disableDeferredWipeout: true, deleteDirs: true
                }
              }
            }
            stage('PyTorch Cugraph GPU Build') {
              agent {
                docker {
                  label "dgl-ci-linux-cpu"
                  image "rapidsai/cugraph_stable_torch-cuda:11.8-base-ubuntu20.04-py3.10-pytorch2.0.0-rapids23.04"
                  args "-u root"
                  alwaysPull true
                }
              }
              steps {
                build_dgl_linux('cugraph')
              }
              post {
                always {
                  sh "chmod -R 777 ." // Fix permission issue
                  cleanWs disableDeferredWipeout: true, deleteDirs: true
                }
              }
            }
            stage('CPU Build (Win64)') {
              agent { label 'dgl-ci-windows-cpu' }
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
                docker {
                  label "dgl-ci-linux-cpu"
                  image "dgllib/dgl-ci-cpu:v240511_1440"
                  args "-u root"
                  alwaysPull true
                }
              }
              steps {
                cpp_unit_test_linux('cpu')
              }
              post {
                always {
                  sh "chmod -R 777 ." // Fix permission issue
                  cleanWs disableDeferredWipeout: true, deleteDirs: true
                }
              }
            }
            stage('C++ GPU') {
              agent {
                docker {
                  label "dgl-ci-linux-gpu"
                  image "dgllib/dgl-ci-gpu:cu121_v240511_1440"
                  args "-u root --runtime nvidia"
                  alwaysPull true
                }
              }
              steps {
                cpp_unit_test_linux('gpu')
              }
              post {
                always {
                  sh "chmod -R 777 ." // Fix permission issue
                  cleanWs disableDeferredWipeout: true, deleteDirs: true
                }
              }
            }
            stage('C++ CPU (Win64)') {
              agent { label 'dgl-ci-windows-cpu' }
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
                docker {
                  label "dgl-ci-linux-cpu"
                  image "dgllib/dgl-ci-cpu:v230810"
                  args "-u root"
                  alwaysPull true
                }
              }
              stages {
                stage('Tensorflow CPU Unit test') {
                  steps {
                    unit_test_linux('tensorflow', 'cpu')
                  }
                  // Tensorflow is deprecated.
                  when { expression { false } }
                }
              }
              post {
                always {
                  sh "chmod -R 777 ." // Fix permission issue
                  cleanWs disableDeferredWipeout: true, deleteDirs: true
                }
              }
            }
            stage('Tensorflow GPU') {
              agent {
                docker {
                  label "dgl-ci-linux-gpu"
                  image "dgllib/dgl-ci-gpu:cu121_v240511_1440"
                  args "-u root --runtime nvidia"
                  alwaysPull true
                }
              }
              stages {
                stage('Tensorflow GPU Unit test') {
                  steps {
                    unit_test_linux('tensorflow', 'gpu')
                  }
                  // Tensorflow does not support cuda 11.6 yet.
                  when { expression { false } }
                }
              }
              post {
                always {
                  sh "chmod -R 777 ." // Fix permission issue
                  cleanWs disableDeferredWipeout: true, deleteDirs: true
                }
              }
            }
            stage('Torch CPU') {
              agent {
                docker {
                  label "dgl-ci-linux-cpu"
                  image "dgllib/dgl-ci-cpu:v240511_1440"
                  args "-u root --shm-size=4gb"
                  alwaysPull true
                }
              }
              stages {
                stage('Torch CPU Unit test') {
                  steps {
                    unit_test_linux('pytorch', 'cpu')
                  }
                }
                stage('Torch CPU Example test') {
                  steps {
                    example_test_linux('pytorch', 'cpu')
                  }
                }
                stage('Torch CPU Tutorial test') {
                  steps {
                    tutorial_test_linux('pytorch')
                  }
                }
              }
              post {
                always {
                  sh "chmod -R 777 ." // Fix permission issue
                  cleanWs disableDeferredWipeout: true, deleteDirs: true
                }
              }
            }
            stage('Torch CPU (Win64)') {
              agent { label 'dgl-ci-windows-cpu' }
              stages {
                stage('Torch CPU (Win64) Unit test') {
                  steps {
                    unit_test_win64('pytorch', 'cpu')
                  }
                }
                stage('Torch CPU (Win64) Example test') {
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
                docker {
                  label "dgl-ci-linux-gpu"
                  image "dgllib/dgl-ci-gpu:cu121_v240511_1440"
                  args "-u root --runtime nvidia --shm-size=8gb"
                  alwaysPull true
                }
              }
              stages {
                stage('Torch GPU Unit test') {
                  steps {
                    sh 'nvidia-smi'
                    unit_test_linux('pytorch', 'gpu')
                  }
                }
                stage('Torch GPU Example test') {
                  steps {
                    example_test_linux('pytorch', 'gpu')
                  }
                }
              }
              post {
                always {
                  sh "chmod -R 777 ." // Fix permission issue
                  cleanWs disableDeferredWipeout: true, deleteDirs: true
                }
              }
            }
            stage('Distributed') {
              agent {
                docker {
                  label "dgl-ci-linux-cpu"
                  image "dgllib/dgl-ci-cpu:v240511_1440"
                  args "-u root --shm-size=8gb"
                  alwaysPull true
                }
              }
              stages {
                stage('Distributed Torch CPU Unit test') {
                  steps {
                    unit_distributed_linux('pytorch', 'cpu')
                  }
                }
              }
              post {
                always {
                  sh "chmod -R 777 ." // Fix permission issue
                  cleanWs disableDeferredWipeout: true, deleteDirs: true
                }
              }
            }
            stage('PyTorch Cugraph GPU') {
              agent {
                docker {
                  label "dgl-ci-linux-gpu"
                  image "rapidsai/cugraph_stable_torch-cuda:11.8-base-ubuntu20.04-py3.10-pytorch2.0.0-rapids23.04"
                  args "-u root --runtime nvidia --shm-size=8gb"
                  alwaysPull true
                }
              }
              stages {
                stage('PyTorch Cugraph GPU Unit test') {
                  steps {
                    sh 'nvidia-smi'
                    unit_test_cugraph('pytorch', 'cugraph')
                  }
                  // Cugraph is under refactoring. Skip the test for now.
                  when { expression { false } }
                }
              }
              post {
                always {
                  sh "chmod -R 777 ." // Fix permission issue
                  cleanWs disableDeferredWipeout: true, deleteDirs: true
                }
              }
            }
            stage('DGL-Go') {
              agent {
                docker {
                  label "dgl-ci-linux-cpu"
                  image "dgllib/dgl-ci-cpu:v240511_1440"
                  args "-u root"
                  alwaysPull true
                }
              }
              stages {
                stage('DGL-Go CPU test') {
                  steps {
                    go_test_linux()
                  }
                }
              }
              post {
                always {
                  sh "chmod -R 777 ." // Fix permission issue
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
      script {
        node("dglci-post-linux") {
          docker.image('dgllib/dgl-ci-awscli:v220418').inside("--pull always --entrypoint=''") {
            sh("rm -rf ci_tmp")
            dir('ci_tmp') {
              sh("curl -k -o cireport.log ${BUILD_URL}consoleText")
              sh("curl -o report.py https://raw.githubusercontent.com/dmlc/dgl/master/tests/scripts/ci_report/report.py")
              sh("curl -o status.py https://raw.githubusercontent.com/dmlc/dgl/master/tests/scripts/ci_report/status.py")
              sh("curl -k -L ${BUILD_URL}wfapi")
              sh("cat status.py")
              sh("pytest --html=report.html --self-contained-html report.py || true")
              sh("aws s3 sync ./ s3://dgl-ci-result/${JOB_NAME}/${BUILD_NUMBER}/${BUILD_ID}/logs/  --exclude '*' --include '*.log' --acl public-read --content-type text/plain")
              sh("aws s3 sync ./ s3://dgl-ci-result/${JOB_NAME}/${BUILD_NUMBER}/${BUILD_ID}/logs/  --exclude '*.log' --acl public-read")

              def comment = sh(returnStdout: true, script: "python3 status.py --result ${currentBuild.currentResult}").trim()
              echo(comment)
              if ((env.BRANCH_NAME).startsWith('PR-')) {
                pullRequest.comment(comment)
              }
            }
          }
        }
        node('dgl-ci-windows-cpu') {
            bat(script: "rmvirtualenv ${BUILD_TAG}", returnStatus: true)
        }
      }
    }
  }
}
