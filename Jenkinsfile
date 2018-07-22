void setBuildStatus(String message, String state) {
  step([
      $class: "GitHubCommitStatusSetter",
      reposSource: [$class: "ManuallyEnteredRepositorySource", url: "https://github.com/jermainewang/dgl-1"],
      contextSource: [$class: "ManuallyEnteredCommitContextSource", context: "ci/jenkins/build-status"],
      errorHandlers: [[$class: "ChangingBuildStatusErrorHandler", result: "UNSTABLE"]],
      statusResultSource: [ $class: "ConditionalStatusResultSource", results: [[$class: "AnyBuildResult", message: message, state: state]] ]
  ]);
}

pipeline {
    agent {
        docker {
            image 'pytorch/pytorch'
        }
    }
    stages {
        stage('SETUP') {
            steps {
                sh 'easy_install nose'
            }
        }
        stage('BUILD') {
            steps {
                dir('python') {
                    sh 'python setup.py install'
                }
            }
        }
        stage('TEST') {
            steps {
                sh 'nosetests tests -v --with-xunit'
            }
        }
    }
    post {
        always {
            junit '*.xml'
        }
        success {
            setBuildStatus("Build succeeded", "SUCCESS");
        }
        failure {
            setBuildStatus("Build failed", "FAILURE");
        }
    }
}
